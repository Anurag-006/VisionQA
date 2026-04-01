package com.anurag.visionqa.ai

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import ai.onnxruntime.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.FloatBuffer
import java.nio.LongBuffer

class MoondreamVLM(private val context: Context) : VLMInterface {

    private var ortEnv: OrtEnvironment? = null
    private var visionSession: OrtSession? = null
    private var embedSession: OrtSession? = null
    private var decoderSession: OrtSession? = null
    private var tokenizer: RealTokenizer? = null
    private val downloader = MoondreamDownloader(context)
    private var ready = false

    // KV Cache
    private var persistentKVCache: MutableMap<String, OnnxTensor>? = null
    private var lastSessionResult: OrtSession.Result? = null
    private var currentSequenceLength = 0

    // Image feature cache — vision encoder runs ONCE per captured image
    private var cachedImageFeatures: Array<FloatArray>? = null
    private var cachedImageBitmap: Bitmap? = null

    companion object {
        private const val TAG = "MoondreamVLM"
        private const val IMAGE_SIZE = 378
        private const val MAX_NEW_TOKENS = 80
        private const val REPETITION_PENALTY = 1.15f
        private const val TEMPERATURE = 0.0f
        private const val TOP_K = 5
    }

    // Standard initialization required by VLMInterface
    override suspend fun initialize(): Boolean = initialize { _, _ -> }

    // Progress-aware initialization for UI reporting
    override suspend fun initialize(onProgress: (String, Int) -> Unit): Boolean =
        withContext(Dispatchers.IO) {
            try {
                Log.d(TAG, "=== MOONDREAM FALLBACK INIT ===")
                if (!downloader.downloadIfNeeded { msg, prog -> onProgress(msg, prog) })
                    return@withContext false

                tokenizer = RealTokenizer(context, downloader.getTokenizerPath())
                Log.i(TAG, "Tokenizer check: ${tokenizer!!.selfTest()}")

                ortEnv = OrtEnvironment.getEnvironment()

                val bigCores = Runtime.getRuntime().availableProcessors().coerceIn(2, 4)

                val options = OrtSession.SessionOptions().apply {
                    setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                    try {
                        addXnnpack(mapOf("intra_op_num_threads" to bigCores.toString()))
                        Log.d(TAG, "✅ XNNPACK Enabled on $bigCores threads")
                    } catch (e: Exception) {
                        Log.w(TAG, "XNNPACK unavailable, falling back: ${e.message}")
                        setIntraOpNumThreads(bigCores)
                    }
                }
                visionSession  = ortEnv!!.createSession(downloader.getVisionPath(), options)
                embedSession   = ortEnv!!.createSession(downloader.getEmbedPath(), options)
                decoderSession = ortEnv!!.createSession(downloader.getDecoderPath(), options)

                ready = true
                true
            } catch (e: Exception) {
                Log.e(TAG, "Init failed", e)
                false
            }
        }

    override fun isReady(): Boolean = ready

    fun resetChat() {
        lastSessionResult?.close()
        lastSessionResult = null
        persistentKVCache = null
        currentSequenceLength = 0
        cachedImageFeatures = null
        cachedImageBitmap = null
        Log.d(TAG, "🧹 Memory wiped.")
    }

    fun clearTextMemory() {
        lastSessionResult?.close()
        lastSessionResult = null
        persistentKVCache = null
        currentSequenceLength = 0
        Log.d(TAG, "🧹 Text memory wiped. Image features retained.")
    }

    // Unified entry point implementation
    override suspend fun chat(
        image: Bitmap,
        question: String,
        systemPrompt: String?,
        conversationHistory: List<Pair<String, String>>,
        onTokenGenerated: ((String) -> Unit)?
    ): String = chatWithOcr(image, question, null, systemPrompt, onTokenGenerated)

    /**
     * Fallback implementation using ONNX Runtime.
     * Injected OCR text is handled here if available.
     */
    private suspend fun chatWithOcr(
        image: Bitmap,
        question: String,
        ocrText: String?,
        systemPrompt: String? = null,
        onTokenGenerated: ((String) -> Unit)? = null
    ): String = withContext(Dispatchers.IO) {
        if (!ready) return@withContext "Fallback model not initialized"

        if (currentSequenceLength > 1500) {
            Log.w(TAG, "⚠️ Context too long, resetting.")
            resetChat()
        }

        try {
            val lq = question.trim().lowercase()
            val isIdentityQuestion = (lq.contains("who are you") ||
                    lq.contains("what are you") ||
                    (lq.contains("your name") && (lq.contains("what") || lq.contains("tell"))) ||
                    lq == "your name?" || lq == "name?")

            if (isIdentityQuestion) {
                val reply = "I am an AI assistant. I do not have a name."
                onTokenGenerated?.invoke(reply)
                return@withContext reply
            }

            val isFollowUp = persistentKVCache != null
            val prompt = tokenizer!!.formatPrompt(
                question     = question,
                ocrText      = ocrText,
                systemPrompt = systemPrompt,
                isFollowUp   = isFollowUp
            )

            val inputIds = tokenizer!!.encode(prompt)
            if (inputIds.isEmpty()) return@withContext "Tokenizer error — see Logcat."

            val textEmbeds  = embedText(inputIds)
            val finalEmbeds = if (!isFollowUp) {
                val imgEmbeds = getOrEncodeImage(image)
                combineEmbeddings(imgEmbeds, textEmbeds)
            } else {
                textEmbeds
            }

            val response = generateWithKVCache(finalEmbeds, MAX_NEW_TOKENS, onTokenGenerated)
            response.trim().removePrefix("Answer:").removePrefix(":").trim()

        } catch (e: Exception) {
            Log.e(TAG, "chatWithOcr error", e)
            "Error: ${e.message}"
        }
    }

    private fun getOrEncodeImage(bitmap: Bitmap): Array<FloatArray> {
        if (cachedImageBitmap === bitmap && cachedImageFeatures != null) return cachedImageFeatures!!
        val features = encodeImage(bitmap)
        cachedImageFeatures = features
        cachedImageBitmap = bitmap
        return features
    }

    private fun encodeImage(bitmap: Bitmap): Array<FloatArray> {
        val resized = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true)
        val pixels  = IntArray(IMAGE_SIZE * IMAGE_SIZE)
        resized.getPixels(pixels, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE)
        val imageData = FloatArray(3 * IMAGE_SIZE * IMAGE_SIZE)
        for (i in pixels.indices) {
            val p = pixels[i]
            imageData[i]                              = (((p shr 16) and 0xFF) / 255f - 0.5f) / 0.5f
            imageData[IMAGE_SIZE * IMAGE_SIZE + i]    = (((p shr  8) and 0xFF) / 255f - 0.5f) / 0.5f
            imageData[2 * IMAGE_SIZE * IMAGE_SIZE + i]= (( p         and 0xFF) / 255f - 0.5f) / 0.5f
        }
        val t = OnnxTensor.createTensor(
            ortEnv, FloatBuffer.wrap(imageData),
            longArrayOf(1, 3, IMAGE_SIZE.toLong(), IMAGE_SIZE.toLong())
        )
        val out = visionSession!!.run(mapOf("pixel_values" to t))
        val features = ((out[0].value as Array<*>)[0]) as Array<FloatArray>
        t.close(); out.close()
        return features
    }

    private fun embedText(inputIds: IntArray): Array<FloatArray> {
        val longs  = LongArray(inputIds.size) { inputIds[it].toLong() }
        val tensor = OnnxTensor.createTensor(
            ortEnv, LongBuffer.wrap(longs), longArrayOf(1, longs.size.toLong())
        )
        val out    = embedSession!!.run(mapOf("input_ids" to tensor))
        val embeds = ((out[0].value as Array<*>)[0]) as Array<FloatArray>
        tensor.close(); out.close()
        return embeds
    }

    private fun combineEmbeddings(a: Array<FloatArray>, b: Array<FloatArray>) = a + b

    private fun generateWithKVCache(
        embeddings: Array<FloatArray>,
        maxTokens: Int,
        onToken: ((String) -> Unit)?
    ): String {
        val generated  = mutableListOf<Int>()
        val seqLen     = embeddings.size
        val hiddenSize = embeddings[0].size
        val eosId      = tokenizer?.getEosId() ?: 50256
        val numLayers  = 24

        val flat = FloatArray(seqLen * hiddenSize)
        for (i in embeddings.indices)
            for (j in embeddings[i].indices)
                flat[i * hiddenSize + j] = embeddings[i][j]

        val inputs  = mutableMapOf<String, OnnxTensor>()
        val created = mutableListOf<OnnxTensor>()

        fun addTensor(key: String, t: OnnxTensor) { inputs[key] = t; created.add(t) }

        addTensor("inputs_embeds", OnnxTensor.createTensor(
            ortEnv, FloatBuffer.wrap(flat), longArrayOf(1, seqLen.toLong(), hiddenSize.toLong())
        ))

        if (persistentKVCache == null) {
            currentSequenceLength = seqLen
            addTensor("attention_mask", OnnxTensor.createTensor(
                ortEnv, LongBuffer.wrap(LongArray(seqLen) { 1L }), longArrayOf(1, seqLen.toLong())
            ))
            addTensor("position_ids", OnnxTensor.createTensor(
                ortEnv, LongBuffer.wrap(LongArray(seqLen) { it.toLong() }), longArrayOf(1, seqLen.toLong())
            ))
            val empty = FloatArray(0); val es = longArrayOf(1, 32, 0, 64)
            for (l in 0 until numLayers) {
                inputs["past_key_values.$l.key"]   = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(empty), es)
                inputs["past_key_values.$l.value"] = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(empty), es)
            }
        } else {
            val total = currentSequenceLength + seqLen
            addTensor("attention_mask", OnnxTensor.createTensor(
                ortEnv, LongBuffer.wrap(LongArray(total) { 1L }), longArrayOf(1, total.toLong())
            ))
            addTensor("position_ids", OnnxTensor.createTensor(
                ortEnv, LongBuffer.wrap(LongArray(seqLen) { (currentSequenceLength + it).toLong() }),
                longArrayOf(1, seqLen.toLong())
            ))
            inputs.putAll(persistentKVCache!!)
            currentSequenceLength = total
        }

        var prevOut: OrtSession.Result? = null

        for (step in 0 until maxTokens) {
            try {
                val out = decoderSession!!.run(inputs)
                prevOut?.close(); prevOut = out

                val logitsRaw = out[0].value
                val rawLogits: FloatArray = when (logitsRaw) {
                    is Array<*> -> (logitsRaw[0] as Array<*>).last() as FloatArray
                    is FloatArray -> logitsRaw
                    else -> throw Exception("Unexpected logits type")
                }

                val logits = applyRepetitionPenalty(rawLogits, generated, REPETITION_PENALTY)
                val tokenId = sampleWithTemperature(logits, TEMPERATURE)

                if (tokenId == eosId) break

                generated.add(tokenId)
                onToken?.invoke(tokenizer!!.decode(intArrayOf(tokenId)))

                val newKV = mutableMapOf<String, OnnxTensor>()
                for (l in 0 until numLayers) {
                    newKV["past_key_values.$l.key"]   = out.get("present.$l.key").get()   as OnnxTensor
                    newKV["past_key_values.$l.value"] = out.get("present.$l.value").get() as OnnxTensor
                }
                created.forEach { it.close() }; created.clear(); inputs.clear()

                val ne = embedText(intArrayOf(tokenId))
                val nt = OnnxTensor.createTensor(
                    ortEnv, FloatBuffer.wrap(ne[0]), longArrayOf(1, 1, hiddenSize.toLong())
                )
                inputs["inputs_embeds"] = nt; created.add(nt)
                currentSequenceLength++

                val at = OnnxTensor.createTensor(
                    ortEnv, LongBuffer.wrap(LongArray(currentSequenceLength) { 1L }),
                    longArrayOf(1, currentSequenceLength.toLong())
                )
                inputs["attention_mask"] = at; created.add(at)

                val pt = OnnxTensor.createTensor(
                    ortEnv, LongBuffer.wrap(longArrayOf((currentSequenceLength - 1).toLong())),
                    longArrayOf(1, 1)
                )
                inputs["position_ids"] = pt; created.add(pt)
                inputs.putAll(newKV)

            } catch (e: Exception) {
                break
            }
        }

        lastSessionResult?.close(); lastSessionResult = prevOut
        persistentKVCache = mutableMapOf()
        if (lastSessionResult != null) {
            for (l in 0 until numLayers) {
                persistentKVCache!!["past_key_values.$l.key"]   =
                    lastSessionResult!!.get("present.$l.key").get()   as OnnxTensor
                persistentKVCache!!["past_key_values.$l.value"] =
                    lastSessionResult!!.get("present.$l.value").get() as OnnxTensor
            }
        }
        created.forEach { it.close() }
        return tokenizer!!.decode(generated.toIntArray())
    }

    // Matches ModelInfo fields: name, size, type, quality
    override fun getModelInfo(): ModelInfo = ModelInfo(
        name    = "Moondream2 ONNX",
        size    = "~1.5GB",
        speed   = "Slow (CPU)",
        quality = "Basic"
    )

    private fun applyRepetitionPenalty(
        logits: FloatArray,
        generatedTokens: List<Int>,
        penalty: Float
    ): FloatArray {
        if (penalty == 1.0f || generatedTokens.isEmpty()) return logits
        val result = logits.copyOf()
        for (tokenId in generatedTokens.toSet()) {
            if (tokenId < result.size) {
                if (tokenId in 15..57) continue
                result[tokenId] = if (result[tokenId] > 0f) result[tokenId] / penalty else result[tokenId] * penalty
            }
        }
        return result
    }

    private fun sampleWithTemperature(logits: FloatArray, temperature: Float): Int {
        if (temperature <= 0f) return logits.indices.maxByOrNull { logits[it] } ?: 0
        val scaled = FloatArray(logits.size) { logits[it] / temperature }
        val topKIndices = scaled.indices.sortedByDescending { scaled[it] }.take(TOP_K).toSet()
        val filtered = FloatArray(scaled.size) { i -> if (i in topKIndices) scaled[i] else Float.NEGATIVE_INFINITY }
        val maxLogit = filtered.filter { it.isFinite() }.maxOrNull() ?: 0f
        val exps = FloatArray(filtered.size) { i -> if (filtered[i].isFinite()) Math.exp((filtered[i] - maxLogit).toDouble()).toFloat() else 0f }
        val sumExp = exps.sum()
        if (sumExp <= 0f) return logits.indices.maxByOrNull { logits[it] } ?: 0
        val probs = FloatArray(exps.size) { exps[it] / sumExp }
        val r = Math.random().toFloat()
        var cumulative = 0f
        for (i in probs.indices) {
            cumulative += probs[i]
            if (r < cumulative) return i
        }
        return probs.indices.maxByOrNull { probs[it] } ?: 0
    }

    override fun cleanup() {
        resetChat()
        visionSession?.close()
        embedSession?.close()
        decoderSession?.close()
        ortEnv?.close()
        ready = false
    }
}