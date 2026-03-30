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
        private const val MAX_NEW_TOKENS = 80 // Visual-only answers are short; 512 was wasteful
        // Repetition penalty: penalises tokens already generated.
        // 1.0 = off. Small quantized models need a stronger penalty (1.5)
        // because their logit distributions are noisier than full-size models.
        private const val REPETITION_PENALTY = 1.15f

        // Temperature for sampling. Small quantized models hallucinate badly
        // at temperatures above ~0.4. Keep this low — we rely on the repetition
        // penalty (not temperature) to prevent loops.
        private const val TEMPERATURE = 0.0f

        // Top-k: only consider this many candidate tokens per step.
        // 5 is tight enough to prevent hallucinated names on a q4 model
        // while still allowing natural variation in phrasing.
        private const val TOP_K = 5
    }

    suspend fun initialize(onProgress: (String, Int) -> Unit = { _, _ -> }): Boolean =
        withContext(Dispatchers.IO) {
            try {
                Log.d(TAG, "=== MOONDREAM INIT ===")
                if (!downloader.downloadIfNeeded { msg, prog -> onProgress(msg, prog) })
                    return@withContext false

                tokenizer = RealTokenizer(context, downloader.getTokenizerPath())

                // ---- TOKENIZER SANITY CHECK ----
                // In Logcat filter by "MoondreamVLM" and look for:
                //   selfTest: encode('Hello world')=[15496, 995] decoded='Hello world'
                // If you see [0,0] or empty list → tokenizer.json didn't load properly.
                Log.i(TAG, "Tokenizer check: ${tokenizer!!.selfTest()}")

                ortEnv = OrtEnvironment.getEnvironment()

//                val options = OrtSession.SessionOptions().apply {
//                    setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
//                    try {
//                        // 1. Try Hardware Acceleration (NNAPI) first!
//                        addNnapi()
//                        Log.d(TAG, "✅ NNAPI (Hardware Acceleration) Enabled!")
//                    } catch (e: Exception) {
//                        Log.w(TAG, "NNAPI unavailable, falling back to CPU: ${e.message}")
//                        // 2. Fall back to XNNPACK if NNAPI fails
//                        try {
//                            addXnnpack(mapOf("intra_op_num_threads" to "4"))
//                        } catch (e2: Exception) {
//                            setIntraOpNumThreads(4)
//                        }
//                    }
//                }


                // Use big cores only. Most Snapdragon SoCs have 4 big cores (Cortex-A7xx).
                // Using efficiency cores for transformer matmul actually slows things down
                // because their cache is too small and they stall on the weight tensors.
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

    override suspend fun initialize(): Boolean = initialize { _, _ -> }
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
        // Notice we DO NOT set cachedImageFeatures to null here.
        // This keeps the image processing instant!
        Log.d(TAG, "🧹 Text memory wiped. Image features retained.")
    }

    // Standard VLMInterface chat (no OCR text)
    override suspend fun chat(
        image: Bitmap,
        question: String,
        systemPrompt: String?,
        conversationHistory: List<Pair<String, String>>,
        onTokenGenerated: ((String) -> Unit)?
    ): String = chatWithOcr(image, question, null, null, onTokenGenerated)

    /**
     * Primary entry point. Pass [ocrText] from OcrHelper if available —
     * it gets injected into the prompt so the model reasons about accurate
     * text instead of guessing from blurry 378×378 pixels.
     */
    suspend fun chatWithOcr(
        image: Bitmap,
        question: String,
        ocrText: String?,
        systemPrompt: String? = null,
        onTokenGenerated: ((String) -> Unit)? = null
    ): String = withContext(Dispatchers.IO) {
        if (!ready) return@withContext "Not initialized"

        if (currentSequenceLength > 1500) {
            Log.w(TAG, "⚠️ Context too long, resetting.")
            resetChat()
        }

        try {
            // Intercept identity questions — moondream2 has no instruction-tuning
            // for identity and will always invent a name from training data.
            // No prompt wording can reliably prevent this, so we short-circuit.
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

            Log.i("VISION_PROMPT", "=== EXACT PROMPT GOING TO MODEL ===\n$prompt")

            val inputIds = tokenizer!!.encode(prompt)

            // Log the first few token IDs so you can verify in Logcat
            Log.d(TAG, "Encoded ${inputIds.size} tokens, first 10: ${inputIds.take(10).toList()}")

            if (inputIds.isEmpty()) {
                Log.e(TAG, "❌ encode() returned empty! Check tokenizer selfTest in Logcat.")
                return@withContext "Tokenizer error — see Logcat."
            }

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

    // -----------------------------------------------------------------------
    // Image encoding with cache
    // -----------------------------------------------------------------------

    private fun getOrEncodeImage(bitmap: Bitmap): Array<FloatArray> {
        if (cachedImageBitmap === bitmap && cachedImageFeatures != null) {
            Log.d(TAG, "♻️ Reusing cached image features.")
            return cachedImageFeatures!!
        }
        Log.d(TAG, "🖼️ Running vision encoder...")
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

    // -----------------------------------------------------------------------
    // Text embedding
    // -----------------------------------------------------------------------

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

    // -----------------------------------------------------------------------
    // Autoregressive generation with KV cache
    // -----------------------------------------------------------------------

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
                    else -> throw Exception("Unexpected logits: ${logitsRaw?.javaClass}")
                }

                // Apply repetition penalty and temperature before sampling.
                val logits = applyRepetitionPenalty(rawLogits, generated, REPETITION_PENALTY)
                val tokenId = sampleWithTemperature(logits, TEMPERATURE)

                if (tokenId == eosId) break

                generated.add(tokenId)
                onToken?.invoke(tokenizer!!.decode(intArrayOf(tokenId)))

                // --- Update KV cache ---
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
                Log.e(TAG, "Step $step error: ${e.message}")
                break
            }
        }

        // Persist KV cache for next turn
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

    // -----------------------------------------------------------------------
    // Misc
    // -----------------------------------------------------------------------

    override fun getModelInfo(): ModelInfo = ModelInfo("Moondream2 ONNX", "~1.5GB", "Good", "Good")

    // -----------------------------------------------------------------------
    // Sampling helpers
    // -----------------------------------------------------------------------

    /**
     * Applies a repetition penalty to logits.
     * For every token that has already been generated, its logit is divided
     * by [penalty] (if positive) or multiplied (if negative), making it
     * less likely to appear again. Standard implementation from HuggingFace.
     */
    private fun applyRepetitionPenalty(
        logits: FloatArray,
        generatedTokens: List<Int>,
        penalty: Float
    ): FloatArray {
        if (penalty == 1.0f || generatedTokens.isEmpty()) return logits
        val result = logits.copyOf()
        for (tokenId in generatedTokens.toSet()) {
            if (tokenId < result.size) {
                // Do NOT penalize digit/punctuation tokens (IDs 15–57 in GPT-2 vocab).
                // Price lists legitimately repeat "210", "250", "/-" many times.
                // Penalizing them was causing the model to refuse printing numbers.
                if (tokenId in 15..57) continue
                result[tokenId] = if (result[tokenId] > 0f)
                    result[tokenId] / penalty
                else
                    result[tokenId] * penalty
            }
        }
        return result
    }
    /**
     * Samples a token using temperature scaling.
     * temperature = 1.0 → greedy argmax (deterministic, prone to loops).
     * temperature < 1.0 → sharper distribution (more focused).
     * temperature > 1.0 → flatter distribution (more random).
     *
     * We use top-k=40 filtering before sampling so extremely low-probability
     * tokens (garbage, hallucinations) are never sampled.
     */
    private fun sampleWithTemperature(logits: FloatArray, temperature: Float): Int {
        if (temperature <= 0f) {
            // Pure greedy
            return logits.indices.maxByOrNull { logits[it] } ?: 0
        }

        // Scale logits by temperature
        val scaled = FloatArray(logits.size) { logits[it] / temperature }

        // Top-k filtering: zero out everything outside top 40 tokens
        val k = TOP_K
        val topKIndices = scaled.indices
            .sortedByDescending { scaled[it] }
            .take(k)
            .toSet()
        val filtered = FloatArray(scaled.size) { i ->
            if (i in topKIndices) scaled[i] else Float.NEGATIVE_INFINITY
        }

        // Softmax
        val maxLogit = filtered.filter { it.isFinite() }.maxOrNull() ?: 0f
        val exps = FloatArray(filtered.size) { i ->
            if (filtered[i].isFinite()) Math.exp((filtered[i] - maxLogit).toDouble()).toFloat()
            else 0f
        }
        val sumExp = exps.sum()
        if (sumExp <= 0f) return logits.indices.maxByOrNull { logits[it] } ?: 0

        val probs = FloatArray(exps.size) { exps[it] / sumExp }

        // Multinomial sample
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
        visionSession?.close(); embedSession?.close()
        decoderSession?.close(); ortEnv?.close()
        ready = false
    }

    fun getDownloader(): MoondreamDownloader = downloader
    fun testTokenizer(text: String): String = tokenizer?.selfTest() ?: "not loaded"
}