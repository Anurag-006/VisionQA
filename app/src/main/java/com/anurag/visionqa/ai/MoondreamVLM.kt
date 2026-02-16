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

    companion object {
        private const val TAG = "MoondreamVLM"
        private const val IMAGE_SIZE = 378
        private const val MAX_NEW_TOKENS = 100
    }

    suspend fun initialize(onProgress: (String, Int) -> Unit = { _, _ -> }): Boolean = withContext(Dispatchers.IO) {
        try {
            Log.d(TAG, "=== KV-CACHED MOONDREAM INIT START ===")
            val downloaded = downloader.downloadIfNeeded { msg, prog -> onProgress(msg, prog) }
            if (!downloaded) return@withContext false

            tokenizer = RealTokenizer(context, downloader.getTokenizerPath())
            ortEnv = OrtEnvironment.getEnvironment()

            val options = OrtSession.SessionOptions().apply {
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)

                // Explicitly enable XNNPACK for highly optimized CPU math
                try {
                    addXnnpack(mapOf("intra_op_num_threads" to "4"))
                    Log.d(TAG, "✅ XNNPACK enabled")
                } catch (e: Exception) {
                    Log.w(TAG, "XNNPACK unavailable, falling back to default CPU")
                    setIntraOpNumThreads(4)
                    setInterOpNumThreads(2)
                }
            }

            visionSession = ortEnv!!.createSession(downloader.getVisionPath(), options)
            embedSession = ortEnv!!.createSession(downloader.getEmbedPath(), options)
            decoderSession = ortEnv!!.createSession(downloader.getDecoderPath(), options)

            ready = true
            Log.d(TAG, "=== INIT COMPLETE ===")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Init failed", e)
            false
        }
    }

    override suspend fun initialize(): Boolean = initialize { _, _ -> }
    override fun isReady(): Boolean = ready

    override suspend fun chat(
        image: Bitmap,
        question: String,
        conversationHistory: List<Pair<String, String>>,
        onTokenGenerated: ((String) -> Unit)?
    ): String = withContext(Dispatchers.IO) {

        if (!ready) return@withContext "Not initialized"

        try {
            Log.d(TAG, "=== CHAT START ===")
            val startTime = System.currentTimeMillis()

            val imageEmbeds = encodeImage(image)
            val prompt = tokenizer!!.formatPrompt(question)
            val inputIds = tokenizer!!.encode(prompt)

            Log.d(TAG, "✅ Tokenized: ${inputIds.size} tokens")

            val textEmbeds = embedText(inputIds)
            val combinedEmbeds = combineEmbeddings(imageEmbeds, textEmbeds)

            val response = generateWithKVCache(combinedEmbeds, MAX_NEW_TOKENS, onTokenGenerated)

            Log.d(TAG, "=== CHAT COMPLETE ===")
            response.trim().removePrefix("Answer:").removePrefix(":").trim()

        } catch (e: Exception) {
            Log.e(TAG, "Chat error", e)
            "Error: ${e.message}"
        }
    }

    private fun encodeImage(bitmap: Bitmap): Array<FloatArray> {
        val resized = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true)
        val pixels = IntArray(IMAGE_SIZE * IMAGE_SIZE)
        resized.getPixels(pixels, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE)

        val imageData = FloatArray(3 * IMAGE_SIZE * IMAGE_SIZE)
        val mean = floatArrayOf(0.5f, 0.5f, 0.5f)
        val std = floatArrayOf(0.5f, 0.5f, 0.5f)

        for (i in pixels.indices) {
            val p = pixels[i]
            val r = ((p shr 16) and 0xFF) / 255f
            val g = ((p shr 8) and 0xFF) / 255f
            val b = (p and 0xFF) / 255f

            imageData[i] = (r - mean[0]) / std[0]
            imageData[IMAGE_SIZE * IMAGE_SIZE + i] = (g - mean[1]) / std[1]
            imageData[2 * IMAGE_SIZE * IMAGE_SIZE + i] = (b - mean[2]) / std[2]
        }

        val imageTensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(imageData), longArrayOf(1, 3, IMAGE_SIZE.toLong(), IMAGE_SIZE.toLong()))
        val outputs = visionSession!!.run(mapOf("pixel_values" to imageTensor))

        val output = outputs[0].value
        val imageFeatures = (output as Array<*>)[0] as Array<FloatArray>

        imageTensor.close()
        outputs.close()
        return imageFeatures
    }

    private fun embedText(inputIds: IntArray): Array<FloatArray> {
        val inputIdsLong = inputIds.map { it.toLong() }.toLongArray()
        val inputIdsTensor = OnnxTensor.createTensor(ortEnv, LongBuffer.wrap(inputIdsLong), longArrayOf(1, inputIdsLong.size.toLong()))

        val outputs = embedSession!!.run(mapOf("input_ids" to inputIdsTensor))
        val output = outputs[0].value
        val textEmbeds = (output as Array<*>)[0] as Array<FloatArray>

        inputIdsTensor.close()
        outputs.close()
        return textEmbeds
    }

    private fun combineEmbeddings(imageEmbeds: Array<FloatArray>, textEmbeds: Array<FloatArray>): Array<FloatArray> {
        return imageEmbeds + textEmbeds
    }

    private fun generateWithKVCache(
        embeddings: Array<FloatArray>,
        maxTokens: Int,
        onToken: ((String) -> Unit)?
    ): String {

        val generatedTokens = mutableListOf<Int>()
        val seqLen = embeddings.size
        val hiddenSize = embeddings[0].size
        val targetEosId = tokenizer?.getEosId() ?: 50256

        val embeddingsFlat = FloatArray(seqLen * hiddenSize)
        for (i in embeddings.indices) {
            for (j in embeddings[i].indices) {
                embeddingsFlat[i * hiddenSize + j] = embeddings[i][j]
            }
        }

        var inputs = mutableMapOf<String, OnnxTensor>()
        inputs["inputs_embeds"] = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(embeddingsFlat), longArrayOf(1, seqLen.toLong(), hiddenSize.toLong()))
        inputs["attention_mask"] = OnnxTensor.createTensor(ortEnv, LongBuffer.wrap(LongArray(seqLen) { 1L }), longArrayOf(1, seqLen.toLong()))
        inputs["position_ids"] = OnnxTensor.createTensor(ortEnv, LongBuffer.wrap(LongArray(seqLen) { it.toLong() }), longArrayOf(1, seqLen.toLong()))

        val numLayers = 24
        val numHeads = 32
        val headDim = hiddenSize / numHeads
        val emptyPastShape = longArrayOf(1, numHeads.toLong(), 0, headDim.toLong())
        val emptyPast = FloatArray(0)

        for (layer in 0 until numLayers) {
            inputs["past_key_values.$layer.key"] = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(emptyPast), emptyPastShape)
            inputs["past_key_values.$layer.value"] = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(emptyPast), emptyPastShape)
        }

        var currentSeqLen = seqLen
        var previousOutputs: OrtSession.Result? = null
        val createdTensors = mutableListOf<OnnxTensor>()

        for (i in 0 until maxTokens) {
            try {
                val outputs = decoderSession!!.run(inputs)
                previousOutputs?.close()
                previousOutputs = outputs

                val logitsOutput = outputs[0].value

                // RESTORED 3D ARRAY HANDLING
                val logits = when (logitsOutput) {
                    is Array<*> -> {
                        val batch = logitsOutput[0] as Array<*>
                        batch.last() as FloatArray // Extract the exact token we need
                    }
                    is FloatArray -> logitsOutput
                    else -> throw Exception("Unexpected logits format")
                }

                val tokenId = logits.indices.maxByOrNull { logits[it] } ?: 0

                Log.d(TAG, "Token $i: $tokenId")

                if (tokenId == targetEosId || tokenId == 2 || tokenId == 1) {
                    Log.d(TAG, "End of sequence triggered.")
                    break
                }

                generatedTokens.add(tokenId)
                onToken?.invoke(tokenizer!!.decode(intArrayOf(tokenId)))

                val newPastKV = mutableMapOf<String, OnnxTensor>()
                for (layer in 0 until numLayers) {
                    newPastKV["past_key_values.$layer.key"] = outputs.get("present.$layer.key").get() as OnnxTensor
                    newPastKV["past_key_values.$layer.value"] = outputs.get("present.$layer.value").get() as OnnxTensor
                }

                createdTensors.forEach { it.close() }
                createdTensors.clear()
                inputs.clear()

                val nextTokenEmbeds = embedText(intArrayOf(tokenId))
                val nextEmbedTensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(nextTokenEmbeds[0]), longArrayOf(1, 1, hiddenSize.toLong()))
                inputs["inputs_embeds"] = nextEmbedTensor
                createdTensors.add(nextEmbedTensor)

                currentSeqLen++
                val newAttnTensor = OnnxTensor.createTensor(ortEnv, LongBuffer.wrap(LongArray(currentSeqLen) { 1L }), longArrayOf(1, currentSeqLen.toLong()))
                inputs["attention_mask"] = newAttnTensor
                createdTensors.add(newAttnTensor)

                val newPosTensor = OnnxTensor.createTensor(ortEnv, LongBuffer.wrap(longArrayOf((currentSeqLen - 1).toLong())), longArrayOf(1, 1))
                inputs["position_ids"] = newPosTensor
                createdTensors.add(newPosTensor)

                inputs.putAll(newPastKV)

            } catch (e: Exception) {
                Log.e(TAG, "Error at token $i", e)
                break
            }
        }

        previousOutputs?.close()
        createdTensors.forEach { it.close() }
        return tokenizer!!.decode(generatedTokens.toIntArray())
    }

    override fun getModelInfo(): ModelInfo = ModelInfo("Moondream2 ONNX", "~1.5GB", "Good", "Good")

    override fun cleanup() {
        visionSession?.close()
        embedSession?.close()
        decoderSession?.close()
        ortEnv?.close()
        ready = false
    }

    fun testTokenizer(text: String): String = "Tokenizer loaded: ${tokenizer?.getVocabSize()}"
    fun getDownloader(): MoondreamDownloader = downloader
}