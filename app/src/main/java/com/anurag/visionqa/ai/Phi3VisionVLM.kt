//package com.anurag.visionqa.ai
//
//import android.content.Context
//import android.graphics.Bitmap
//import android.util.Log
//import ai.onnxruntime.*
//import kotlinx.coroutines.Dispatchers
//import kotlinx.coroutines.withContext
//import java.nio.FloatBuffer
//import java.nio.LongBuffer
//
//class Phi3VisionVLM(private val context: Context) : VLMInterface {
//
//    private var ortEnv: OrtEnvironment? = null
//    private var visionSession: OrtSession? = null
//    private var embedSession: OrtSession? = null
//    private var textSession: OrtSession? = null
//
//    private val downloader = Phi3Downloader(context)
//    private val tokenizer = SimpleTokenizer()
//    private var ready = false
//
//    companion object {
//        private const val TAG = "Phi3VisionVLM"
//        private const val IMAGE_SIZE = 336
//    }
//
//    // This handles the specialized progress-based initialization
//    suspend fun initialize(onProgress: (String, Int) -> Unit): Boolean = withContext(Dispatchers.IO) {
//        try {
//            Log.d(TAG, "=== INITIALIZATION START ===")
//            if (!downloader.downloadIfNeeded(onProgress)) {
//                Log.e(TAG, "Downloader failed to provide files.")
//                return@withContext false
//            }
//
//            ortEnv = OrtEnvironment.getEnvironment()
//            val options = OrtSession.SessionOptions().apply {
//                setIntraOpNumThreads(1) // Reduced to 1 thread to save memory during init
//                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT) // Lower opt to save RAM
//            }
//
//            // 1. Load Vision
//            Log.d(TAG, "Loading Vision Session...")
//            try {
//                visionSession = ortEnv!!.createSession(downloader.getVisionPath(), options)
//            } catch (e: Exception) {
//                Log.e(TAG, "Vision Session failed: ${e.message}")
//                return@withContext false
//            }
//
//            // 2. Load Embedding
//            Log.d(TAG, "Loading Embedding Session...")
//            try {
//                embedSession = ortEnv!!.createSession(downloader.getEmbedPath(), options)
//            } catch (e: Exception) {
//                Log.e(TAG, "Embedding Session failed: ${e.message}")
//                return@withContext false
//            }
//
//            // 3. Load Main Text (The 2.33GB File)
//            Log.d(TAG, "Loading Text Session (2.33GB Weights)...")
//            try {
//                // CRITICAL: Ensure the .data file is in the same folder as the .onnx file
//                textSession = ortEnv!!.createSession(downloader.getTextPath(), options)
//            } catch (e: Exception) {
//                Log.e(TAG, "Text Session failed: ${e.message}. This is likely an Out of Memory (OOM) error.")
//                return@withContext false
//            }
//
//            ready = true
//            Log.d(TAG, "✅ All sessions successfully active")
//            true
//        } catch (e: Exception) {
//            Log.e(TAG, "Initialization failed: ${e.message}")
//            e.printStackTrace()
//            false
//        }
//    }
//    // FIX 1: Explicitly implement the no-parameter version required by VLMInterface
//    override suspend fun initialize(): Boolean = initialize { _, _ -> }
//
//    override fun isReady(): Boolean = ready
//
//    fun testNative(): String {
//        return if (ready && textSession != null) "✅ Multi-Session ONNX Active" else "❌ Sessions not loaded"
//    }
//
//    suspend fun testTextOnly(prompt: String): String = withContext(Dispatchers.IO) {
//        if (!ready || textSession == null) return@withContext "Error: Model not ready"
//        try {
//            val inputIds = tokenizer.encode(tokenizer.formatPrompt(prompt))
//            val inputs = mutableMapOf<String, OnnxTensor>()
//            val shape = longArrayOf(1, inputIds.size.toLong())
//
//            inputs["input_ids"] = OnnxTensor.createTensor(ortEnv, LongBuffer.wrap(inputIds), shape)
//            inputs["attention_mask"] = OnnxTensor.createTensor(ortEnv, LongBuffer.wrap(LongArray(inputIds.size){1L}), shape)
//
//            val outputs = textSession!!.run(inputs)
//            decodeOutput(outputs)
//        } catch (e: Exception) { "Error: ${e.message}" }
//    }
//
//    override suspend fun chat(image: Bitmap, question: String, history: List<Pair<String, String>>): String = withContext(Dispatchers.IO) {
//        if (!ready) return@withContext "Model not initialized"
//
//        try {
//            // 1. Process image through Vision Session
//            val imageData = preprocessImage(image)
//            val imgShape = longArrayOf(1, 3, IMAGE_SIZE.toLong(), IMAGE_SIZE.toLong())
//            val visionInputs = mutableMapOf<String, OnnxTensor>()
//            visionInputs["pixel_values"] = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(imageData), imgShape)
//
//            // Note: Vision session produces embeddings to be combined with text in future versions
//            visionSession!!.run(visionInputs).use { /* Process visual embeddings */ }
//
//            // 2. Process text
//            val inputIds = tokenizer.encode(tokenizer.formatPrompt(question))
//            val txtShape = longArrayOf(1, inputIds.size.toLong())
//            val textInputs = mutableMapOf<String, OnnxTensor>()
//            textInputs["input_ids"] = OnnxTensor.createTensor(ortEnv, LongBuffer.wrap(inputIds), txtShape)
//            textInputs["attention_mask"] = OnnxTensor.createTensor(ortEnv, LongBuffer.wrap(LongArray(inputIds.size){1L}), txtShape)
//
//            val finalOutput = textSession!!.run(textInputs)
//            decodeOutput(finalOutput)
//        } catch (e: Exception) { "Error: ${e.message}" }
//    }
//
//    // FIX 2: Re-added the missing preprocessImage helper
//    private fun preprocessImage(bitmap: Bitmap): FloatArray {
//        val resized = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true)
//        val pixels = IntArray(IMAGE_SIZE * IMAGE_SIZE)
//        resized.getPixels(pixels, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE)
//
//        // CHW format (Channels, Height, Width)
//        val floatArray = FloatArray(3 * IMAGE_SIZE * IMAGE_SIZE)
//        for (i in 0 until IMAGE_SIZE * IMAGE_SIZE) {
//            val pixel = pixels[i]
//            // Extract RGB and normalize (0-1 range)
//            floatArray[i] = ((pixel shr 16) and 0xFF) / 255.0f
//            floatArray[IMAGE_SIZE * IMAGE_SIZE + i] = ((pixel shr 8) and 0xFF) / 255.0f
//            floatArray[2 * IMAGE_SIZE * IMAGE_SIZE + i] = (pixel and 0xFF) / 255.0f
//        }
//        return floatArray
//    }
//
//    private fun decodeOutput(outputs: OrtSession.Result): String {
//        return try {
//            val outputTensor = outputs[0] as OnnxTensor
//            // Standard ONNX output is often Array<LongArray> for token IDs
//            val output = outputTensor.value as Array<LongArray>
//            tokenizer.decode(output[0])
//        } catch (e: Exception) {
//            "Response received (processing results...)"
//        }
//    }
//
//    override fun getModelInfo() = ModelInfo("Phi-3-Vision (Split ONNX)", "~3.2GB", "90-120s", "High")
//
//    override fun cleanup() {
//        visionSession?.close()
//        embedSession?.close()
//        textSession?.close()
//        ortEnv?.close()
//        ready = false
//    }
//}