package com.anurag.visionqa.ai

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

class MiniCPMVLM(private val context: Context) : VLMInterface {

    private var ready = false
    // 1. Initialize the downloader
    private val downloader = MiniCPMDownloader(context)

    companion object {
        private const val TAG           = "MiniCPMVLM"
        private const val MODEL_FILE    = "ggml-model-Q4_K_M.gguf"
        private const val MMPROJ_FILE   = "mmproj-model-f16.gguf"
        private const val MODEL_SUBDIR  = "minicpm"
        private const val N_THREADS     = 4
        private const val N_CTX         = 2048
        private const val MAX_TOKENS    = 512
        private const val MAX_IMAGE_DIM = 448
    }

    override suspend fun initialize(): Boolean = initialize { _, _ -> }

    override suspend fun initialize(onProgress: (String, Int) -> Unit): Boolean =
        withContext(Dispatchers.IO) {
            Log.d(TAG, "=== INITIALIZE START ===")

            // 2. RUN THE DOWNLOADER FIRST
            if (!downloader.downloadIfNeeded { msg, prog -> onProgress(msg, prog) }) {
                Log.e(TAG, "❌ Download failed or was interrupted.")
                return@withContext false
            }

            // 3. PROCEED WITH LOADING (Files are guaranteed to exist now)
            val modelDir = File(context.getExternalFilesDir(null), MODEL_SUBDIR)
            val modelPath  = File(modelDir, MODEL_FILE).absolutePath
            val mmprojPath = File(modelDir, MMPROJ_FILE).absolutePath

            Log.d(TAG, "Model dir: ${modelDir.absolutePath}")
            Log.d(TAG, "Free RAM: ${Runtime.getRuntime().freeMemory() / 1_000_000}MB")

            onProgress("Loading model into RAM...", 95)
            Log.d(TAG, "Calling LlamaJNI.loadModel...")

            val loadStart = System.currentTimeMillis()
            val ok = try {
                LlamaJNI.loadModel(modelPath, mmprojPath, N_THREADS, N_CTX)
            } catch (e: Exception) {
                Log.e(TAG, "❌ LlamaJNI.loadModel threw exception: ${e.message}", e)
                false
            }

            val loadMs = System.currentTimeMillis() - loadStart
            Log.d(TAG, "loadModel returned: $ok in ${loadMs}ms")
            ready = ok

            if (ok) {
                Log.d(TAG, "✅ MiniCPM-V ready")
                onProgress("✅ Ready!", 100)
            } else {
                Log.e(TAG, "❌ loadModel() returned false — check C++ Logcat for details")
                onProgress("❌ Model load failed.", 0)
            }
            ok
        }

    override fun isReady() = ready

    override suspend fun chat(
        image: Bitmap,
        question: String,
        systemPrompt: String?,
        conversationHistory: List<Pair<String, String>>,
        onTokenGenerated: ((String) -> Unit)?
    ): String = withContext(Dispatchers.IO) {
        if (!ready) return@withContext "Model not loaded."

        // 1. Scale the image to a safe size
        val safeImage = capImageSize(image, MAX_IMAGE_DIM)

        // 2. Capture dimensions IMMEDIATELY while the bitmap is alive
        val width  = safeImage.width
        val height = safeImage.height

        // 3. Extract pixels into a standard IntArray
        val pixels = IntArray(width * height)
        safeImage.getPixels(pixels, 0, width, 0, 0, width, height)

        // 4. Run the AI generation
        // Note: We pass the RAW 'width' and 'height' variables, NOT 'safeImage.width'
        val result = try {
            LlamaJNI.generate(
                pixelData     = pixels,
                imageWidth    = width,
                imageHeight   = height,
                prompt        = question,
                maxTokens     = MAX_TOKENS,
                tokenCallback = { tokenBytes ->
                    val tokenStr = String(tokenBytes, Charsets.UTF_8)
                    onTokenGenerated?.invoke(tokenStr)
                }
            )
        } catch (e: Exception) {
            Log.e(TAG, "❌ Generation error: ${e.message}")
            "Error during generation."
        }

        // 5. FINALLY recycle the safeImage copy now that the AI is finished with it
        if (safeImage !== image) {
            safeImage.recycle()
        }

        result.trim()
    }

    fun abort() {
        LlamaJNI.abortGeneration()
    }

    private fun capImageSize(bitmap: Bitmap, maxDim: Int): Bitmap {
        val w = bitmap.width
        val h = bitmap.height
        if (w <= maxDim && h <= maxDim) return bitmap
        val scale = maxDim.toFloat() / maxOf(w, h)
        return Bitmap.createScaledBitmap(bitmap, (w * scale).toInt(), (h * scale).toInt(), true)
    }

    override fun getModelInfo() = ModelInfo(
        name    = "Qwen2-VL 2B (Redmi Test)",
        size    = "~1.7GB",
        speed   = "CPU Optimized",
        quality = "Lite Multimodal"
    )

    override fun cleanup() {
        LlamaJNI.freeModel()
        ready = false
    }
}