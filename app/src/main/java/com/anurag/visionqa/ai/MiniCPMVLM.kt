package com.anurag.visionqa.ai

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

class MiniCPMVLM(private val context: Context) : VLMInterface {

    private var ready = false
    private val downloader = MiniCPMDownloader(context)

    // Conversation history stored as formatted turn strings.
    // Each entry is ONE complete assistant reply.
    // Format: "<|im_start|>assistant\nANSWER<|im_end|>\n"
    // The C++ side receives the full concatenation of all entries as `history`.
    private val conversationHistory = mutableListOf<String>()
    private var lastAnswer = ""

    companion object {
        private const val TAG          = "MiniCPMVLM"
        private const val MODEL_FILE   = "ggml-model-Q4_K_M.gguf"
        private const val MMPROJ_FILE  = "mmproj-model-f16.gguf"
        private const val MODEL_SUBDIR = "minicpm"

        // Snapdragon 8 Gen 2 (iQOO Neo 9 Pro) core layout:
        //   1x Cortex-X3  prime  @ 3.2 GHz
        //   4x Cortex-A715 perf  @ 2.8 GHz
        //   3x Cortex-A510 eff   @ 2.0 GHz
        //
        // Use prime + perf cores only (5 total).
        // Efficiency cores hurt inference speed due to cache thrashing.
        private const val N_THREADS       = 5
        private const val N_THREADS_BATCH = 5  // all big cores for prefill/image encoding

        // 8192 tokens: ~1024 image tokens + system prompt + plenty of
        // multi-turn headroom. Safe on Neo 9 Pro (12–16 GB RAM).
        private const val N_CTX        = 8192

        // 768 tokens ≈ 3–5 paragraphs. Model stops naturally via EOS if shorter.
        // Raised from 512 to allow fuller, untruncated answers.
        private const val MAX_TOKENS   = 768

        // MiniCPM-V 2.6 native resolution. Do not reduce — loses visual detail.
        private const val MAX_IMAGE_DIM = 448
    }

    override suspend fun initialize(): Boolean = initialize { _, _ -> }

    override suspend fun initialize(onProgress: (String, Int) -> Unit): Boolean =
        withContext(Dispatchers.IO) {
            Log.d(TAG, "=== INITIALIZE START ===")

            if (!downloader.downloadIfNeeded { msg, prog -> onProgress(msg, prog) }) {
                Log.e(TAG, "❌ Download failed.")
                return@withContext false
            }

            val modelDir   = File(context.getExternalFilesDir(null), MODEL_SUBDIR)
            val modelPath  = File(modelDir, MODEL_FILE).absolutePath
            val mmprojPath = File(modelDir, MMPROJ_FILE).absolutePath

            Log.d(TAG, "Model:  $modelPath (${File(modelPath).length() / 1_000_000}MB)")
            Log.d(TAG, "MMProj: $mmprojPath (${File(mmprojPath).length() / 1_000_000}MB)")

            val actManager = context.getSystemService(Context.ACTIVITY_SERVICE)
                    as android.app.ActivityManager
            val memInfo = android.app.ActivityManager.MemoryInfo()
            actManager.getMemoryInfo(memInfo)
            val availMB = memInfo.availMem / 1_000_000
            Log.d(TAG, "Available system RAM: ${availMB}MB")

            if (availMB < 2000) {
                onProgress("❌ Not enough free RAM (${availMB}MB). Close other apps and retry.", 0)
                return@withContext false
            }

            onProgress("Loading model into RAM...", 95)

            val loadStart = System.currentTimeMillis()
            val ok = try {
                LlamaJNI.loadModel(
                    modelPath    = modelPath,
                    mmprojPath   = mmprojPath,
                    threads      = N_THREADS,
                    threadsBatch = N_THREADS_BATCH,
                    ctx          = N_CTX
                )
            } catch (e: Exception) {
                Log.e(TAG, "❌ LlamaJNI.loadModel threw: ${e.message}", e)
                false
            }

            val elapsed = System.currentTimeMillis() - loadStart
            ready = ok

            if (ok) {
                Log.d(TAG, "✅ MiniCPM-V 2.6 ready in ${elapsed}ms " +
                        "(threads=$N_THREADS batch=$N_THREADS_BATCH ctx=$N_CTX)")
                onProgress("✅ Ready!", 100)
            } else {
                Log.e(TAG, "❌ loadModel() returned false")
                onProgress("❌ Model load failed. Try restarting the app.", 0)
            }
            ok
        }

    override fun isReady() = ready

    /**
     * Call this whenever the user picks a NEW image.
     * Clears the C++ KV cache and resets conversation history so the next
     * generate() call re-encodes the new image from scratch.
     */
    fun resetForNewImage() {
        conversationHistory.clear()
        lastAnswer = ""
        LlamaJNI.resetCache()
        Log.d(TAG, "Conversation and KV cache reset for new image.")
    }

    override suspend fun chat(
        image: Bitmap,
        question: String,
        systemPrompt: String?,
        conversationHistory: List<Pair<String, String>>,
        onTokenGenerated: ((String) -> Unit)?
    ): String = withContext(Dispatchers.IO) {
        if (!ready) return@withContext "❌ Model not loaded. Restart the app."
        if (question.isBlank()) return@withContext "❌ Please type a question."
        if (image.width == 0 || image.height == 0) return@withContext "❌ Invalid image."

        val safeImage = capImageSize(image, MAX_IMAGE_DIM)
        val width     = safeImage.width
        val height    = safeImage.height

        val pixels = IntArray(width * height)
        safeImage.getPixels(pixels, 0, width, 0, 0, width, height)

        // Build the history string from all prior exchanges.
        // On the first call this is empty — the C++ side encodes the image.
        // On follow-ups this contains all prior assistant turns so the model
        // has full context, while image re-encoding is skipped (fast path).
        val historyStr = this@MiniCPMVLM.conversationHistory.joinToString("")

        Log.d(TAG, "chat() | turn=${this@MiniCPMVLM.conversationHistory.size + 1} " +
                "history_len=${historyStr.length} question=$question")

        val result = try {
            LlamaJNI.generate(
                pixelData     = pixels,
                imageWidth    = width,
                imageHeight   = height,
                prompt        = question,
                history       = historyStr,
                maxTokens     = MAX_TOKENS,
                tokenCallback = { tokenBytes ->
                    val tokenStr = String(tokenBytes, Charsets.UTF_8)
                    onTokenGenerated?.invoke(tokenStr)
                }
            )
        } catch (e: Exception) {
            Log.e(TAG, "❌ Generation error: ${e.message}", e)
            "❌ Generation error: ${e.message}"
        }

        if (safeImage !== image) safeImage.recycle()

        val trimmed = result.trim()

        // Record this exchange so the next turn has full context.
        if (!trimmed.startsWith("❌")) {
            this@MiniCPMVLM.conversationHistory.add(
                "<|im_start|>assistant\n${trimmed}<|im_end|>\n"
            )
            lastAnswer = trimmed
        }

        trimmed
    }

    fun abort() = LlamaJNI.abortGeneration()

    private fun capImageSize(bitmap: Bitmap, maxDim: Int): Bitmap {
        val w = bitmap.width
        val h = bitmap.height
        if (w <= maxDim && h <= maxDim) return bitmap
        val scale = maxDim.toFloat() / maxOf(w, h)
        return Bitmap.createScaledBitmap(bitmap, (w * scale).toInt(), (h * scale).toInt(), true)
    }

    override fun getModelInfo() = ModelInfo(
        name    = "MiniCPM-V 2.6",
        size    = "~2.3GB",
        speed   = "CPU Optimized",
        quality = "Multimodal VLM"
    )

    override fun cleanup() {
        LlamaJNI.freeModel()
        ready = false
    }
}