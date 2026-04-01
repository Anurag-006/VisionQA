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
    // Each entry is ONE complete exchange: the assistant reply to a prior question.
    // Format per entry: "<|im_start|>assistant\nANSWER<|im_end|>\n<|im_start|>user\nNEXT_Q<|im_end|>\n"
    // The C++ side receives the full concatenation of all entries as `history`.
    private val conversationHistory = mutableListOf<String>()
    private var lastAnswer = ""

    companion object {
        private const val TAG          = "MiniCPMVLM"
        private const val MODEL_FILE   = "ggml-model-Q4_K_M.gguf"
        private const val MMPROJ_FILE  = "mmproj-model-f16.gguf"
        private const val MODEL_SUBDIR = "minicpm"

        // 4 threads is the sweet spot on most Android SoCs.
        // More threads = cache contention = slower, not faster.
        private const val N_THREADS    = 4

        // 4096 gives: ~1024 image tokens + ~50 system/prompt + ~512 answer headroom
        // per turn, with room for several follow-ups before context fills.
        private const val N_CTX        = 4096

        // 512 tokens ≈ 2–3 paragraphs. Model stops naturally via EOS if shorter.
        private const val MAX_TOKENS   = 512

        // MiniCPM-V 2.6 native resolution. Do not reduce — loses detail.
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
                LlamaJNI.loadModel(modelPath, mmprojPath, N_THREADS, N_CTX)
            } catch (e: Exception) {
                Log.e(TAG, "❌ LlamaJNI.loadModel threw: ${e.message}", e)
                false
            }

            val elapsed = System.currentTimeMillis() - loadStart
            ready = ok

            if (ok) {
                Log.d(TAG, "✅ MiniCPM-V 2.6 ready in ${elapsed}ms")
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
        // On follow-ups this contains all prior assistant+user turns so the
        // model has full context, but image re-encoding is skipped (fast path).
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

        // Record this exchange in history so the next turn has full context.
        // Format: assistant reply + next user turn opener.
        // The next question is appended by C++ based on the `prompt` param.
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