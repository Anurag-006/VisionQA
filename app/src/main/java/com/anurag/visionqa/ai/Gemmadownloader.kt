package com.anurag.visionqa.ai

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.TimeUnit
import com.anurag.visionqa.BuildConfig

/**
 * Downloads the VisionQA vision model for use with MediaPipe LLM Inference API.
 *
 * Model is identified internally only — the user-facing name is always "VisionQA".
 *
 * For production: serve the model from your own CDN or use Play Asset Delivery.
 */
class GemmaDownloader(private val context: Context) {

    // Set your HuggingFace personal access token here.
    // Generate one at: https://huggingface.co/settings/tokens (read scope is enough)
    val HF_TOKEN = BuildConfig.HF_TOKEN

    private val client = OkHttpClient.Builder()
        .connectTimeout(60, TimeUnit.SECONDS)
        .readTimeout(900, TimeUnit.SECONDS)   // 15 min for large model on mobile data
        .writeTimeout(60, TimeUnit.SECONDS)
        .followRedirects(true)
        .build()

    // External storage so the model survives app updates
    val modelDir: File = File(context.getExternalFilesDir(null), "gemma3ne2b")

    companion object {
        private const val TAG = "VisionQADownloader"
        private const val MODEL_FILENAME = "gemma-3n-E2B-it-int4.task"
        private const val MODEL_URL =
            "https://huggingface.co/google/gemma-3n-E2B-it-litert-preview/resolve/main/gemma-3n-E2B-it-int4.task"
        private const val MIN_MODEL_BYTES = 2_800_000_000L  // ~3.1 GB
        const val MODEL_SIZE_LABEL = "~3.1 GB"
    }

    init {
        if (!modelDir.exists()) modelDir.mkdirs()
    }

    suspend fun downloadIfNeeded(onProgress: (String, Int) -> Unit): Boolean =
        withContext(Dispatchers.IO) {
            val modelFile = File(modelDir, MODEL_FILENAME)

            if (modelExists()) {
                Log.d(TAG, "✅ Already present: ${modelFile.length() / 1_000_000}MB")
                // User-facing: VisionQA branding only
                onProgress("VisionQA model ready ($MODEL_SIZE_LABEL)", 100)
                return@withContext true
            }

            Log.d(TAG, "Downloading to ${modelDir.absolutePath}")
            onProgress("Downloading VisionQA model ($MODEL_SIZE_LABEL)...", 0)

            return@withContext try {
                downloadFile(modelFile) { pct ->
                    onProgress("Downloading VisionQA model... $pct%", pct)
                }

                if (modelFile.length() < MIN_MODEL_BYTES) {
                    modelFile.delete()
                    onProgress("Download incomplete. Please retry on Wi-Fi.", 0)
                    false
                } else {
                    Log.d(TAG, "✅ Done: ${modelFile.length() / 1_000_000}MB")
                    onProgress("Download complete! Initializing...", 100)
                    true
                }
            } catch (e: Exception) {
                Log.e(TAG, "Download failed: ${e.message}", e)
                if (modelFile.exists() && modelFile.length() < MIN_MODEL_BYTES) modelFile.delete()
                onProgress(friendlyError(e), 0)
                false
            }
        }

    fun getModelPath(): String = File(modelDir, MODEL_FILENAME).absolutePath

    fun modelExists(): Boolean {
        val f = File(modelDir, MODEL_FILENAME)
        return f.exists() && f.length() >= MIN_MODEL_BYTES
    }

    fun deleteModel() = File(modelDir, MODEL_FILENAME).delete()

    private fun downloadFile(destination: File, onProgress: (Int) -> Unit) {
        val req = Request.Builder().url(MODEL_URL).apply {
            if (HF_TOKEN.isNotBlank()) addHeader("Authorization", "Bearer $HF_TOKEN")
        }.build()

        val response = client.newCall(req).execute()
        if (!response.isSuccessful) throw Exception("HTTP ${response.code}: ${response.message}")

        val body = response.body ?: throw Exception("Empty response body")
        val contentLength = body.contentLength()
        var totalBytes = 0L; var lastPct = -1

        body.byteStream().use { input ->
            FileOutputStream(destination).use { output ->
                val buf = ByteArray(256 * 1024)
                var n: Int
                while (input.read(buf).also { n = it } != -1) {
                    output.write(buf, 0, n); totalBytes += n
                    if (contentLength > 0) {
                        val pct = ((totalBytes * 100) / contentLength).toInt()
                        if (pct != lastPct) { onProgress(pct); lastPct = pct }
                    }
                }
                output.flush(); onProgress(100)
            }
        }
    }

    private fun friendlyError(e: Exception) = when {
        (e.message ?: "").contains("401") || (e.message ?: "").contains("403") ->
            "Access denied. Please contact support or check your internet connection."
        (e.message ?: "").contains("Unable to resolve") ->
            "No internet connection. Please connect and try again."
        (e.message ?: "").contains("timeout") ->
            "Download timed out. Please use Wi-Fi and retry."
        else -> "Download failed. Please retry."
    }
}