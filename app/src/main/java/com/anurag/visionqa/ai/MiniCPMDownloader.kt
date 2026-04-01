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

class MiniCPMDownloader(private val context: Context) {

    private val client = OkHttpClient.Builder()
        .connectTimeout(300, TimeUnit.SECONDS)
        .readTimeout(300, TimeUnit.SECONDS)
        .writeTimeout(300, TimeUnit.SECONDS)
        .build()

    private val modelDir = context.getExternalFilesDir("minicpm") ?: File(context.filesDir, "minicpm")

    companion object {
        private const val TAG = "MiniCPMDownloader"

        // 🚨 TEMPORARY REDMI TEST URLS (Qwen2-VL 2B) 🚨
        private const val BRAIN_URL   = "https://huggingface.co/bartowski/Qwen2-VL-2B-Instruct-GGUF/resolve/main/Qwen2-VL-2B-Instruct-Q4_K_M.gguf"
        private const val EYES_URL  = "https://huggingface.co/bartowski/Qwen2-VL-2B-Instruct-GGUF/resolve/main/mmproj-Qwen2-VL-2B-Instruct-f16.gguf"
    }

    init {
        if (!modelDir.exists()) {
            modelDir.mkdirs()
        }
    }

    suspend fun downloadIfNeeded(
        onProgress: (String, Int) -> Unit
    ): Boolean = withContext(Dispatchers.IO) {

        val brainFile = File(modelDir, "ggml-model-Q4_K_M.gguf")
        val eyesFile = File(modelDir, "mmproj-model-f16.gguf")

        if (modelsExist()) {
            onProgress("Models ready (~1.7GB)", 100)
            return@withContext true
        }

        try {
            // 1. Download the Brain (986 MB)
            if (!brainFile.exists() || brainFile.length() < 900_000_000L) {
                onProgress("Downloading Qwen Brain (986MB)...", 0)
                downloadFile(
                    url = BRAIN_URL,
                    destination = brainFile,
                    onProgress = { p -> onProgress("Brain: $p%", (p * 0.6).toInt()) }
                )
            }

            // 2. Download the Eyes (710 MB)
            if (!eyesFile.exists() || eyesFile.length() < 700_000_000L) {
                onProgress("Downloading Vision Projector (710MB)...", 60)
                downloadFile(
                    url = EYES_URL,
                    destination = eyesFile,
                    onProgress = { p -> onProgress("Eyes: $p%", 60 + (p * 0.4).toInt()) }
                )
            }

            Log.d(TAG, "✅ All GGUF downloads complete!")
            onProgress("Download complete! Ready to Initialize.", 100)
            true

        } catch (e: Exception) {
            Log.e(TAG, "❌ Download failed", e)
            onProgress("Download failed: ${e.message}", 0)
            if (brainFile.length() < 900_000_000L) brainFile.delete()
            if (eyesFile.length() < 700_000_000L) eyesFile.delete()
            false
        }
    }

    private fun downloadFile(
        url: String,
        destination: File,
        onProgress: (Int) -> Unit
    ) {
        Log.d(TAG, "Starting download: $url")
        val request = Request.Builder().url(url).build()
        val response = client.newCall(request).execute()

        if (!response.isSuccessful) throw Exception("HTTP ${response.code}: ${response.message}")

        val body = response.body ?: throw Exception("Empty response body")
        val contentLength = body.contentLength()

        body.byteStream().use { input ->
            FileOutputStream(destination).use { output ->
                val buffer = ByteArray(64 * 1024) // 64KB buffer for faster writing
                var totalBytes = 0L
                var bytes: Int
                var lastProgress = 0
                var lastLogTime = System.currentTimeMillis()

                while (input.read(buffer).also { bytes = it } != -1) {
                    output.write(buffer, 0, bytes)
                    totalBytes += bytes

                    if (contentLength > 0) {
                        val progress = ((totalBytes * 100) / contentLength).toInt()
                        val now = System.currentTimeMillis()

                        if (progress >= lastProgress + 1 || now - lastLogTime > 1000) {
                            onProgress(progress)
                            lastProgress = progress
                            lastLogTime = now
                        }
                    }
                }
                output.flush()
                onProgress(100)
            }
        }
    }

    fun modelsExist(): Boolean {
        val brain = File(modelDir, "ggml-model-Q4_K_M.gguf")
        val eyes = File(modelDir, "mmproj-model-f16.gguf")

        // Qwen Validation: Brain > 900MB, Eyes > 700MB
        return brain.exists() && brain.length() > 900_000_000L &&
                eyes.exists() && eyes.length() > 700_000_000L
    }

    fun deleteModels() {
        File(modelDir, "ggml-model-Q4_K_M.gguf").delete()
        File(modelDir, "mmproj-model-f16.gguf").delete()
    }
}