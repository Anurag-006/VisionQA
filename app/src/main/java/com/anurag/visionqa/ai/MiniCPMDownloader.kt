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

    val modelDir = File(context.getExternalFilesDir(null), "minicpm")

    companion object {
        private const val TAG = "MiniCPMDownloader"

        // MiniCPM-V 2.0 Q4_K_M — ~1.9GB model, ~400MB mmproj
        // Works on Redmi 13 5G (8GB) AND iQOO Neo 10 (16GB)
        private const val BRAIN_URL   = "https://huggingface.co/openbmb/MiniCPM-V-2_6-gguf/resolve/main/ggml-model-Q2_K.gguf"
        private const val EYES_URL  = "https://huggingface.co/openbmb/MiniCPM-V-2_6-gguf/resolve/main/mmproj-model-f16.gguf"        // Exact filenames the loader expects — do not change these
        private const val LOCAL_BRAIN = "ggml-model-Q2_K.gguf"
        private const val LOCAL_EYES  = "mmproj-model-f16.gguf"

        // MiniCPM-V 2.0 Q4_K_M sizes:
        // Brain: ~1.85GB  →  min check: 1.8GB
        // Eyes:  ~390MB   →  min check: 350MB
        private const val MIN_BRAIN_BYTES = 2_200_000_000L  // ~2.4GB
        private const val MIN_EYES_BYTES  =   800_000_000L    }

    init {
        if (!modelDir.exists()) modelDir.mkdirs()
    }

    suspend fun downloadIfNeeded(
        onProgress: (String, Int) -> Unit
    ): Boolean = withContext(Dispatchers.IO) {

        val brainFile = File(modelDir, LOCAL_BRAIN)
        val eyesFile  = File(modelDir, LOCAL_EYES)

        if (modelsExist()) {
            Log.d(TAG, "✅ Models already present.")
            onProgress("MiniCPM-V 2.0 ready (~2.2GB)", 100)
            return@withContext true
        }

        Log.d(TAG, "Starting download. Dir: ${modelDir.absolutePath}")

        try {
            // 1. Download Brain (~1.85GB)
            if (!brainFile.exists() || brainFile.length() < MIN_BRAIN_BYTES) {
                onProgress("Downloading MiniCPM Brain (~1.85GB)...", 0)
                Log.d(TAG, "Downloading brain from: $BRAIN_URL")
                downloadFile(
                    url         = BRAIN_URL,
                    destination = brainFile,
                    onProgress  = { p -> onProgress("Brain: $p%", (p * 0.75).toInt()) }
                )
                Log.d(TAG, "Brain downloaded: ${brainFile.length() / 1_000_000}MB")
            }

            // 2. Download Eyes (~390MB)
            if (!eyesFile.exists() || eyesFile.length() < MIN_EYES_BYTES) {
                onProgress("Downloading Vision Projector (~390MB)...", 75)
                Log.d(TAG, "Downloading mmproj from: $EYES_URL")
                downloadFile(
                    url         = EYES_URL,
                    destination = eyesFile,
                    onProgress  = { p -> onProgress("Vision: $p%", 75 + (p * 0.25).toInt()) }
                )
                Log.d(TAG, "Eyes downloaded: ${eyesFile.length() / 1_000_000}MB")
            }

            Log.d(TAG, "✅ All MiniCPM-V 2.0 downloads complete!")
            onProgress("Download complete! Initializing model...", 100)
            true

        } catch (e: Exception) {
            Log.e(TAG, "❌ Download failed: ${e.message}", e)
            onProgress("Download failed: ${e.message}", 0)
            // Clean up partial files so next launch retries
            if (brainFile.exists() && brainFile.length() < MIN_BRAIN_BYTES) brainFile.delete()
            if (eyesFile.exists()  && eyesFile.length()  < MIN_EYES_BYTES)  eyesFile.delete()
            false
        }
    }

    private fun downloadFile(url: String, destination: File, onProgress: (Int) -> Unit) {
        val request  = Request.Builder().url(url).build()
        val response = client.newCall(request).execute()

        if (!response.isSuccessful) throw Exception("HTTP ${response.code}: ${response.message}")

        val body          = response.body ?: throw Exception("Empty response body")
        val contentLength = body.contentLength()

        body.byteStream().use { input ->
            FileOutputStream(destination).use { output ->
                val buffer      = ByteArray(128 * 1024) // 128KB chunks
                var totalBytes  = 0L
                var bytes: Int
                var lastProgress = -1

                while (input.read(buffer).also { bytes = it } != -1) {
                    output.write(buffer, 0, bytes)
                    totalBytes += bytes

                    if (contentLength > 0) {
                        val progress = ((totalBytes * 100) / contentLength).toInt()
                        if (progress != lastProgress) {
                            onProgress(progress)
                            lastProgress = progress
                        }
                    }
                }
                output.flush()
                onProgress(100)
            }
        }
    }

    fun modelsExist(): Boolean {
        val brain = File(modelDir, LOCAL_BRAIN)
        val eyes  = File(modelDir, LOCAL_EYES)
        val ok = brain.exists() && brain.length() >= MIN_BRAIN_BYTES &&
                eyes.exists()  && eyes.length()  >= MIN_EYES_BYTES
        Log.d(TAG, "modelsExist=$ok brain=${brain.length()/1_000_000}MB eyes=${eyes.length()/1_000_000}MB")
        return ok
    }

    fun deleteModels() {
        File(modelDir, LOCAL_BRAIN).delete()
        File(modelDir, LOCAL_EYES).delete()
        Log.d(TAG, "Models deleted.")
    }
}