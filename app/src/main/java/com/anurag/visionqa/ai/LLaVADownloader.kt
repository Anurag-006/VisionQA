//package com.anurag.visionqa.ai
//
//import android.content.Context
//import android.util.Log
//import kotlinx.coroutines.Dispatchers
//import kotlinx.coroutines.withContext
//import okhttp3.OkHttpClient
//import okhttp3.Request
//import java.io.File
//import java.io.FileOutputStream
//import java.util.concurrent.TimeUnit
//
//class LLaVADownloader(private val context: Context) {
//
//    private val client = OkHttpClient.Builder()
//        .connectTimeout(60, TimeUnit.SECONDS)
//        .readTimeout(60, TimeUnit.SECONDS)
//        .writeTimeout(60, TimeUnit.SECONDS)
//        .build()
//
//    private val modelDir = File(context.filesDir, "models")
//
//    companion object {
//        private const val TAG = "LLaVADownloader"
//    }
//
//    init {
//        if (!modelDir.exists()) {
//            modelDir.mkdirs()
//        }
//    }
//
//    suspend fun downloadIfNeeded(
//        onProgress: (String, Int) -> Unit
//    ): Boolean = withContext(Dispatchers.IO) {
//
//        val modelFile = File(modelDir, "bakllava-q4.gguf")
//        val mmProjFile = File(modelDir, "bakllava-mmproj-f16.gguf")
//
//        // Check if already downloaded and valid
//        if (modelFile.exists() && mmProjFile.exists()) {
//            val modelSize = modelFile.length() / (1024 * 1024)
//            val mmProjSize = mmProjFile.length() / (1024 * 1024)
//
//            Log.d(TAG, "Found existing models: ${modelSize}MB + ${mmProjSize}MB")
//
//            if (modelSize > 50 && mmProjSize > 50) {
//                onProgress("Models ready (${modelSize}MB + ${mmProjSize}MB)", 100)
//                return@withContext true
//            } else {
//                Log.d(TAG, "Models too small, re-downloading")
//                modelFile.delete()
//                mmProjFile.delete()
//            }
//        }
//
//        try {
//            // Download BakLLaVA text model (~600MB Q4)
//            if (!modelFile.exists()) {
//                onProgress("Downloading BakLLaVA model (600MB)...", 0)
//
//                val modelUrl = "https://huggingface.co/mys/ggml_bakllava-1/resolve/main/ggml-model-q4_k.gguf"
//
//                Log.d(TAG, "Downloading model from: $modelUrl")
//
//                downloadFileWithRetry(
//                    url = modelUrl,
//                    destination = modelFile,
//                    onProgress = { progress ->
//                        onProgress("BakLLaVA model: $progress%", (progress * 0.7).toInt())
//                    }
//                )
//            }
//
//            // Download vision projector (~240MB)
//            if (!mmProjFile.exists()) {
//                onProgress("Downloading vision encoder (240MB)...", 70)
//
//                val mmProjUrl = "https://huggingface.co/mys/ggml_bakllava-1/resolve/main/mmproj-model-f16.gguf"
//
//                Log.d(TAG, "Downloading mmproj from: $mmProjUrl")
//
//                downloadFileWithRetry(
//                    url = mmProjUrl,
//                    destination = mmProjFile,
//                    onProgress = { progress ->
//                        onProgress("Vision encoder: $progress%", 70 + (progress * 0.3).toInt())
//                    }
//                )
//            }
//
//            Log.d(TAG, "Download complete!")
//            onProgress("Download complete!", 100)
//            true
//
//        } catch (e: Exception) {
//            Log.e(TAG, "Download failed", e)
//            onProgress("Download failed: ${e.message}", 0)
//
//            // Cleanup partial downloads
//            modelFile.delete()
//            mmProjFile.delete()
//
//            false
//        }
//    }
//
//    private fun downloadFileWithRetry(
//        url: String,
//        destination: File,
//        maxRetries: Int = 3,
//        onProgress: (Int) -> Unit
//    ) {
//        var attempt = 0
//        var lastException: Exception? = null
//
//        while (attempt < maxRetries) {
//            try {
//                downloadFile(url, destination, onProgress)
//                return
//            } catch (e: Exception) {
//                lastException = e
//                attempt++
//
//                Log.w(TAG, "Download attempt $attempt failed: ${e.message}")
//
//                if (attempt < maxRetries) {
//                    Log.d(TAG, "Retrying in 2 seconds...")
//                    Thread.sleep(2000)
//                    destination.delete()
//                }
//            }
//        }
//
//        throw lastException ?: Exception("Download failed after $maxRetries attempts")
//    }
//
//    private fun downloadFile(
//        url: String,
//        destination: File,
//        onProgress: (Int) -> Unit
//    ) {
//        Log.d(TAG, "Starting download: $url")
//        Log.d(TAG, "Destination: ${destination.absolutePath}")
//
//        val request = Request.Builder().url(url).build()
//        val response = client.newCall(request).execute()
//
//        if (!response.isSuccessful) {
//            throw Exception("HTTP ${response.code}: ${response.message}")
//        }
//
//        val body = response.body ?: throw Exception("Empty response body")
//        val contentLength = body.contentLength()
//
//        Log.d(TAG, "Content length: ${contentLength / (1024 * 1024)}MB")
//
//        body.byteStream().use { input ->
//            FileOutputStream(destination).use { output ->
//                val buffer = ByteArray(8192)
//                var totalBytes = 0L
//                var bytes: Int
//                var lastProgress = 0
//                var lastLog = System.currentTimeMillis()
//
//                while (input.read(buffer).also { bytes = it } != -1) {
//                    output.write(buffer, 0, bytes)
//                    totalBytes += bytes
//
//                    if (contentLength > 0) {
//                        val progress = ((totalBytes * 100) / contentLength).toInt()
//
//                        // Update progress every 2% or 5 seconds
//                        val now = System.currentTimeMillis()
//                        if (progress >= lastProgress + 2 || now - lastLog > 5000) {
//                            onProgress(progress)
//                            lastProgress = progress
//                            lastLog = now
//
//                            Log.d(TAG, "Progress: ${totalBytes / (1024 * 1024)}MB / ${contentLength / (1024 * 1024)}MB ($progress%)")
//                        }
//                    }
//                }
//
//                output.flush()
//                onProgress(100)
//
//                Log.d(TAG, "Download complete: ${destination.absolutePath}")
//                Log.d(TAG, "File size: ${destination.length() / (1024 * 1024)}MB")
//            }
//        }
//    }
//
//    fun getModelPath(): String = File(modelDir, "bakllava-q4.gguf").absolutePath
//    fun getMMProjPath(): String = File(modelDir, "bakllava-mmproj-f16.gguf").absolutePath
//
//    fun modelsExist(): Boolean {
//        val model = File(modelDir, "bakllava-q4.gguf")
//        val mmproj = File(modelDir, "bakllava-mmproj-f16.gguf")
//
//        return model.exists() && model.length() > 50_000_000 &&
//                mmproj.exists() && mmproj.length() > 50_000_000
//    }
//
//    fun deleteModels() {
//        File(modelDir, "bakllava-q4.gguf").delete()
//        File(modelDir, "bakllava-mmproj-f16.gguf").delete()
//        Log.d(TAG, "Models deleted")
//    }
//}