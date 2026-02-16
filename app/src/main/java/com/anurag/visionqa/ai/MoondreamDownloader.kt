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

class MoondreamDownloader(private val context: Context) {

    private val client = OkHttpClient.Builder()
        .connectTimeout(120, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .writeTimeout(120, TimeUnit.SECONDS)
        .build()

    private val modelDir = File(context.filesDir, "models")

    companion object {
        private const val TAG = "MoondreamDownloader"
    }

    init {
        if (!modelDir.exists()) {
            modelDir.mkdirs()
        }
    }

    suspend fun downloadIfNeeded(
        onProgress: (String, Int) -> Unit
    ): Boolean = withContext(Dispatchers.IO) {

        val visionFile = File(modelDir, "vision_encoder.onnx")
        val embedFile = File(modelDir, "embed_tokens.onnx")
        val decoderFile = File(modelDir, "decoder_model.onnx")
        val tokenizerFile = File(modelDir, "tokenizer.json")

        // Clean up the old monolithic model to save phone storage!
        val oldModel = File(modelDir, "moondream-q4.onnx")
        if (oldModel.exists()) {
            oldModel.delete()
            Log.d(TAG, "Deleted incompatible monolithic model.")
        }

        // Check if all 4 files exist
        if (modelsExist()) {
            onProgress("Models ready (~1.5GB)", 100)
            return@withContext true
        }

        try {
            // 1. Download Vision Encoder (280MB)
            if (!visionFile.exists() || visionFile.length() < 100_000_000) {
                onProgress("Downloading Vision AI (280MB)...", 0)
                downloadFile(
                    url = "https://huggingface.co/Xenova/moondream2/resolve/main/onnx/vision_encoder_q4.onnx",
                    destination = visionFile,
                    onProgress = { p -> onProgress("Vision AI: $p%", (p * 0.2).toInt()) }
                )
            }

            // 2. Download Text Embedder (419MB)
            if (!embedFile.exists() || embedFile.length() < 100_000_000) {
                onProgress("Downloading Text Embedder (419MB)...", 20)
                downloadFile(
                    url = "https://huggingface.co/Xenova/moondream2/resolve/main/onnx/embed_tokens_q4.onnx",
                    destination = embedFile,
                    onProgress = { p -> onProgress("Text Embedder: $p%", 20 + (p * 0.2).toInt()) }
                )
            }

            // 3. Download Main Decoder (824MB)
            if (!decoderFile.exists() || decoderFile.length() < 100_000_000) {
                onProgress("Downloading Main Decoder (824MB)...", 40)
                downloadFile(
                    url = "https://huggingface.co/Xenova/moondream2/resolve/main/onnx/decoder_model_merged_q4.onnx",
                    destination = decoderFile,
                    onProgress = { p -> onProgress("Main Decoder: $p%", 40 + (p * 0.55).toInt()) }
                )
            }

            // 4. Download Tokenizer (~2MB)
            if (!tokenizerFile.exists() || tokenizerFile.length() < 1000) {
                onProgress("Downloading Tokenizer...", 95)
                downloadFile(
                    url = "https://huggingface.co/vikhyatk/moondream2/resolve/main/tokenizer.json",
                    destination = tokenizerFile,
                    onProgress = { p -> onProgress("Tokenizer: $p%", 95 + (p * 0.05).toInt()) }
                )
            }

            Log.d(TAG, "✅ All downloads complete!")
            onProgress("Download complete!", 100)
            true

        } catch (e: Exception) {
            Log.e(TAG, "❌ Download failed", e)
            onProgress("Download failed: ${e.message}", 0)
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
                val buffer = ByteArray(8192)
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

                        if (progress >= lastProgress + 2 || now - lastLogTime > 5000) {
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

    // --- CRITICAL PATHS FOR KV CACHE MOONDREAMVLM.KT ---
    fun getVisionPath(): String = File(modelDir, "vision_encoder.onnx").absolutePath
    fun getEmbedPath(): String = File(modelDir, "embed_tokens.onnx").absolutePath
    fun getDecoderPath(): String = File(modelDir, "decoder_model.onnx").absolutePath

    fun getTokenizerPath(): String = File(modelDir, "tokenizer.json").absolutePath

    fun modelsExist(): Boolean {
        val vision = File(modelDir, "vision_encoder.onnx")
        val embed = File(modelDir, "embed_tokens.onnx")
        val decoder = File(modelDir, "decoder_model.onnx")
        val tokenizer = File(modelDir, "tokenizer.json")

        return vision.exists() && embed.exists() && decoder.exists() && tokenizer.exists()
    }

    fun deleteModels() {
        try {
            File(modelDir, "vision_encoder.onnx").delete()
            File(modelDir, "embed_tokens.onnx").delete()
            File(modelDir, "decoder_model.onnx").delete()
            File(modelDir, "tokenizer.json").delete()
            Log.d(TAG, "✅ Deleted all model files")
        } catch (e: Exception) {
            Log.e(TAG, "Error deleting models", e)
        }
    }

    fun getModelInfo(): String {
        return buildString {
            appendLine("=== Split ONNX Model Files ===")
            appendLine("Location: ${modelDir.absolutePath}")
            appendLine("Status: ${if (modelsExist()) "✅ Ready" else "❌ Missing"}")
        }
    }
}