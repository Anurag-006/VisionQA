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
//import java.io.InputStream
//import java.util.concurrent.TimeUnit
//
//class Phi3Downloader(private val context: Context) {
//
//    private val client = OkHttpClient.Builder()
//        .connectTimeout(60, TimeUnit.SECONDS)
//        .readTimeout(60, TimeUnit.SECONDS)
//        .build()
//
//    private val modelDir = File(context.filesDir, "models")
//
//    // Define the file pairs (onnx + data)
//    private val modelFiles = listOf(
//        "phi-3-v-128k-instruct-text.onnx",
//        "phi-3-v-128k-instruct-text.onnx.data",
//        "phi-3-v-128k-instruct-vision.onnx",
//        "phi-3-v-128k-instruct-vision.onnx.data",
//        "phi-3-v-128k-instruct-embedding.onnx",
//        "phi-3-v-128k-instruct-embedding.onnx.data"
//    )
//
//    private val baseUrl = "https://huggingface.co/microsoft/Phi-3-vision-128k-instruct-onnx/resolve/main/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"
//
//    companion object {
//        private const val TAG = "Phi3Downloader"
//    }
//
//    init {
//        if (!modelDir.exists()) modelDir.mkdirs()
//    }
//
//    suspend fun downloadIfNeeded(onProgress: (String, Int) -> Unit): Boolean = withContext(Dispatchers.IO) {
//        try {
//            modelFiles.forEachIndexed { index, fileName ->
//                val destination = File(modelDir, fileName)
//                val url = "$baseUrl/$fileName"
//
//                // Calculate cumulative progress
//                val baseProgress = (index.toFloat() / modelFiles.size * 100).toInt()
//                val stepWeight = 100 / modelFiles.size
//
//                downloadFileWithResume(url, destination) { fileProgress ->
//                    val totalProgress = baseProgress + (fileProgress * stepWeight / 100)
//                    onProgress("Downloading $fileName: $fileProgress%", totalProgress)
//                }
//            }
//            onProgress("All models ready", 100)
//            true
//        } catch (e: Exception) {
//            Log.e(TAG, "Download failed", e)
//            onProgress("Error: ${e.message}", 0)
//            false
//        }
//    }
//
//    private fun downloadFileWithResume(url: String, destination: File, onProgress: (Int) -> Unit) {
//        val existingSize = if (destination.exists()) destination.length() else 0L
//
//        val request = Request.Builder()
//            .url(url)
//            .header("Range", "bytes=$existingSize-")
//            .build()
//
//        client.newCall(request).execute().use { response ->
//            if (response.code == 416) { // Already complete
//                onProgress(100)
//                return
//            }
//            if (!response.isSuccessful) throw Exception("HTTP ${response.code}")
//
//            val body = response.body ?: throw Exception("Empty body")
//            val isResume = response.code == 206
//            val totalBytesToDownload = body.contentLength()
//            val fullSize = if (isResume) existingSize + totalBytesToDownload else totalBytesToDownload
//
//            val outputStream = FileOutputStream(destination, isResume)
//            body.byteStream().use { input ->
//                outputStream.use { output ->
//                    val buffer = ByteArray(65536)
//                    var bytesRead: Int
//                    var totalDownloaded = if (isResume) existingSize else 0L
//                    var lastProgress = -1
//
//                    while (input.read(buffer).also { bytesRead = it } != -1) {
//                        output.write(buffer, 0, bytesRead)
//                        totalDownloaded += bytesRead
//                        val progress = ((totalDownloaded * 100) / fullSize).toInt()
//                        if (progress != lastProgress) {
//                            onProgress(progress)
//                            lastProgress = progress
//                        }
//                    }
//                }
//            }
//        }
//    }
//
//    // Return the main text model path for ONNX initialization
//    fun getModelPath(): String = File(modelDir, "phi-3-v-128k-instruct-text.onnx").absolutePath
//    fun getTextPath(): String = File(modelDir, "phi-3-v-128k-instruct-text.onnx").absolutePath
//    fun getVisionPath(): String = File(modelDir, "phi-3-v-128k-instruct-vision.onnx").absolutePath
//    fun getEmbedPath(): String = File(modelDir, "phi-3-v-128k-instruct-embedding.onnx").absolutePath
//}