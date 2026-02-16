//package com.anurag.visionqa.ai
//
//import android.content.Context
//import android.graphics.Bitmap
//import android.util.Log
//import kotlinx.coroutines.*
//import java.io.File
//import java.io.FileOutputStream
//
//class LLaVAVLM(private val context: Context) : VLMInterface {
//
//    private var nativePtr: Long = 0
//    private val downloader = LLaVADownloader(context)
//    private var libraryLoaded = false
//
//    companion object {
//        private const val TAG = "LLaVAVLM"
//
//        init {
//            try {
//                Log.d(TAG, "Loading native library...")
//                System.loadLibrary("llava_jni")
//                Log.d(TAG, "✅ Native library loaded successfully")
//            } catch (e: UnsatisfiedLinkError) {
//                Log.e(TAG, "❌ Failed to load native library", e)
//                Log.e(TAG, "Library name: llava_jni")
//                Log.e(TAG, "Error: ${e.message}")
//            } catch (e: Exception) {
//                Log.e(TAG, "❌ Unexpected error loading library", e)
//            }
//        }
//    }
//
//    suspend fun initialize(
//        onProgress: (String, Int) -> Unit = { _, _ -> }
//    ): Boolean = withContext(Dispatchers.IO) {
//        try {
//            Log.d(TAG, "=== INITIALIZATION START ===")
//            Log.d(TAG, "Thread: ${Thread.currentThread().name}")
//
//            libraryLoaded = true
//
//            Log.d(TAG, "Downloading models...")
//            val downloaded = downloader.downloadIfNeeded(onProgress)
//
//            if (!downloaded) {
//                Log.e(TAG, "❌ Download failed")
//                return@withContext false
//            }
//
//            val modelPath = downloader.getModelPath()
//            val mmProjPath = downloader.getMMProjPath()
//
//            Log.d(TAG, "Model: $modelPath")
//            Log.d(TAG, "MMProj: $mmProjPath")
//
//            // Verify files
//            val modelFile = File(modelPath)
//            val mmProjFile = File(mmProjPath)
//
//            Log.d(TAG, "Model exists: ${modelFile.exists()}, size: ${modelFile.length()}")
//            Log.d(TAG, "MMProj exists: ${mmProjFile.exists()}, size: ${mmProjFile.length()}")
//
//            Log.d(TAG, "About to call nativeInit...")
//            Log.d(TAG, "  modelPath: $modelPath")
//            Log.d(TAG, "  mmProjPath: $mmProjPath")
//            Log.d(TAG, "  nCtx: 512")
//            Log.d(TAG, "  nThreads: 2")
//
//            try {
//                Log.d(TAG, ">>> Calling nativeInit (this may take 30-60 seconds)...")
//
//                nativePtr = nativeInit(
//                    modelPath = modelPath,
//                    mmprojPath = mmProjPath,
//                    nCtx = 512,
//                    nThreads = 2
//                )
//
//                Log.d(TAG, "<<< nativeInit returned: $nativePtr")
//
//            } catch (e: UnsatisfiedLinkError) {
//                Log.e(TAG, "❌ nativeInit method not found!", e)
//                return@withContext false
//            } catch (e: Exception) {
//                Log.e(TAG, "❌ nativeInit threw exception", e)
//                return@withContext false
//            }
//
//            if (nativePtr == 0L) {
//                Log.e(TAG, "❌ nativeInit returned 0 (failed)")
//                return@withContext false
//            }
//
//            Log.d(TAG, "✅ Initialization complete: $nativePtr")
//            Log.d(TAG, "=== INITIALIZATION COMPLETE ===")
//            true
//
//        } catch (e: Exception) {
//            Log.e(TAG, "❌ Initialization exception", e)
//            false
//        }
//    }
//
//    override suspend fun initialize(): Boolean = initialize { _, _ -> }
//
//    override fun isReady(): Boolean {
//        val ready = nativePtr != 0L && libraryLoaded
//        Log.d(TAG, "isReady: $ready (ptr=$nativePtr, lib=$libraryLoaded)")
//        return ready
//    }
//
//    // Test function - text only
//    suspend fun testTextOnly(prompt: String): String = withContext(Dispatchers.IO) {
//        Log.d(TAG, "=== TEST TEXT ONLY START ===")
//        Log.d(TAG, "Thread: ${Thread.currentThread().name}")
//        Log.d(TAG, "nativePtr: $nativePtr")
//        Log.d(TAG, "libraryLoaded: $libraryLoaded")
//        Log.d(TAG, "Prompt: $prompt")
//
//        if (nativePtr == 0L) {
//            Log.e(TAG, "❌ Not initialized (nativePtr=0)")
//            return@withContext "Error: Not initialized"
//        }
//
//        if (!libraryLoaded) {
//            Log.e(TAG, "❌ Library not loaded")
//            return@withContext "Error: Library not loaded"
//        }
//
//        try {
//            Log.d(TAG, "About to call nativeGenerateTextOnly...")
//            Log.d(TAG, "  ptr: $nativePtr")
//            Log.d(TAG, "  prompt: $prompt")
//
//            Log.d(TAG, ">>> Entering native method...")
//
//            val result = nativeGenerateTextOnly(nativePtr, prompt)
//
//            Log.d(TAG, "<<< Returned from native method")
//            Log.d(TAG, "Result: $result")
//            Log.d(TAG, "=== TEST TEXT ONLY COMPLETE ===")
//
//            result
//
//        } catch (e: UnsatisfiedLinkError) {
//            Log.e(TAG, "❌ Native method not found!", e)
//            Log.e(TAG, "Method signature: nativeGenerateTextOnly(Long, String)")
//            "Error: Native method not found - ${e.message}"
//        } catch (e: Exception) {
//            Log.e(TAG, "❌ Exception during native call", e)
//            Log.e(TAG, "Exception type: ${e.javaClass.name}")
//            Log.e(TAG, "Message: ${e.message}")
//            Log.e(TAG, "Stack trace:", e)
//            "Error: ${e.javaClass.simpleName} - ${e.message}"
//        }
//    }
//
//    override suspend fun chat(
//        image: Bitmap,
//        question: String,
//        conversationHistory: List<Pair<String, String>>
//    ): String = withContext(Dispatchers.IO) {
//        Log.d(TAG, "=== CHAT START ===")
//        "Not implemented yet - use testTextOnly()"
//    }
//
//    override fun getModelInfo(): ModelInfo {
//        return ModelInfo(
//            name = "BakLLaVA (TinyLlama 1.1B)",
//            size = "~840MB",
//            speed = "1-3 seconds",
//            quality = "Good for testing"
//        )
//    }
//
//    override fun cleanup() {
//        Log.d(TAG, "=== CLEANUP START ===")
//        if (nativePtr != 0L) {
//            try {
//                Log.d(TAG, "Calling nativeDestroy($nativePtr)...")
//                nativeDestroy(nativePtr)
//                Log.d(TAG, "✅ Cleanup complete")
//            } catch (e: Exception) {
//                Log.e(TAG, "❌ Cleanup error", e)
//            }
//            nativePtr = 0
//        }
//        Log.d(TAG, "=== CLEANUP COMPLETE ===")
//    }
//
//    fun testNative(): String {
//        Log.d(TAG, "=== TEST NATIVE START ===")
//        return try {
//            Log.d(TAG, "Calling nativeTest()...")
//            val result = nativeTest()
//            Log.d(TAG, "nativeTest returned: $result")
//            result
//        } catch (e: UnsatisfiedLinkError) {
//            Log.e(TAG, "❌ nativeTest not found!", e)
//            "Error: nativeTest method not found"
//        } catch (e: Exception) {
//            Log.e(TAG, "❌ nativeTest exception", e)
//            "Error: ${e.message}"
//        }
//    }
//
//    private external fun nativeTest(): String
//    private external fun nativeInit(modelPath: String, mmprojPath: String, nCtx: Int, nThreads: Int): Long
//    private external fun nativeGenerate(ptr: Long, imagePath: String, prompt: String): String
//    private external fun nativeGenerateTextOnly(ptr: Long, prompt: String): String
//    private external fun nativeDestroy(ptr: Long)
//}