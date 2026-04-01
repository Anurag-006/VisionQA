package com.anurag.visionqa.ai

import android.util.Log

object LlamaJNI {
    init {
        try {
            Log.d("LlamaJNI", "--- Loading ggml-base ---")
            System.loadLibrary("ggml-base")
            Log.d("LlamaJNI", "✅ ggml-base loaded")

            Log.d("LlamaJNI", "--- Loading ggml-cpu ---")
            System.loadLibrary("ggml-cpu")
            Log.d("LlamaJNI", "✅ ggml-cpu loaded")

            Log.d("LlamaJNI", "--- Loading ggml core ---")
            System.loadLibrary("ggml")
            Log.d("LlamaJNI", "✅ ggml loaded")

            Log.d("LlamaJNI", "--- Loading llama ---")
            System.loadLibrary("llama")
            Log.d("LlamaJNI", "✅ llama loaded")

            Log.d("LlamaJNI", "--- Loading multimodal bridge (mtmd) ---")
            System.loadLibrary("mtmd")
            Log.d("LlamaJNI", "✅ mtmd loaded")

            Log.d("LlamaJNI", "--- Loading JNI wrapper (visionqa_jni) ---")
            System.loadLibrary("visionqa_jni")
            Log.d("LlamaJNI", "✅ visionqa_jni loaded")

            Log.d("LlamaJNI", "🎉 ALL modular native libraries loaded successfully")

        } catch (e: UnsatisfiedLinkError) {
            Log.e("LlamaJNI", "❌ LIBRARY LOAD FAILED: ${e.message}")
            Log.e("LlamaJNI", "❌ Full error:", e)
        } catch (e: Exception) {
            Log.e("LlamaJNI", "❌ Unexpected error loading libraries: ${e.message}", e)
        }
    }
    /**
     * Loads the language model and multimodal projector.
     * Safe to call multiple times — frees existing state before reloading.
     */
    external fun loadModel(
        modelPath:  String,
        mmprojPath: String,
        threads:    Int,
        ctx:        Int
    ): Boolean

    /**
     * Runs vision+language inference.
     * Blocks the calling thread — always call from Dispatchers.IO.
     *
     * @param pixelData     Raw ARGB pixels from Bitmap.getPixels()
     * @param imageWidth    Width of the bitmap
     * @param imageHeight   Height of the bitmap
     * @param prompt        User question — plain text, no template wrapping
     * @param maxTokens     Hard cap on output tokens
     * @param tokenCallback Called once per generated token piece for streaming
     */
    external fun generate(
        pixelData:     IntArray,
        imageWidth:    Int,
        imageHeight:   Int,
        prompt:        String,
        maxTokens:     Int,
        tokenCallback: (ByteArray) -> Unit
    ): String

    /**
     * Signals the C++ generation loop to stop immediately.
     * Thread-safe — uses std::atomic internally, returns instantly.
     */
    external fun abortGeneration()

    /**
     * Frees all native model state.
     * Blocks until any in-progress generation finishes.
     */
    external fun freeModel()
}