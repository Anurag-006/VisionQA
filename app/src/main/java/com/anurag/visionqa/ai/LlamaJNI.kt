package com.anurag.visionqa.ai

import android.util.Log

object LlamaJNI {
    init {
        try {
            System.loadLibrary("omp")
            System.loadLibrary("ggml-base")
            System.loadLibrary("ggml-cpu")

            // Only load Vulkan on Adreno 7xx (Snapdragon 8 Gen 1+)
            // Adreno 613 (Redmi 13 5G) does not support fp16 and will reject the model
            val renderer = try {
                val egl = android.opengl.EGL14.eglGetDisplay(android.opengl.EGL14.EGL_DEFAULT_DISPLAY)
                android.opengl.EGL14.eglInitialize(egl, null, 0, null, 0)
                android.os.Build.HARDWARE  // fallback identifier
            } catch (e: Exception) { "" }

            val board = android.os.Build.BOARD.lowercase()
            val hardware = android.os.Build.HARDWARE.lowercase()
            val isCapableGPU = board.contains("kalama") ||   // Snapdragon 8 Gen 2 (iQOO Neo 9 Pro)
                    board.contains("crow") ||      // Snapdragon 8 Gen 3
                    hardware.contains("kalama")

//            if (isCapableGPU) {
//                System.loadLibrary("ggml-vulkan")
//                Log.d("LlamaJNI", "✅ Vulkan enabled for $board")
//            } else {
//                Log.d("LlamaJNI", "⚠️ Vulkan skipped — unsupported GPU ($board)")
//            }

            System.loadLibrary("ggml")
            System.loadLibrary("llama")
            System.loadLibrary("mtmd")
            System.loadLibrary("visionqa_jni")
            Log.d("LlamaJNI", "✅ All native libraries loaded.")
        } catch (e: UnsatisfiedLinkError) {
            Log.e("LlamaJNI", "❌ LIBRARY LOAD FAILED: ${e.message}")
        }
    }
    /**
     * Loads the language model and multimodal projector.
     *
     * @param modelPath    Absolute path to the GGUF language model file.
     * @param mmprojPath   Absolute path to the GGUF multimodal projector file.
     * @param threads      Threads for token generation. On Snapdragon 8 Gen 2
     *                     use 5 (1 X3 prime + 4 A715 perf cores).
     * @param threadsBatch Threads for prefill / image encoding. Same as [threads]
     *                     on Snapdragon 8 Gen 2 — all big cores are useful here.
     * @param ctx          KV cache context size in tokens (8192 recommended for
     *                     Neo 9 Pro which has 12–16 GB RAM).
     */
    external fun loadModel(
        modelPath:    String,
        mmprojPath:   String,
        threads:      Int,
        threadsBatch: Int,
        ctx:          Int
    ): Boolean

    /**
     * Runs vision+language inference.
     * Blocks the calling thread — always call from Dispatchers.IO.
     *
     * On the FIRST call per image, pass an empty string for [history].
     * The C++ side will encode the image and save the KV cache position.
     *
     * On FOLLOW-UP calls, pass all prior assistant turns formatted as:
     *   "<|im_start|>assistant\nANSWER<|im_end|>\n"
     * The C++ side rewinds the cache to after the image and replays
     * history as cheap text — skipping image re-encoding entirely.
     *
     * Call [resetCache] before the first question on a NEW image.
     *
     * @param pixelData     Raw ARGB pixels from Bitmap.getPixels()
     * @param imageWidth    Width of the bitmap
     * @param imageHeight   Height of the bitmap
     * @param prompt        Current user question — plain text, no template wrapping
     * @param history       All prior assistant turns formatted as template strings
     * @param maxTokens     Hard cap on output tokens
     * @param tokenCallback Called once per generated token piece for streaming
     */
    external fun generate(
        pixelData:     IntArray,
        imageWidth:    Int,
        imageHeight:   Int,
        prompt:        String,
        history:       String,
        maxTokens:     Int,
        tokenCallback: (ByteArray) -> Unit
    ): String

    /**
     * Rewinds the KV cache to the cold state.
     * Call this when the user picks a new image so the next generate()
     * re-encodes the new image from scratch.
     */
    external fun resetCache()

    /** Signals the C++ generation loop to stop immediately. Thread-safe. */
    external fun abortGeneration()

    /** Frees all native model state. */
    external fun freeModel()
}