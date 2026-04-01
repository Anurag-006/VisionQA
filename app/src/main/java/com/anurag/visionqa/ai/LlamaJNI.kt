package com.anurag.visionqa.ai

import android.util.Log

object LlamaJNI {
    init {
        try {
            System.loadLibrary("ggml-base")
            System.loadLibrary("ggml-cpu")
            System.loadLibrary("ggml")
            System.loadLibrary("llama")
            System.loadLibrary("mtmd")
            System.loadLibrary("visionqa_jni")
            Log.d("LlamaJNI", "✅ All native libraries loaded.")
        } catch (e: UnsatisfiedLinkError) {
            Log.e("LlamaJNI", "❌ LIBRARY LOAD FAILED: ${e.message}")
        } catch (e: Exception) {
            Log.e("LlamaJNI", "❌ Unexpected error: ${e.message}", e)
        }
    }

    /** Loads the language model and multimodal projector. */
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