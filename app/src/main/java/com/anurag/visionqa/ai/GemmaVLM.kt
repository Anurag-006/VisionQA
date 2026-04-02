package com.anurag.visionqa.ai

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.genai.llminference.GraphOptions
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.google.mediapipe.tasks.genai.llminference.LlmInference.LlmInferenceOptions
import com.google.mediapipe.tasks.genai.llminference.LlmInferenceSession
import com.google.mediapipe.tasks.genai.llminference.LlmInferenceSession.LlmInferenceSessionOptions
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

/**
 * VisionQA inference engine — powered by MediaPipe LLM Inference API.
 *
 * Internal implementation detail: uses Gemma 3n E2B under the hood.
 * All user-facing strings reference "VisionQA" only.
 */
class GemmaVLM(private val context: Context) : VLMInterface {

    private var engine: LlmInference? = null
    private var currentSession: LlmInferenceSession? = null
    private val downloader = GemmaDownloader(context)
    private var ready = false

    companion object {
        private const val TAG = "VisionQA"
        private const val MAX_TOKENS = 1024
        private const val MAX_IMAGE_DIM = 768
    }

    override suspend fun initialize(): Boolean = initialize { _, _ -> }

    override suspend fun initialize(onProgress: (String, Int) -> Unit): Boolean =
        withContext(Dispatchers.IO) {
            if (!downloader.downloadIfNeeded { msg, pct -> onProgress(msg, pct) })
                return@withContext false

            onProgress("Starting VisionQA engine...", 90)
            engine = buildEngine()

            if (engine == null) {
                onProgress("Engine failed to start. Please restart the app.", 0)
                return@withContext false
            }

            ready = true
            onProgress("Ready!", 100)
            true
        }

    override fun abort() {
        try { currentSession?.close() } catch (_: Exception) {}
        currentSession = null
    }

    private fun buildEngine(): LlmInference? {
        return try {
            val options = LlmInferenceOptions.builder()
                .setModelPath(downloader.getModelPath())
                .setMaxTokens(MAX_TOKENS)
                .setMaxNumImages(1)
                .setPreferredBackend(LlmInference.Backend.GPU)
                .build()
            LlmInference.createFromOptions(context, options)
        } catch (e: Exception) {
            Log.e(TAG, "Engine build failed: ${e.message}")
            null
        }
    }

    override fun isReady() = ready

    fun resetForNewImage() {
        Log.d(TAG, "New image — resetting session")
        currentSession?.close()
        currentSession = null
    }

    override suspend fun chat(
        image: Bitmap,
        question: String,
        systemPrompt: String?,
        conversationHistory: List<Pair<String, String>>,
        onTokenGenerated: ((String) -> Unit)?
    ): String = withContext(Dispatchers.IO) {
        if (!ready || engine == null) return@withContext "VisionQA is not ready. Please wait."

        return@withContext try {
            currentSession?.close()

            val sessionOptions = LlmInferenceSessionOptions.builder()
                .setTemperature(0.1f)
                .setTopK(40)
                .setGraphOptions(
                    GraphOptions.builder()
                        .setEnableVisionModality(true)
                        .build()
                )
                .build()

            val session = LlmInferenceSession.createFromOptions(engine!!, sessionOptions)
            currentSession = session

            val safeImage = capImageSize(image, MAX_IMAGE_DIM)
            session.addImage(BitmapImageBuilder(safeImage).build())

            generateStreaming(session, question.trim(), onTokenGenerated)

        } catch (e: Exception) {
            Log.e(TAG, "chat() error: ${e.message}", e)
            resetForNewImage()
            "Something went wrong. Please try again."
        }
    }

    private suspend fun generateStreaming(
        session: LlmInferenceSession,
        prompt: String,
        onTokenGenerated: ((String) -> Unit)?
    ): String = suspendCancellableCoroutine { cont ->
        val sb = StringBuilder()
        var completed = false

        try {
            session.addQueryChunk(prompt)

            session.generateResponseAsync { partialResult: String, isDone: Boolean ->
                if (partialResult.isNotEmpty()) {
                    sb.append(partialResult)
                    onTokenGenerated?.invoke(partialResult)
                }
                if (isDone && !completed) {
                    completed = true
                    if (cont.isActive) cont.resume(sb.toString().trim())
                }
            }
        } catch (e: Exception) {
            if (cont.isActive) cont.resumeWithException(e)
        }

        cont.invokeOnCancellation {
            try { session.close() } catch (_: Exception) {}
            currentSession = null
        }
    }

    private fun capImageSize(src: Bitmap, maxDim: Int): Bitmap {
        if (src.width <= maxDim && src.height <= maxDim) return src
        val scale = maxDim.toFloat() / maxOf(src.width, src.height)
        return Bitmap.createScaledBitmap(
            src,
            (src.width * scale).toInt(),
            (src.height * scale).toInt(),
            true
        )
    }

    override fun getModelInfo() = ModelInfo(
        name    = "VisionQA",
        size    = "Multimodal",
        speed   = "On-device",
        quality = "High Quality"
    )

    override fun cleanup() {
        currentSession?.close()
        currentSession = null
        engine?.close()
        engine = null
        ready = false
    }
}