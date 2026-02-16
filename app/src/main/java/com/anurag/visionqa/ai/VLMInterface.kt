package com.anurag.visionqa.ai

import android.graphics.Bitmap

interface VLMInterface {
    suspend fun initialize(): Boolean
    fun isReady(): Boolean

    suspend fun chat(
        image: Bitmap,
        question: String,
        conversationHistory: List<Pair<String, String>> = emptyList(),
        onTokenGenerated: ((String) -> Unit)? = null
    ): String

    fun getModelInfo(): ModelInfo
    fun cleanup()
}

data class ModelInfo(
    val name: String,
    val size: String,
    val speed: String,
    val quality: String
)