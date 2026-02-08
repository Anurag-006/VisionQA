package com.anurag.visionqa.ai

import android.graphics.Bitmap

fun runFakeAI(bitmap: Bitmap, question: String): String {
    Thread.sleep(2000)
    return "FAKE AI RESULT\nImage: ${bitmap.width}x${bitmap.height}\nQuestion: $question"
}
