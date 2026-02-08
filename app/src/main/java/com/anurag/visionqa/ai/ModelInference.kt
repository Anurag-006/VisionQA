package com.anurag.visionqa.ai

import android.content.Context
import android.graphics.Bitmap
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.label.ImageLabeling
import com.google.mlkit.vision.label.defaults.ImageLabelerOptions
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import kotlinx.coroutines.tasks.await

class ModelInference(context: Context) {

    private val imageLabeler = ImageLabeling.getClient(
        ImageLabelerOptions.Builder()
            .setConfidenceThreshold(0.4f)
            .build()
    )

    private val textRecognizer = TextRecognition.getClient(
        TextRecognizerOptions.DEFAULT_OPTIONS
    )

    suspend fun analyzeWithQuestion(bitmap: Bitmap, question: String): String {
        val image = InputImage.fromBitmap(bitmap, 0)

        // 1. Detect objects
        val labels = try {
            imageLabeler.process(image).await()
        } catch (e: Exception) {
            return "❌ Error detecting objects: ${e.message}"
        }

        // 2. Detect text
        val textResult = try {
            textRecognizer.process(image).await()
        } catch (e: Exception) {
            return "❌ Error reading text: ${e.message}"
        }

        // 3. FIX: Pass the 'bitmap' into buildIntelligentAnswer
        return buildIntelligentAnswer(
            question = question,
            objects = labels.map { it.text to it.confidence },
            detectedText = textResult.text,
            bitmap = bitmap
        )
    }

    private fun buildIntelligentAnswer(
        question: String,
        objects: List<Pair<String, Float>>,
        detectedText: String,
        bitmap: Bitmap  // Parameter added here
    ): String {
        val q = question.lowercase().trim()

        val topObjects = objects.take(5)
        val objectsList = topObjects.joinToString(", ") {
            "${it.first} (${(it.second * 100).toInt()}%)"
        }

        val hasNumbers = detectedText.any { it.isDigit() }
        val hasCurrency = detectedText.contains("₹") || detectedText.contains("$") ||
                detectedText.contains("rs", ignoreCase = true)

        return when {
            // COLOR QUESTIONS - Passes bitmap to answerColorQuestion
            q.contains("color") || q.contains("colour") -> {
                answerColorQuestion(q, objects, detectedText, bitmap)
            }

            // PRICE QUESTIONS
            q.contains("price") || q.contains("cost") || q.contains("how much") -> {
                if (hasCurrency || hasNumbers) {
                    "💰 **Answer**: I found price information:\n\n$detectedText"
                } else {
                    "❌ **Answer**: No price information visible in the image.\n\nDetected text: $detectedText"
                }
            }

            // WHAT/IDENTIFY QUESTIONS
            q.contains("what is") || q.contains("what are") || q.contains("identify") -> {
                answerWhatQuestion(q, objects, detectedText)
            }

            // SEE/SHOW QUESTIONS
            q.contains("what do you see") || q.contains("what can you see") -> {
                buildString {
                    append("👁️ **Answer**: Here's what I can see:\n\n")
                    append("🏷️ Objects: $objectsList\n\n")
                    if (detectedText.isNotBlank()) {
                        append("📝 Text: ${detectedText.take(200)}")
                    }
                }
            }

            // READ/TEXT QUESTIONS
            q.contains("read") || q.contains("text") || q.contains("written") || q.contains("says") -> {
                if (detectedText.isNotBlank()) {
                    "📖 **Answer**: Here's the text I found:\n\n$detectedText"
                } else {
                    "❌ **Answer**: No clear text detected in the image."
                }
            }

            // COUNT QUESTIONS
            q.contains("how many") || q.contains("count") -> {
                "🔢 **Answer**: I detected ${objects.size} distinct objects:\n\n$objectsList"
            }

            // YES/NO QUESTIONS
            q.startsWith("is there") || q.startsWith("can you see") || q.contains("is there") -> {
                answerYesNoQuestion(q, objects, detectedText)
            }

            else -> {
                answerGenericQuestion(q, objects, detectedText)
            }
        }
    }

    private fun answerColorQuestion(
        question: String,
        objects: List<Pair<String, Float>>,
        text: String,
        bitmap: Bitmap
    ): String {
        // Calls the ColorDetector object you created
        val dominantColor = ColorDetector.getDominantColor(bitmap)

        val targetObject = when {
            question.contains("screen") -> "screen"
            question.contains("keyboard") -> "keyboard"
            else -> null
        }

        return if (targetObject != null) {
            "🎨 **Answer**: The dominant color of the $targetObject appears to be **$dominantColor**.\n\n(Note: Based on center region analysis)\n\nDetected: ${objects.take(3).joinToString(", ") { it.first }}"
        } else {
            "🎨 **Answer**: The dominant color in the center of the image appears to be **$dominantColor**."
        }
    }

    private fun answerWhatQuestion(
        question: String,
        objects: List<Pair<String, Float>>,
        text: String
    ): String {
        val keywords = listOf("keyboard", "screen", "computer", "desk", "monitor", "mouse", "text")
        val askedAbout = keywords.find { question.contains(it) }

        return if (askedAbout != null) {
            val found = objects.find { it.first.contains(askedAbout, ignoreCase = true) }
            if (found != null) {
                "✅ **Answer**: Yes, I can see a **${found.first}** (${(found.second * 100).toInt()}% confidence)\n\n${if (text.isNotBlank()) "Text visible: ${text.take(100)}" else ""}"
            } else {
                "❓ **Answer**: I don't clearly see a **$askedAbout** in this image.\n\nWhat I do see: ${objects.take(3).joinToString(", ") { it.first }}"
            }
        } else {
            "🔍 **Answer**: The main objects in the image are:\n\n${objects.take(5).joinToString("\n") { "• ${it.first} (${(it.second * 100).toInt()}%)" }}"
        }
    }

    private fun answerYesNoQuestion(
        question: String,
        objects: List<Pair<String, Float>>,
        text: String
    ): String {
        val keywords = listOf("keyboard", "screen", "computer", "desk", "monitor", "mouse", "text", "price", "number")
        val lookingFor = keywords.find { question.contains(it) }

        return if (lookingFor != null) {
            val found = objects.any { it.first.contains(lookingFor, ignoreCase = true) } ||
                    text.contains(lookingFor, ignoreCase = true)

            if (found) {
                "✅ **Answer**: Yes! I can see **$lookingFor** in the image."
            } else {
                "❌ **Answer**: No, I don't clearly see **$lookingFor** in this image."
            }
        } else {
            "🤔 **Answer**: I'm not sure what you're looking for, but I see: ${objects.take(2).joinToString { it.first }}"
        }
    }

    private fun answerGenericQuestion(
        question: String,
        objects: List<Pair<String, Float>>,
        text: String
    ): String {
        return buildString {
            append("💬 **Answer to**: \"$question\"\n\n")
            if (objects.isNotEmpty()) {
                append("🏷️ Objects: ${objects.take(3).joinToString { it.first }}\n")
            }
            if (text.isNotBlank()) {
                append("📝 Text detected: ${text.take(50)}...\n")
            }
            append("\n💡 Try: \"What color is this?\" or \"What do you see?\"")
        }
    }
}