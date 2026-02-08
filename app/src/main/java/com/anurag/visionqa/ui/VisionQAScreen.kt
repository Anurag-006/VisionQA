package com.anurag.visionqa.ui

import android.graphics.Bitmap
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.anurag.visionqa.ai.ModelInference
import android.graphics.BitmapFactory
import androidx.camera.core.ImageProxy
import java.nio.ByteBuffer
import kotlinx.coroutines.*

fun imageProxyToBitmap(image: ImageProxy): Bitmap {
    val buffer: ByteBuffer = image.planes[0].buffer
    val bytes = ByteArray(buffer.remaining())
    buffer.get(bytes)
    return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
}

@Composable
fun VisionQAScreen() {
    val context = LocalContext.current
    val model = remember { ModelInference(context) }

    var imageCapture by remember { mutableStateOf<ImageCapture?>(null) }
    var capturedBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var questionText by remember { mutableStateOf("") }
    var resultText by remember { mutableStateOf("Point camera and capture an image") }
    var isProcessing by remember { mutableStateOf(false) }

    Column(modifier = Modifier.fillMaxSize()) {

        // Camera Preview
        CameraPreview(
            modifier = Modifier.weight(0.5f),
            onImageCaptureReady = {
                imageCapture = it
            }
        )

        // Question Input
        OutlinedTextField(
            value = questionText,
            onValueChange = { questionText = it },
            label = { Text("Ask a question about the image") },
            placeholder = { Text("e.g., What objects do you see?") },
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            enabled = !isProcessing
        )

        // Result Display
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp)
        ) {
            Text(
                text = resultText,
                modifier = Modifier.padding(16.dp),
                style = MaterialTheme.typography.bodyMedium
            )
        }

        Spacer(modifier = Modifier.height(8.dp))

        // Capture Button
        Button(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp),
            onClick = {
                val capture = imageCapture ?: return@Button

                isProcessing = true
                resultText = "Capturing image..."

                capture.takePicture(
                    ContextCompat.getMainExecutor(context),
                    object : ImageCapture.OnImageCapturedCallback() {

                        override fun onCaptureSuccess(image: ImageProxy) {
                            val bitmap = imageProxyToBitmap(image)
                            image.close()

                            capturedBitmap = bitmap
                            resultText = "Image captured! Click 'Analyze' to process."
                            isProcessing = false
                        }

                        override fun onError(exception: ImageCaptureException) {
                            resultText = "Capture failed: ${exception.message}"
                            isProcessing = false
                        }
                    }
                )
            },
            enabled = !isProcessing
        ) {
            Text("Capture Image")
        }

        Spacer(modifier = Modifier.height(8.dp))

        // Analyze Button
        Button(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp)
                .padding(bottom = 16.dp),
            onClick = {
                val bitmap = capturedBitmap
                val question = questionText

                if (bitmap == null) {
                    resultText = "Please capture an image first"
                    return@Button
                }

                if (question.isBlank()) {
                    resultText = "Please enter a question"
                    return@Button
                }

                isProcessing = true
                resultText = "Analyzing..."

                // Run in background thread
                // Run in coroutine
                kotlinx.coroutines.GlobalScope.launch(kotlinx.coroutines.Dispatchers.IO) {
                    try {
                        val answer = model.analyzeWithQuestion(bitmap, question)

                        // Update UI on main thread
                        kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.Main) {
                            resultText = answer
                            isProcessing = false
                        }
                    } catch (e: Exception) {
                        kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.Main) {
                            resultText = "Error: ${e.message}"
                            isProcessing = false
                        }
                    }
                }
            },
            enabled = !isProcessing && capturedBitmap != null
        ) {
            Text("Analyze Image + Question")
        }
    }
}