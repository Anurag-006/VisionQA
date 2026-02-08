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
    var resultText by remember { mutableStateOf("No image captured yet") }

    Column(modifier = Modifier.fillMaxSize()) {

        CameraPreview(
            modifier = Modifier.weight(1f),
            onImageCaptureReady = {
                imageCapture = it
            }
        )

        Text(
            text = resultText,
            modifier = Modifier.padding(16.dp)
        )

        Button(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            onClick = {
                val capture = imageCapture ?: return@Button

                capture.takePicture(
                    ContextCompat.getMainExecutor(context),
                    object : ImageCapture.OnImageCapturedCallback() {

                        override fun onCaptureSuccess(image: ImageProxy) {
                            val bitmap = imageProxyToBitmap(image)
                            image.close()

                            val classIndex = model.run(bitmap)
                            resultText = "Detected class index: $classIndex"
                        }

                        override fun onError(exception: ImageCaptureException) {
                            resultText = "Capture failed: ${exception.message}"
                        }
                    }
                )
            }
        ) {
            Text("Capture")
        }
    }
}
