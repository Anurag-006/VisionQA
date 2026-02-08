package com.anurag.visionqa.camera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.core.content.ContextCompat
import java.io.File

class CameraController(
    private val imageCapture: ImageCapture
) {
    fun capture(
        context: Context,
        onResult: (Bitmap) -> Unit,
        onError: (String) -> Unit
    ) {
        val photoFile = File(
            context.cacheDir,
            "capture_${System.currentTimeMillis()}.jpg"
        )

        val outputOptions = ImageCapture.OutputFileOptions
            .Builder(photoFile)
            .build()

        imageCapture.takePicture(
            outputOptions,
            ContextCompat.getMainExecutor(context),
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(
                    outputFileResults: ImageCapture.OutputFileResults
                ) {
                    val bitmap = BitmapFactory.decodeFile(photoFile.absolutePath)
                    onResult(bitmap)
                }

                override fun onError(exception: ImageCaptureException) {
                    onError(exception.message ?: "Capture failed")
                }
            }
        )
    }
}
