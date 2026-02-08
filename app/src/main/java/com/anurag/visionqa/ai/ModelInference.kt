package com.anurag.visionqa.ai

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class ModelInference(context: Context) {

    private val interpreter: Interpreter
    private val inputSize = 224

    init {
        val modelBuffer = loadModelFile(context)
        interpreter = Interpreter(modelBuffer)
    }

    private fun loadModelFile(context: Context): MappedByteBuffer {
        val afd = context.assets.openFd("model.tflite")
        val inputStream = FileInputStream(afd.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            afd.startOffset,
            afd.declaredLength
        )
    }

    fun run(bitmap: Bitmap): Int {
        val resizedBitmap =
            Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)

        val inputBuffer =
            ByteBuffer.allocateDirect(inputSize * inputSize * 3)
        inputBuffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(inputSize * inputSize)
        resizedBitmap.getPixels(
            pixels,
            0,
            inputSize,
            0,
            0,
            inputSize,
            inputSize
        )

        for (pixel in pixels) {
            inputBuffer.put(((pixel shr 16) and 0xFF).toByte()) // R
            inputBuffer.put(((pixel shr 8) and 0xFF).toByte())  // G
            inputBuffer.put((pixel and 0xFF).toByte())          // B
        }

        inputBuffer.rewind()

        // MobileNet v1 outputs 1001 classes
        val output = Array(1) { ByteArray(1001) }

        interpreter.run(inputBuffer, output)

        var maxIndex = 0
        var maxValue = output[0][0].toInt() and 0xFF

        for (i in output[0].indices) {
            val value = output[0][i].toInt() and 0xFF
            if (value > maxValue) {
                maxValue = value
                maxIndex = i
            }
        }

        return maxIndex
    }
}
