package com.anurag.visionqa.ai

import android.graphics.Bitmap
import android.graphics.Color

object ColorDetector {

    fun getDominantColor(bitmap: Bitmap): String {
        // Sample pixels from center region
        val centerX = bitmap.width / 2
        val centerY = bitmap.height / 2
        val sampleSize = 50

        var totalR = 0
        var totalG = 0
        var totalB = 0
        var count = 0

        for (x in (centerX - sampleSize)..(centerX + sampleSize)) {
            for (y in (centerY - sampleSize)..(centerY + sampleSize)) {
                if (x in 0 until bitmap.width && y in 0 until bitmap.height) {
                    val pixel = bitmap.getPixel(x, y)
                    totalR += Color.red(pixel)
                    totalG += Color.green(pixel)
                    totalB += Color.blue(pixel)
                    count++
                }
            }
        }

        val avgR = totalR / count
        val avgG = totalG / count
        val avgB = totalB / count

        return getColorName(avgR, avgG, avgB)
    }

    private fun getColorName(r: Int, g: Int, b: Int): String {
        return when {
            r < 50 && g < 50 && b < 50 -> "Black/Dark"
            r > 200 && g > 200 && b > 200 -> "White/Light"
            r > g && r > b -> "Red/Warm"
            g > r && g > b -> "Green"
            b > r && b > g -> "Blue"
            r > 150 && g > 150 -> "Yellow"
            else -> "Mixed/Gray"
        }
    }
}