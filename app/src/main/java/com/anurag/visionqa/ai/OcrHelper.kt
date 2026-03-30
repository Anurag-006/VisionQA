package com.anurag.visionqa.ai

import android.graphics.Bitmap
import android.util.Log
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import kotlin.coroutines.resume
import kotlin.coroutines.suspendCoroutine
import kotlin.math.abs

/**
 * Wraps MLKit text recognition and returns structured output.
 *
 * Instead of discarding MLKit's bounding box data (which we were doing by
 * just calling result.text), we use the spatial coordinates to reconstruct
 * the two-column layout of menus, price lists, signs, etc.
 *
 * This means "Mutton Biryani" on the left at y=120 is correctly paired with
 * "Rs. 250/-" on the right at y=122, regardless of how the raw text string
 * is ordered.
 */
class OcrHelper {

    private val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)

    companion object {
        private const val TAG = "OcrHelper"
        // Used in buildStructuredLines to detect price-column elements
        private val PRICE_PATTERN = Regex("""\d{2,4}\s*/?-?""")
    }

    data class OcrResult(
        /** Raw text exactly as MLKit returns it — used for general questions */
        val rawText: String,
        /** Spatially reconstructed lines — used for structured queries */
        val structuredLines: List<StructuredLine>
    )

    /**
     * A single logical line of text reconstructed from spatial position.
     * For a two-column layout like a menu, one line may have a left and right part.
     */
    data class StructuredLine(
        val leftText: String,           // e.g. "Mutton Biryani"
        val rightText: String = "",     // e.g. "Rs. 250/-"  (empty if single-column)
        val y: Int = 0                  // vertical position for ordering
    ) {
        /** Returns the full line as a single string */
        fun fullText() = if (rightText.isBlank()) leftText
        else "$leftText  $rightText"
    }

    suspend fun extractText(bitmap: Bitmap): OcrResult? = suspendCoroutine { cont ->
        val image = InputImage.fromBitmap(bitmap, 0)
        recognizer.process(image)
            .addOnSuccessListener { result ->
                val raw = result.text.trim()
                if (raw.isBlank()) {
                    Log.d(TAG, "No text detected.")
                    cont.resume(null)
                    return@addOnSuccessListener
                }

                Log.i(TAG, "=== OCR RAW (${raw.length} chars) ===\n$raw\n=== END ===")

                // Build structured lines from bounding box data
                val structured = buildStructuredLines(result)
                Log.i(TAG, "=== STRUCTURED (${structured.size} lines) ===")
                structured.forEach { Log.i(TAG, "  L:'${it.leftText}' R:'${it.rightText}'") }

                cont.resume(OcrResult(raw, structured))
            }
            .addOnFailureListener { e ->
                Log.e(TAG, "OCR failed: ${e.message}")
                cont.resume(null)
            }
    }

    /**
     * Reconstructs spatial layout from MLKit's bounding boxes.
     *
     * Algorithm:
     * 1. Collect all text elements with their centre-y and centre-x positions
     * 2. Find the image midpoint x — elements left of it are "left column",
     *    elements right of it are "right column"
     * 3. Group elements by y-position (within a threshold) into logical rows
     * 4. Each row produces one StructuredLine with left and right parts
     *
     * This correctly handles any two-column layout (menus, price lists, forms)
     * without any regex or hardcoded format assumptions.
     */
    private fun buildStructuredLines(
        result: com.google.mlkit.vision.text.Text
    ): List<StructuredLine> {

        data class Element(val text: String, val left: Int, val right: Int, val cy: Int)

        // Collect all line-level elements with their bounding boxes
        val elements = mutableListOf<Element>()
        for (block in result.textBlocks) {
            for (line in block.lines) {
                val box = line.boundingBox ?: continue
                val cy = (box.top + box.bottom) / 2
                elements.add(Element(line.text.trim(), box.left, box.right, cy))
            }
        }
        if (elements.isEmpty()) return emptyList()

        // Dynamic Y threshold: use average line height so it works on any image scale
        val avgLineHeight = elements
            .mapNotNull { result.textBlocks.flatMap { b -> b.lines }
                .find { l -> l.text.trim() == it.text }?.boundingBox }
            .map { it.height() }
            .average()
            .let { if (it.isNaN()) 30.0 else it }
        val Y_THRESHOLD = (avgLineHeight * 0.6).toInt().coerceIn(15, 60)

        // Sort by vertical position
        elements.sortBy { it.cy }

        // Group into rows by proximity
        val rows = mutableListOf<MutableList<Element>>()
        for (el in elements) {
            val lastRow = rows.lastOrNull()
            if (lastRow != null && abs(el.cy - lastRow.map { it.cy }.average()) <= Y_THRESHOLD) {
                lastRow.add(el)
            } else {
                rows.add(mutableListOf(el))
            }
        }

        // Find the true text split point: gap between rightmost left-column
        // element and leftmost right-column element across ALL rows.
        // This is far more accurate than maxX/2.
        val allRights = rows.filter { it.size >= 2 }.flatMap { row ->
            val sorted = row.sortedBy { it.left }
            // The gap between consecutive elements in a multi-element row
            sorted.zipWithNext().map { (a, b) -> Pair(a.right, b.left) }
        }
        // midX = average of gap midpoints in rows that have 2+ elements
        val midX = if (allRights.isNotEmpty()) {
            allRights.map { (r, l) -> (r + l) / 2 }.average().toInt()
        } else {
            // Fallback: use the left edge of elements that look like prices
            val priceLeftEdges = elements
                .filter { PRICE_PATTERN.containsMatchIn(it.text) }
                .map { it.left }
            if (priceLeftEdges.isNotEmpty()) priceLeftEdges.min()
            else elements.maxOf { it.right } / 2
        }

        Log.d(TAG, "Dynamic midX=$midX, Y_THRESHOLD=$Y_THRESHOLD")

        // Convert each row to a StructuredLine
        return rows.map { row ->
            val sorted = row.sortedBy { it.left }
            val leftParts  = sorted.filter { (it.left + it.right) / 2 <= midX }
                .joinToString(" ") { it.text }
            val rightParts = sorted.filter { (it.left + it.right) / 2 > midX }
                .joinToString(" ") { it.text }
            StructuredLine(leftParts, rightParts, row.map { it.cy }.average().toInt())
        }.filter { it.leftText.isNotBlank() || it.rightText.isNotBlank() }
    }

    fun close() { recognizer.close() }
}