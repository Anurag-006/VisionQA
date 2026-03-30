package com.anurag.visionqa.ai

/**
 * Answers questions directly from structured OCR output.
 *
 * Uses spatial line data from OcrHelper (bounding boxes) instead of regex,
 * so it works on any two-column layout — menus, price lists, signs, forms —
 * without hardcoded format assumptions.
 *
 * Routing logic:
 *   - Text/reading questions   → return raw OCR text
 *   - Price/list questions     → parse structured lines into item+price pairs
 *   - Comparison questions     → compute cheapest/most expensive from pairs
 *   - General questions        → return null (Moondream handles it)
 */
object OcrQueryHandler {

    private val PRICE_REGEX = Regex(
        """Rs\.?\s*\d[\d,]*\s*/?\-*|₹\s*\d[\d,]*|\$\s*\d[\d,.]*""",
        RegexOption.IGNORE_CASE
    )

    private val PRICE_VALUE_REGEX = Regex("""[\d,]+""")

    fun tryAnswer(question: String, ocrResult: OcrHelper.OcrResult?): String? {
        if (ocrResult == null) return null
        val q = question.trim().lowercase()
        android.util.Log.d("OcrQueryHandler",
            "tryAnswer: q='$q' | lines=${ocrResult.structuredLines.size} | " +
                    "pairs=${extractPairs(ocrResult.structuredLines).size}")

        return when {
            isReadQuestion(q)           -> formatReading(ocrResult.rawText)
            isCheapestQuestion(q)       -> findCheapest(ocrResult.structuredLines)
            isMostExpensive(q)          -> findMostExpensive(ocrResult.structuredLines)
            isSpecificItemPrice(q)      -> findSpecificPrice(ocrResult.structuredLines, ocrResult.rawText, q)
            isPriceQuestion(q)          -> formatPrices(ocrResult.structuredLines, ocrResult.rawText)
            isListQuestion(q)           -> formatList(ocrResult.structuredLines, ocrResult.rawText)
            isCountQuestion(q)          -> formatCount(ocrResult.structuredLines)
            isAvailabilityQuestion(q)   -> checkAvailability(ocrResult.structuredLines, q)
            else                        -> null
        }
    }

    // -----------------------------------------------------------------------
    // Intent detection
    // -----------------------------------------------------------------------

    private fun isReadQuestion(q: String) =
        (q.contains("read") || q.contains("written") || q.contains("words") ||
                (q.contains("say") && !q.contains("dish") && !q.contains("item") && !q.contains("menu")) ||
                (q.contains("text") && !q.contains("price") && !q.contains("cost"))) &&
                !isPriceQuestion(q) && !isListQuestion(q)

    private fun isPriceQuestion(q: String) =
        q.contains("price") || q.contains("cost") || q.contains("rate") ||
                q.contains("how much") || q.contains("rupee") || q.contains("rs") ||
                q.contains("₹") || q.contains("charge") ||
                (q.contains("each") && q.contains("biryani")) ||
                (q.contains("prices") && q.contains("of"))

    private fun isListQuestion(q: String) =
        q.contains("list") || q.contains("all") || q.contains("available") ||
                q.contains("what are") || q.contains("items") || q.contains("options") ||
                q.contains("show") || q.contains("menu") || q.contains("dishes") ||
                q.contains("give") || q.contains("tell me") ||
                (q.contains("what") && q.contains("see"))
    private fun isCheapestQuestion(q: String) =
        q.contains("cheap") || q.contains("least expensive") ||
                q.contains("lowest price") || q.contains("minimum") ||
                q.contains("budget") || q.contains("affordable") ||
                (q.contains("most") && q.contains("affordable"))

    private fun isMostExpensive(q: String) =
        q.contains("expensive") || q.contains("costliest") ||
                q.contains("highest price") || q.contains("maximum") ||
                q.contains("most expensive") || q.contains("priciest")

    private fun isCountQuestion(q: String) =
        (q.contains("how many") || q.contains("count") || q.contains("number of")) &&
                (q.contains("item") || q.contains("dish") || q.contains("option") ||
                        q.contains("biryani") || q.contains("thing"))

    private fun isAvailabilityQuestion(q: String) =
        q.contains("is there") || q.contains("do they have") || q.contains("do you have") ||
                q.contains("available") && q.contains("?")

    private fun isSpecificItemPrice(q: String) =
        (q.contains("price of") || q.contains("cost of") || q.contains("how much is") ||
                q.contains("how much does") || q.contains("how much for")) &&
                q.length > 20  // Has an item name after the keyword
    // -----------------------------------------------------------------------
    // Formatters
    // -----------------------------------------------------------------------

    private fun formatReading(raw: String): String {
        // Only show item names, not the raw OCR dump with prices mixed in
        val lines = raw.lines().map { it.trim() }.filter {
            it.isNotBlank() && !PRICE_REGEX.containsMatchIn(it) && !isNoise(it)
        }
        return if (lines.isEmpty()) raw.trim()
        else "The image shows:\n" + lines.joinToString("\n") { "• $it" }
    }

    private fun formatPrices(lines: List<OcrHelper.StructuredLine>, raw: String): String {
        val pairs = extractPairs(lines)
        if (pairs.isEmpty()) return "I could not find price information in the image."
        // Group items by price so output is compact, not a wall of text
        val grouped = pairs.groupBy { it.second }
        return buildString {
            appendLine("Here are the items and their prices:")
            grouped.entries.sortedBy { parsePriceValue(it.key) }.forEach { (price, items) ->
                appendLine("\n$price:")
                items.forEach { appendLine("  • ${it.first}") }
            }
        }.trim()
    }

    private fun formatList(lines: List<OcrHelper.StructuredLine>, raw: String): String {
        val pairs = extractPairs(lines)
        if (pairs.isNotEmpty()) {
            val grouped = pairs.groupBy { it.second }
            return buildString {
                appendLine("Items available:")
                grouped.entries.sortedBy { parsePriceValue(it.key) }.forEach { (price, items) ->
                    appendLine("\n$price:")
                    items.forEach { appendLine("  • ${it.first}") }
                }
            }.trim()
        }
        val items = extractItemsOnly(lines)
        if (items.isEmpty()) return formatReading(raw)
        return "Items available:\n" + items.joinToString("\n") { "• $it" }
    }

    private fun findCheapest(lines: List<OcrHelper.StructuredLine>): String {
        val pairs = extractPairs(lines)
        if (pairs.isEmpty()) return "Could not find price information in the image."
        val minPrice = pairs.minOf { parsePriceValue(it.second) }
        val cheapest = pairs.filter { parsePriceValue(it.second) == minPrice }
        return if (cheapest.size == 1) {
            "The cheapest dish is ${cheapest[0].first} at ${cheapest[0].second}."
        } else {
            "The cheapest options are all at ₹$minPrice/-:\n" +
                    cheapest.joinToString("\n") { "  • ${it.first}" }
        }
    }

    private fun findMostExpensive(lines: List<OcrHelper.StructuredLine>): String {
        val pairs = extractPairs(lines)
        if (pairs.isEmpty()) return "Could not find price information in the image."
        val maxPrice = pairs.maxOf { parsePriceValue(it.second) }
        val priciest = pairs.filter { parsePriceValue(it.second) == maxPrice }
        return if (priciest.size == 1) {
            "The most expensive dish is ${priciest[0].first} at ${priciest[0].second}."
        } else {
            "The most expensive options are all at ₹$maxPrice/-:\n" +
                    priciest.joinToString("\n") { "  • ${it.first}" }
        }
    }

    private fun formatCount(lines: List<OcrHelper.StructuredLine>): String {
        val pairs = extractPairs(lines)
        val items = if (pairs.isNotEmpty()) pairs.map { it.first } else extractItemsOnly(lines)
        return "There are ${items.size} items visible in the image."
    }

    private fun findSpecificPrice(
        lines: List<OcrHelper.StructuredLine>,
        raw: String,
        query: String
    ): String {
        val pairs = extractPairs(lines)
        if (pairs.isEmpty()) return formatReading(raw)

        // Extract the item name from the query — everything after the keyword
        val keywords = listOf("price of", "cost of", "how much is", "how much does", "how much for")
        var itemQuery = query
        for (kw in keywords) {
            val idx = query.indexOf(kw)
            if (idx >= 0) { itemQuery = query.substring(idx + kw.length).trim().trimEnd('?', '.'); break }
        }

        // Fuzzy match against known items
        val best = pairs.minByOrNull { levenshtein(itemQuery, it.first.lowercase()) }
        return if (best != null && levenshtein(itemQuery, best.first.lowercase()) < itemQuery.length / 2 + 2)
            "${best.first} costs ${best.second}."
        else
            "I couldn't find \"$itemQuery\" in the menu. Available items:\n" +
                    pairs.joinToString("\n") { "• ${it.first} — ${it.second}" }
    }

    private fun checkAvailability(lines: List<OcrHelper.StructuredLine>, query: String): String {
        val pairs = extractPairs(lines)
        val items = if (pairs.isNotEmpty()) pairs.map { it.first } else extractItemsOnly(lines)
        // Heuristic: last significant word cluster in the query is the item being asked about
        val stopWords = setOf("is", "there", "do", "they", "have", "you", "any", "a", "an", "the")
        val itemQuery = query.split(" ")
            .filter { it.length > 2 && it !in stopWords }
            .joinToString(" ")

        val match = items.firstOrNull {
            levenshtein(itemQuery.lowercase(), it.lowercase()) < itemQuery.length / 2 + 2
        }
        return if (match != null) {
            val price = pairs.find { it.first == match }?.second
            if (price != null) "Yes, $match is available at $price."
            else "Yes, $match is available."
        } else {
            "I couldn't find that item on the menu."
        }
    }


    // -----------------------------------------------------------------------
    // Core structured line parser
    // -----------------------------------------------------------------------

    /**
     * Extracts (itemName, price) pairs from spatially structured lines.
     *
     * Because OcrHelper already split each line into left/right columns
     * using bounding box midpoint, this is now trivial:
     *   - leftText = item name (e.g. "Mutton Biryani")
     *   - rightText = price (e.g. "Rs. 250/-")
     *
     * We also handle the layout where items and prices are on separate lines
     * (rightText is blank, but the next line's rightText has the price).
     */
    private fun extractPairs(lines: List<OcrHelper.StructuredLine>): List<Pair<String, String>> {

        // ── Strategy 1: Inline pairs (item + price on the same row) ──────────
        val inlinePairs = mutableListOf<Pair<String, String>>()
        var pendingItem: String? = null

        for (line in lines) {
            val left  = line.leftText.trim()
            val right = line.rightText.trim()
            val leftHasPrice  = PRICE_REGEX.containsMatchIn(left)
            val rightHasPrice = PRICE_REGEX.containsMatchIn(right)

            when {
                left.isNotBlank() && rightHasPrice && !leftHasPrice && !isNoise(left) -> {
                    inlinePairs.add(Pair(cleanItemName(left), cleanPrice(right)))
                    pendingItem = null
                }
                leftHasPrice && rightHasPrice -> { pendingItem = null }
                left.isNotBlank() && right.isBlank() && !leftHasPrice && !isNoise(left) -> {
                    pendingItem = cleanItemName(left)
                }
                left.isBlank() && rightHasPrice && pendingItem != null -> {
                    inlinePairs.add(Pair(pendingItem!!, cleanPrice(right)))
                    pendingItem = null
                }
                leftHasPrice && right.isBlank() && pendingItem != null -> {
                    inlinePairs.add(Pair(pendingItem!!, cleanPrice(left)))
                    pendingItem = null
                }
                else -> { if (isHeader(left) || isNoise(left)) pendingItem = null }
            }
        }
        if (inlinePairs.isNotEmpty()) return inlinePairs

        // ── Strategy 2: Two-block zipper ─────────────────────────────────────
        // ML Kit reads the name column and price column as separate blocks.
        // Headers (CHICKEN, MUTTON) must be EXCLUDED from item list but their
        // row-count must still align with the price list, so we count carefully.

        // Separate all lines into: items (orderable food names) vs prices
        val itemLines = mutableListOf<String>()
        val priceLines = mutableListOf<String>()

        for (line in lines) {
            val text = (line.leftText + line.rightText).trim()
            when {
                text.isBlank() -> { /* skip */ }
                PRICE_REGEX.containsMatchIn(text) -> {
                    // Extract ALL price tokens from this line (a line can have "210/- 170/-")
                    PRICE_REGEX.findAll(text).forEach { priceLines.add(cleanPrice(it.value)) }
                }
                isNoise(text) -> { /* skip */ }
                isHeader(text) -> { /* skip headers — don't add to items */ }
                else -> itemLines.add(cleanItemName(text))
            }
        }

        android.util.Log.d("OcrQueryHandler",
            "Zipper: ${itemLines.size} items, ${priceLines.size} prices")

        if (itemLines.isNotEmpty() && priceLines.isNotEmpty()) {
            // Zip by index up to the shorter list
            return itemLines.zip(priceLines)
        }

        return emptyList()
    }

    private fun cleanPrice(raw: String): String {
        // Normalize OCR noise: "21@/-" → "210/-", "2101-" → "210/-"
        val cleaned = raw.replace("@", "0").replace(Regex("(\\d+)1-"), "$10/-")
        val num = Regex("""[\d,]+""").find(cleaned)?.value?.replace(",", "") ?: return raw.trim()
        return "₹$num/-"
    }

    private fun extractItemsOnly(lines: List<OcrHelper.StructuredLine>): List<String> {
        return lines
            .map { it.leftText.trim() }
            .filter { it.isNotBlank() && !isHeader(it) && !isNoise(it) && !PRICE_REGEX.containsMatchIn(it) }
            .map { cleanItemName(it) }
            .filter { it.length > 2 }
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    private fun isHeader(text: String): Boolean {
        // Remove punctuation/brackets for comparison
        val stripped = text.replace(Regex("[^A-Za-z0-9 ]"), "").trim()
        if (stripped.isEmpty()) return true
        // It's a header if: mostly uppercase AND no price AND short-ish
        val upperRatio = stripped.count { it.isUpperCase() }.toFloat() / stripped.count { it.isLetter() }.coerceAtLeast(1)
        return upperRatio >= 0.6f && !PRICE_REGEX.containsMatchIn(text) && stripped.length < 40
    }

    private fun isNoise(text: String): Boolean {
        if (text.length <= 2) return true
        if (text.contains(".com") || text.contains("http") || text.contains("www.")) return true
        // Do NOT flag price-like strings as noise — "210/-" is valid data
        if (PRICE_REGEX.containsMatchIn(text)) return false
        if (text.all { it.isDigit() || it in ".,/ -" }) return true
        return false
    }

    private fun cleanItemName(text: String): String =
        text.trim()
            .trimStart('.', ',', ':', '-', '[', '(', ' ')
            .trimEnd('.', ',', ':', '-', ']', ')', ' ')
            .replace(Regex("^[^A-Za-z]+"), "")  // strip any leading non-letter garbage
            .trim()

    /**
     * Extracts numeric value from a price string for comparison.
     * "Rs. 250/-" → 250, "₹1,200" → 1200
     */
    private fun parsePriceValue(priceStr: String): Int {
        val digits = PRICE_VALUE_REGEX.findAll(priceStr)
            .joinToString("") { it.value.replace(",", "") }
        return digits.toIntOrNull() ?: 0
    }

    private fun levenshtein(a: String, b: String): Int {
        val dp = Array(a.length + 1) { IntArray(b.length + 1) }
        for (i in 0..a.length) dp[i][0] = i
        for (j in 0..b.length) dp[0][j] = j
        for (i in 1..a.length) for (j in 1..b.length) {
            dp[i][j] = if (a[i - 1] == b[j - 1]) dp[i - 1][j - 1]
            else 1 + minOf(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        }
        return dp[a.length][b.length]
    }
}