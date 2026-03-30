package com.anurag.visionqa.ai

import android.content.Context
import android.util.Log
import org.json.JSONObject
import java.io.File

class RealTokenizer(private val context: Context, tokenizerPath: String? = null) {

    private val vocab = mutableMapOf<String, Int>()
    private val reverseVocab = mutableMapOf<Int, String>()
    private val merges = mutableListOf<Pair<String, String>>()
    private val mergePriority = mutableMapOf<String, Int>()

    private val byteEncoder: Map<Int, Char> = buildByteEncoder()
    private val byteDecoder: Map<Char, Int> = byteEncoder.entries.associate { (k, v) -> v to k }

    private var eosTokenId = 50256
    private val unkTokenId = 0

    companion object {
        private const val TAG = "RealTokenizer"

        fun buildByteEncoder(): Map<Int, Char> {
            val bs = mutableListOf<Int>()
            for (b in 33..126) bs.add(b)
            for (b in 161..172) bs.add(b)
            for (b in 174..255) bs.add(b)
            val cs = bs.map { it.toChar() }.toMutableList()
            var n = 0
            for (b in 0..255) {
                if (b !in bs) { bs.add(b); cs.add((256 + n).toChar()); n++ }
            }
            return bs.zip(cs).toMap()
        }
    }

    init {
        Log.d(TAG, "=== TOKENIZER INIT ===")
        loadTokenizer(tokenizerPath)
        if (vocab.isNotEmpty() && merges.isNotEmpty()) {
            Log.d(TAG, "✅ Tokenizer ready. ${selfTest()}")
        } else {
            Log.e(TAG, "❌ Tokenizer failed to load!")
        }
    }

    private fun loadTokenizer(path: String?) {
        if (path != null) {
            val f = File(path)
            if (f.exists() && f.length() > 100) {
                try {
                    parseTokenizerJson(f.readText())
                    if (vocab.isNotEmpty() && merges.isNotEmpty()) return
                } catch (e: Exception) { Log.e(TAG, "File parse failed: ${e.message}") }
            } else { Log.w(TAG, "File missing/small: $path") }
        }
        try {
            val json = context.assets.open("tokenizer.json").bufferedReader().use { it.readText() }
            parseTokenizerJson(json)
            if (vocab.isNotEmpty() && merges.isNotEmpty()) return
        } catch (e: Exception) { Log.w(TAG, "Assets not found: ${e.message}") }
        Log.e(TAG, "❌ Could not load tokenizer — using char fallback.")
        buildCharFallback()
    }

    private fun parseTokenizerJson(json: String) {
        try {
            val root  = JSONObject(json)
            val model = root.optJSONObject("model")
                ?: throw IllegalArgumentException("No 'model' key")

            model.optJSONObject("vocab")?.let { vocabJson ->
                vocabJson.keys().forEach { token ->
                    val id = vocabJson.getInt(token)
                    vocab[token] = id; reverseVocab[id] = token
                }
            } ?: throw IllegalArgumentException("No 'vocab'")

            vocab["<|endoftext|>"]?.let { eosTokenId = it }

            model.optJSONArray("merges")?.let { arr ->
                for (i in 0 until arr.length()) {
                    val line = arr.getString(i)
                    val idx  = line.indexOf(' ')
                    if (idx < 0) continue
                    val left  = line.substring(0, idx)
                    val right = line.substring(idx + 1)
                    merges.add(Pair(left, right))
                    mergePriority["$left $right"] = i
                }
            } ?: throw IllegalArgumentException("No 'merges'")

            root.optJSONArray("added_tokens")?.let { arr ->
                for (i in 0 until arr.length()) {
                    val obj     = arr.getJSONObject(i)
                    val content = obj.getString("content")
                    val id      = obj.getInt("id")
                    vocab[content] = id; reverseVocab[id] = content
                }
            }

            Log.d(TAG, "Parsed: ${vocab.size} tokens, ${merges.size} merges, EOS=$eosTokenId")
        } catch (e: Exception) {
            Log.e(TAG, "parseTokenizerJson: ${e.message}")
            vocab.clear(); reverseVocab.clear(); merges.clear(); mergePriority.clear()
        }
    }

    private fun buildCharFallback() {
        eosTokenId = 50256
        vocab["<|endoftext|>"] = 50256; reverseVocab[50256] = "<|endoftext|>"
        for (i in 32..126) {
            val ch = i.toChar().toString()
            vocab[ch] = i; reverseVocab[i] = ch
        }
    }

    // -----------------------------------------------------------------------
    // Encoding
    // -----------------------------------------------------------------------

    fun encode(text: String): IntArray {
        if (merges.isEmpty() || vocab.isEmpty()) {
            Log.w(TAG, "encode() called but tokenizer not loaded!")
            return IntArray(0)
        }
        return encodeBPE(text).toIntArray()
    }

    private fun encodeBPE(text: String): List<Int> {
        val tokenIds = mutableListOf<Int>()
        for (chunk in splitChunks(text)) {
            if (chunk.isEmpty()) continue
            val symbols = chunk.toByteArray(Charsets.UTF_8)
                .map { b -> byteEncoder[b.toInt() and 0xFF]!!.toString() }
                .toMutableList()
            applyMerges(symbols)
            for (sym in symbols) {
                val id = vocab[sym]
                if (id != null) tokenIds.add(id)
                else for (ch in sym) tokenIds.add(vocab[ch.toString()] ?: unkTokenId)
            }
        }
        return tokenIds
    }

    private fun splitChunks(text: String): List<String> {
        val chunks = mutableListOf<String>()
        val sb     = StringBuilder()
        var i      = 0
        while (i < text.length) {
            val ch = text[i]
            when {
                ch == '\n' || ch == '\r' -> {
                    if (sb.isNotEmpty()) { chunks.add(sb.toString()); sb.clear() }
                    chunks.add(ch.toString())
                }
                ch == ' ' -> {
                    if (sb.isNotEmpty()) { chunks.add(sb.toString()); sb.clear() }
                    if (i + 1 < text.length && text[i + 1].isLetterOrDigit()) sb.append(' ')
                    else chunks.add(" ")
                }
                ch.isLetterOrDigit() -> sb.append(ch)
                else -> {
                    if (sb.isNotEmpty()) { chunks.add(sb.toString()); sb.clear() }
                    chunks.add(ch.toString())
                }
            }
            i++
        }
        if (sb.isNotEmpty()) chunks.add(sb.toString())
        return chunks
    }

    private fun applyMerges(symbols: MutableList<String>) {
        while (symbols.size > 1) {
            var bestRank = Int.MAX_VALUE
            var bestIdx  = -1
            for (i in 0 until symbols.size - 1) {
                val rank = mergePriority["${symbols[i]} ${symbols[i + 1]}"]
                if (rank != null && rank < bestRank) { bestRank = rank; bestIdx = i }
            }
            if (bestIdx == -1) break
            val merged = symbols[bestIdx] + symbols[bestIdx + 1]
            symbols[bestIdx] = merged
            symbols.removeAt(bestIdx + 1)
        }
    }

    // -----------------------------------------------------------------------
    // Decoding
    // -----------------------------------------------------------------------

    fun decode(tokens: IntArray): String {
        if (tokens.isEmpty()) return ""
        return try {
            val rawStr = tokens
                .filter { it != eosTokenId }
                .mapNotNull { reverseVocab[it] }
                .joinToString("")
            val bytes = rawStr.mapNotNull { byteDecoder[it]?.toByte() }.toByteArray()
            bytes.toString(Charsets.UTF_8)
        } catch (e: Exception) {
            Log.e(TAG, "decode error: ${e.message}")
            ""
        }
    }

    // -----------------------------------------------------------------------
    // Prompt formatting
    // -----------------------------------------------------------------------

    fun formatPrompt(
        question: String,
        ocrText: String? = null,     // kept for signature compatibility, ignored for visual queries
        systemPrompt: String? = null,
        isFollowUp: Boolean = false
    ): String {
        // Since OCR questions are handled by OcrQueryHandler before reaching here,
        // the VLM only receives pure visual questions ("What color is X?", "Describe the scene").
        // Keeping the prompt minimal (<80 tokens) is critical for 1.8B model attention quality.
        // No few-shot examples — they consumed ~200 tokens and caused context saturation.
        val safeQuestion = question.replace("\"", "'").trim()
        return "\n\nQuestion: $safeQuestion\n\nAnswer:"
    }
    // --- AND HERE ---
    private fun sanitizeOcrText(text: String): String {
        return text
            .replace(Regex("[<>|\\[\\]]"), "")    // remove special-token chars
            .replace(Regex("[ \\t]+"), " ")       // collapse multiple spaces/tabs into a single space
            .replace(Regex("\\n+"), "\n")         // PRESERVE newlines, but collapse multiple empty lines into one
            .trim()
    }

    // -----------------------------------------------------------------------
    // Diagnostics
    // -----------------------------------------------------------------------

    fun getVocabSize(): Int = vocab.size
    fun getEosId(): Int = eosTokenId

    fun selfTest(): String {
        val ids     = encode("Hello world").toList()
        val decoded = decode(encode("Hello world"))
        return "selfTest: encode('Hello world')=$ids decoded='$decoded' (expect [15496, 995])"
    }
}