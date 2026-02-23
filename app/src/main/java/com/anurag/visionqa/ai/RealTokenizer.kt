package com.anurag.visionqa.ai

import android.content.Context
import android.util.Log
import org.json.JSONObject
import java.io.File

class RealTokenizer(private val context: Context, tokenizerPath: String? = null) {

    private val vocab = mutableMapOf<String, Int>()
    private val reverseVocab = mutableMapOf<Int, String>()
    private val merges = mutableListOf<Pair<String, String>>()

    private val bosTokenId = 1
    private var eosTokenId = 50256 // Default for Phi/Moondream
    private val padTokenId = 0
    private val unkTokenId = 0

    companion object {
        private const val TAG = "RealTokenizer"
    }

    init {
        Log.d(TAG, "=== TOKENIZER INIT START ===")
        Log.d(TAG, "Provided path: $tokenizerPath")
        loadTokenizer(tokenizerPath)
        Log.d(TAG, "Vocab size: ${vocab.size}")
        Log.d(TAG, "Merges: ${merges.size}")
        Log.d(TAG, "=== TOKENIZER INIT COMPLETE ===")
    }

    private fun loadTokenizer(path: String?) {
        try {
            if (path != null && File(path).exists()) {
                Log.d(TAG, "Loading from file: $path")
                val file = File(path)
                parseTokenizerJson(file.readText())
                return
            }
            try {
                Log.d(TAG, "Trying to load from assets/tokenizer.json")
                val json = context.assets.open("tokenizer.json").bufferedReader().use { it.readText() }
                parseTokenizerJson(json)
                return
            } catch (e: Exception) {
                Log.w(TAG, "Asset not found: ${e.message}")
            }
            buildSimpleVocab()
        } catch (e: Exception) {
            Log.e(TAG, "❌ Tokenizer load failed", e)
            buildSimpleVocab()
        }
    }

    private fun parseTokenizerJson(json: String) {
        val tokenizer = JSONObject(json)

        val model = tokenizer.optJSONObject("model")
        model?.optJSONObject("vocab")?.let { vocabJson ->
            vocabJson.keys().forEach { token ->
                val id = vocabJson.getInt(token)
                vocab[token] = id
                reverseVocab[id] = token
            }
        }

        vocab["<|endoftext|>"]?.let {
            eosTokenId = it
        }

        model?.optJSONArray("merges")?.let { mergesArray ->
            for (i in 0 until mergesArray.length()) {
                val merge = mergesArray.getString(i).split(" ")
                if (merge.size == 2) {
                    merges.add(Pair(merge[0], merge[1]))
                }
            }
        }

        tokenizer.optJSONArray("added_tokens")?.let { addedTokens ->
            for (i in 0 until addedTokens.length()) {
                val token = addedTokens.getJSONObject(i)
                val content = token.getString("content")
                val id = token.getInt("id")
                vocab[content] = id
                reverseVocab[id] = content
            }
        }

        Log.d(TAG, "✅ Loaded ${vocab.size} tokens, EOS is $eosTokenId")
    }

    private fun buildSimpleVocab() {
        vocab["<|endoftext|>"] = eosTokenId
        val commonWords = listOf("the", "a", "an", "is", "image", "picture")
        commonWords.forEachIndexed { index, word ->
            val id = index + 10
            vocab[word] = id
            reverseVocab[id] = word
        }
    }

    fun encode(text: String): IntArray {
        val tokens = mutableListOf<Int>()

        if (merges.isNotEmpty()) {
            tokens.addAll(encodeBPE(text))
        } else {
            tokens.addAll(encodeWords(text))
        }

        return tokens.toIntArray()
    }

    private fun encodeBPE(text: String): List<Int> {
        val tokens = mutableListOf<Int>()

        val mappedText = text.replace(" ", "Ġ").replace("\n", "Ċ")

        val regex = Regex("(?=Ġ)|(?=Ċ)|(?=\\p{Punct})|(?<=\\p{Punct})")
        val chunks = mappedText.split(regex).filter { it.isNotEmpty() }

        for (chunk in chunks) {
            var remaining = chunk
            while (remaining.isNotEmpty()) {
                var found = false
                for (len in remaining.length downTo 1) {
                    val sub = remaining.substring(0, len)
                    if (vocab.containsKey(sub)) {
                        tokens.add(vocab[sub]!!)
                        remaining = remaining.substring(len)
                        found = true
                        break
                    }
                }
                if (!found) {
                    tokens.add(unkTokenId)
                    remaining = remaining.drop(1)
                }
            }
        }
        return tokens
    }

    private fun encodeWords(text: String): List<Int> {
        return text.split(Regex("\\s+|(?=[.,!?])|(?<=[.,!?])"))
            .filter { it.isNotBlank() }
            .map { word -> vocab[word] ?: vocab[word.lowercase()] ?: unkTokenId }
    }

    fun decode(tokens: IntArray): String {
        return tokens
            .filter { it !in setOf(bosTokenId, eosTokenId, padTokenId, unkTokenId) }
            .mapNotNull { reverseVocab[it] }
            .joinToString("")
            .replace("Ġ", " ")
            .replace("Ċ", "\n")
            .replace("▁", " ")
        // REMOVED .trim() from here so streamed tokens keep their spaces!
    }

    // Replace your current formatPrompt with this:
    fun formatPrompt(question: String, systemPrompt: String? = null, isFollowUp: Boolean = false): String {
        val builder = StringBuilder()

        // Only inject the persona on the very first turn!
        if (!isFollowUp && !systemPrompt.isNullOrBlank()) {
            builder.append("Instructions: $systemPrompt\n\n")
        }

        builder.append("Question: $question\n\nAnswer:")

        // The model expects newlines before starting
        return "\n\n$builder"
    }

    fun getVocabSize(): Int = vocab.size
    fun getEosId(): Int = eosTokenId
}