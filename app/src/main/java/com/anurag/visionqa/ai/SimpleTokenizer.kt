package com.anurag.visionqa.ai

class SimpleTokenizer {

    private val vocab = mutableMapOf<String, Long>()
    private val reverseVocab = mutableMapOf<Long, String>()

    init {
        buildVocab()
    }

    private fun buildVocab() {
        fun reg(word: String, id: Long) {
            vocab[word.lowercase()] = id
            reverseVocab[id] = word
        }

        // --- MANDATORY GPT-2 / MOONDREAM IDs ---
        reg("<|endoftext|>", 50256L)
        reg("<image>", 50257L)
        reg("question", 11601L)
        reg("answer", 18620L)

        // Common Helper Tokens
        reg("\n", 198L)
        reg(":", 25L)
        reg(" ", 220L)
        reg(".", 13L)
        reg("?", 30L)
        reg(",", 11L)
        reg("yes", 9891L)
        reg("no", 3763L)
        reg("Yes", 3363L) // Capitalized
        reg("No", 1400L)
        reg("pen", 11270L)

        // Common vocabulary mapped to real GPT-2 IDs
        val common = mapOf(
            "the" to 464L, "be" to 307L, "to" to 284L, "of" to 259L,
            "and" to 290L, "a" to 64L, "in" to 287L, "that" to 326L,
            "have" to 423L, "i" to 72L, "it" to 314L, "for" to 337L,
            "not" to 407L, "on" to 319L, "with" to 351L, "he" to 339L,
            "as" to 271L, "you" to 345L, "do" to 466L, "at" to 262L,
            "this" to 341L, "but" to 475L, "his" to 465L, "by" to 534L,
            "from" to 422L, "they" to 612L, "we" to 492L, "say" to 910L,
            "her" to 502L, "she" to 703L, "or" to 393L, "an" to 281L,
            "will" to 481L, "my" to 616L, "one" to 530L, "all" to 477L,
            "would" to 561L, "there" to 612L, "their" to 511L, "what" to 644L,
            "is" to 318L, "can" to 460L, "see" to 766L, "describe" to 11051L,
            "image" to 3012L, "picture" to 4377L, "photo" to 4349L,
            "tell" to 980L, "me" to 502L, "color" to 3122L, "red" to 2266L,
            "blue" to 3635L, "green" to 4015L, "black" to 2844L, "white" to 2441L,
            "car" to 1097L, "dog" to 3290L, "cat" to 3797L, "person" to 1048L,
            "man" to 582L, "woman" to 2415L, "tree" to 5509L, "sky" to 5524L
        )

        common.forEach { (word, id) -> reg(word, id) }
    }

    fun encode(text: String): LongArray {
        val tokens = mutableListOf<Long>()
        val words = text.lowercase().replace("?", " ?").replace(".", " .").split(Regex("\\s+"))

        for (word in words) {
            if (word.isEmpty()) continue
            // Safe hash fallback for unknown words keeping them in valid range
            val tokenId = vocab[word] ?: (word.hashCode().toLong() and 0x7FFFFFFF) % 40000
            tokens.add(tokenId)
        }

        return tokens.toLongArray()
    }

    fun decode(tokens: LongArray): String {
        return tokens.toList()
            .map { id -> reverseVocab[id] ?: " " }
            .joinToString("")
            .replace("  ", " ")
    }

    fun formatPrompt(question: String): String {
        return "<image>\n\nQuestion: $question\n\nAnswer:"
    }
}