package io.github.kazeevn.silerotts.core

/**
 * Port of `classify_text` / `build_type_ids_inference` from the Silero package:
 * the pitch predictor is conditioned on the utterance type of every character
 * (statement, wh-question, yes/no question, alternative question, tag question,
 * exclamation).
 */
object SentenceType {
    const val STATEMENT = 0
    private val TYPE_IDS = mapOf("st" to 0, "wh_q" to 1, "general_q" to 2, "alternative_q" to 3, "tag_q" to 4, "exclam" to 5)

    private val WH_FORMS = """кто кого кому кем ком что чего чему чем чём чё чо где куда откуда когда почему зачем как
        почём отчего насколько сколько скольких скольким сколькими какой какая какое какие какого каких какому
        каким какими каком какую чей чья чьё чье чьи чьего чьей чьих чьему чьим чьими чьём чьем чью который
        которая которое которые которого которой которых которому которым которыми котором которую каков какова
        каково каковы""".split(Regex("\\s+")).filter { it.isNotEmpty() }.toSet()
    private val LEADING_FILLERS = "а ну и так вот слышь слушай скажите скажи пожалуйста вобще вообще типа короче".split(' ').toSet()

    // applied to lower-cased text (the original uses re.IGNORECASE)
    private val TAG_RE = Regex(
        listOf(
            "[,\\s]+(правда|верно|да)\\s*\\?$",
            "[,\\s]+(не\\s+так\\s+ли|не\\s+правда\\s+ли|разве\\s+не\\s+так|ведь\\s+так)\\s*\\?$",
            "[,\\s]+ведь\\s*\\?$",
            "[,\\s]+а\\s*\\?$",
        ).joinToString("|") { "(?:$it)" }
    )
    private val SENT_SPLIT = Regex("(?<=[.!?])\\s+")
    private const val QUOTE_OPEN = "\"«“„"
    private const val QUOTE_CLOSE = "\"»”’"

    private fun stripQuotes(s0: String): String {
        var s = s0.trim()
        while (s.isNotEmpty() && s[0] in QUOTE_OPEN) s = s.substring(1).trim()
        while (s.isNotEmpty() && s[s.length - 1] in QUOTE_CLOSE) s = s.substring(0, s.length - 1).trim()
        return s
    }

    private fun isWordLetter(c: Char) = c in 'а'..'я' || c == 'ё' || c in 'a'..'z'

    /** `[а-яёa-z]+` matches of a lower-cased string. */
    private fun words(lower: String): List<String> {
        val out = ArrayList<String>()
        var i = 0
        while (i < lower.length) {
            if (!isWordLetter(lower[i])) {
                i++
                continue
            }
            var j = i
            while (j < lower.length && isWordLetter(lower[j])) j++
            out.add(lower.substring(i, j))
            i = j
        }
        return out
    }

    /** `\bили\b`, with Unicode word characters. */
    private fun hasOr(lower: String): Boolean {
        var idx = lower.indexOf("или")
        while (idx >= 0) {
            val before = idx == 0 || !isUnicodeWordChar(lower[idx - 1])
            val after = idx + 3 >= lower.length || !isUnicodeWordChar(lower[idx + 3])
            if (before && after) return true
            idx = lower.indexOf("или", idx + 1)
        }
        return false
    }

    private fun isUnicodeWordChar(c: Char) = c.isLetterOrDigit() || c == '_'

    fun classifySentence(s0: String): String {
        val s = s0.trim()
        if (s.isEmpty()) return "st"
        val clean = stripQuotes(s)
        if (clean.isEmpty()) return "st"
        var tail = clean
        if (tail.endsWith("?!") || tail.endsWith("?..") || tail.endsWith("?...")) tail = tail.trimEnd('.', '!')
        if (tail.endsWith("?")) {
            val q = stripQuotes(tail).replace("+", "").replace('ё', 'е').replace('Ё', 'Е')
                .replace(Regex("\\s+"), " ").trim()
            val lower = q.lowercase()
            if (TAG_RE.containsMatchIn(lower)) return "tag_q"
            val content = ArrayList<String>()
            for (w in words(lower).take(8)) {
                if (w in LEADING_FILLERS) continue
                content.add(w)
                if (content.size >= 4) break
            }
            if (content.any { it in WH_FORMS }) return "wh_q"
            if (hasOr(lower)) return "alternative_q"
            return "general_q"
        }
        if (clean.endsWith("!")) return "exclam"
        return "st"
    }

    fun classifyText(text: String): List<String> {
        if (text.isBlank()) return listOf("st")
        return SENT_SPLIT.split(text.trim()).map { it.trim() }.filter { it.isNotEmpty() }.map { classifySentence(it) }
    }

    /**
     * Per-token utterance type ids: the original maps characters of the *raw*
     * text sentence by sentence onto the token sequence (offset by the start
     * token), padding with the first sentence's type.
     */
    fun typeIds(text: String, seqLen: Int): LongArray {
        val out = LongArray(seqLen)
        if (text.isBlank()) return out
        val types = classifyText(text)
        val sentences = SENT_SPLIT.split(text.trim())
        val default = TYPE_IDS.getValue(types[0]).toLong()
        out.fill(default)
        var idx = 1
        for ((i, s) in sentences.withIndex()) {
            val tid = TYPE_IDS.getValue(if (i < types.size) types[i] else types.last()).toLong()
            val n = s.length + if (i < sentences.size - 1) 1 else 0
            for (k in 0 until n) {
                if (idx >= seqLen) return out
                out[idx++] = tid
            }
        }
        return out
    }
}
