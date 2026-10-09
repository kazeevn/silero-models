package io.github.kazeevn.silerotts.core

import java.nio.ByteBuffer
import java.nio.charset.StandardCharsets

/** Port of `custom_tokenizers/bert_tokenizer.py` (no lower-casing, no accent stripping). */
class BertTokenizer(private val vocab: StringHashTable, private val ids: ModelConfig.BertIds) {

    private fun isPunct(c: Char): Boolean {
        val cp = c.code
        if (cp in 33..47 || cp in 58..64 || cp in 91..96 || cp in 123..126) return true
        return when (Character.getType(c).toByte()) {
            Character.CONNECTOR_PUNCTUATION, Character.DASH_PUNCTUATION, Character.START_PUNCTUATION,
            Character.END_PUNCTUATION, Character.INITIAL_QUOTE_PUNCTUATION, Character.FINAL_QUOTE_PUNCTUATION,
            Character.OTHER_PUNCTUATION -> true
            else -> false
        }
    }

    private fun isControl(c: Char): Boolean {
        if (c == '\t' || c == '\n' || c == '\r') return false
        return when (Character.getType(c).toByte()) {
            Character.CONTROL, Character.FORMAT, Character.SURROGATE, Character.PRIVATE_USE, Character.UNASSIGNED -> true
            else -> false
        }
    }

    private fun isWhitespace(c: Char) =
        c == ' ' || c == '\t' || c == '\n' || c == '\r' || Character.getType(c).toByte() == Character.SPACE_SEPARATOR

    private fun basicTokens(text: String): List<String> {
        val cleaned = StringBuilder(text.length)
        for (c in text) {
            if (c.code == 0 || c.code == 0xFFFD || isControl(c)) continue
            cleaned.append(if (isWhitespace(c)) ' ' else c)
        }
        val out = ArrayList<String>()
        for (tok in cleaned.split(' ')) {
            if (tok.isEmpty()) continue
            if (tok == HOMO || tok == HOMO_END) {
                out.add(tok)
                continue
            }
            var start = 0
            for (i in tok.indices) {
                if (isPunct(tok[i])) {
                    if (i > start) out.add(tok.substring(start, i))
                    out.add(tok[i].toString())
                    start = i + 1
                }
            }
            if (start < tok.length) out.add(tok.substring(start))
        }
        return out
    }

    private fun wordpiece(tok: String, out: MutableList<Int>) {
        if (tok.length > 100) {
            out.add(ids.unk)
            return
        }
        val pieces = ArrayList<Int>()
        var start = 0
        while (start < tok.length) {
            var end = tok.length
            var cur = -1
            while (start < end) {
                cur = vocab.get(tok, start, end, start > 0)
                if (cur >= 0) break
                end--
            }
            if (cur < 0) {
                out.add(ids.unk)
                return
            }
            pieces.add(cur)
            start = end
        }
        out.addAll(pieces)
    }

    fun encode(text: String): IntArray {
        val out = ArrayList<Int>()
        out.add(ids.cls)
        for (t in basicTokens(text)) {
            when (t) {
                HOMO -> out.add(ids.homoStart)
                HOMO_END -> out.add(ids.homoEnd)
                else -> wordpiece(t, out)
            }
        }
        out.add(ids.sep)
        return out.toIntArray()
    }

    companion object {
        const val HOMO = "[HOMO]"
        const val HOMO_END = "[/HOMO]"
    }
}

/**
 * int8 BERT word embeddings (bert_emb.bin), read in place from the mapped
 * asset. Rows are de-quantized exactly like the original package does.
 */
class BertEmbeddings(buffer: ByteBuffer) {
    private val data = buffer.littleEndian()
    val dim: Int
    private val rows: Int
    private val scale: Float
    private val zeroPoint: Float

    init {
        val magic = ByteArray(4).also { data.get(it) }
        require(String(magic, StandardCharsets.US_ASCII) == "BEM1") { "not a BEM1 file" }
        rows = data.getInt()
        dim = data.getInt()
        scale = data.getFloat()
        zeroPoint = data.getFloat()
    }

    /** Embeddings of [ids], zero-padded to [size] rows. */
    fun lookup(ids: IntArray, size: Int): FloatArray {
        val out = FloatArray(size * dim)
        for ((i, id) in ids.withIndex()) {
            require(id in 0 until rows) { "wordpiece id $id out of range" }
            val base = HEADER + id * dim
            for (k in 0 until dim) out[i * dim + k] = (data.get(base + k).toFloat() - zeroPoint) * scale
        }
        return out
    }

    private companion object {
        const val HEADER = 20
    }
}

/** Port of `models/homosolver.py`: BERT picks the reading of known homographs ("з+амок" / "зам+ок"). */
class HomoSolver(
    private val network: Network,
    private val embeddings: BertEmbeddings,
    private val sizes: SignatureSizes,
    private val tokenizer: BertTokenizer,
    homodictTsv: String,
    private val ids: ModelConfig.BertIds,
) : java.io.Closeable {
    override fun close() = network.close()

    /** Classifier logit for the homograph between the [start] and [end] markers of [tokens]. */
    private fun logit(tokens: IntArray, start: Int, end: Int): Float {
        val size = sizes.fit(tokens.size)
        val inputs = mapOf(
            "embeddings" to embeddings.lookup(tokens, size),
            "mask" to lengthMask(tokens.size, size),
            "start" to intArrayOf(start),
            "end" to intArrayOf(end),
        )
        return network.run(sizes.name(size), inputs, OUTPUTS).getValue("logit")[0]
    }

    /** word -> variants sorted the way the original picks them (index = model prediction). */
    val homodict: Map<String, List<String>>
    /** Homographs differing by "ё" (used when "ё" restoration for homographs is off). */
    val yoHomographs: Set<String>

    init {
        val d = HashMap<String, List<String>>()
        for (line in homodictTsv.lineSequence()) {
            if (line.isEmpty()) continue
            val p = line.split('\t')
            d[p[0]] = p.subList(1, p.size)
        }
        homodict = d
        yoHomographs = d.filterValues { v -> v.any { 'ё' in it } }.keys
    }

    private fun isWordChar(c: Char) = c in 'а'..'я' || c in 'А'..'Я' || c == 'ё' || c == 'Ё' || c == STRESS

    /**
     * Picks the readings of the homographs in [sentence]. [left] and [right]
     * are the (cleaned) text around it: BERT sees them as context, but only
     * the words of [sentence] are resolved. The original package resolves the
     * whole input text at once; the engine synthesizes it sentence by sentence
     * and passes the neighbouring text here instead.
     */
    fun solve(
        sentence: String,
        putStress: Boolean = true,
        putYo: Boolean = true,
        stressSingleVowel: Boolean = true,
        left: String = "",
        right: String = "",
    ): String {
        if (!(putStress || putYo)) return sentence
        data class Found(val start: Int, val end: Int, val word: String, val ids: IntArray, val marker: Int, val markerEnd: Int)
        val found = ArrayList<Found>()
        // regex (?=.*[а-яё])[а-яё+]+ (ignore case): maximal runs; only pure-letter runs can be dictionary words
        var i = 0
        while (i < sentence.length) {
            if (!isWordChar(sentence[i])) {
                i++
                continue
            }
            var j = i
            while (j < sentence.length && isWordChar(sentence[j])) j++
            val word = sentence.substring(i, j)
            if (word.lowercase() in homodict) {
                val marked = sentence.substring(0, i) + " ${BertTokenizer.HOMO} " + word + " ${BertTokenizer.HOMO_END} " + sentence.substring(j)
                var tok = tokenizer.encode(listOf(left, marked, right).filter { it.isNotEmpty() }.joinToString(" "))
                var s = tok.indexOf(ids.homoStart)
                var e = tok.indexOf(ids.homoEnd)
                if (tok.size > ids.maxLen) {
                    // keep a window of the context around the homograph (the original
                    // has no limit, but BERT has 512 positions)
                    val inner = ids.maxLen - 2
                    val from = (s - inner / 2).coerceIn(1, tok.size - 1 - inner)
                    tok = intArrayOf(ids.cls) + tok.copyOfRange(from, from + inner) + intArrayOf(ids.sep)
                    s -= from - 1
                    e -= from - 1
                }
                found.add(Found(i, j, word, tok, s, e))
            }
            i = j
        }
        if (found.isEmpty()) return sentence

        // One homograph per run: the sequences of a sentence all have the same
        // length, so the original's batch never contains padding either.
        val logits = FloatArray(found.size) { logit(found[it].ids, found[it].marker, found[it].markerEnd) }

        val out = StringBuilder(sentence)
        var offset = 0
        for ((k, f) in found.withIndex()) {
            // round(sigmoid(x)) == 1  <=>  x > 0
            val pred = if (logits[k] > 0f) 1 else 0
            var variant = homodict.getValue(f.word.lowercase())[pred]
            if (!putYo) variant = variant.replace('ё', 'е')
            val nVowels = variant.count { it.lowercaseChar() in VOWELS }
            val stressIdx = variant.indexOf(STRESS)
            val plain = variant.replace(STRESS.toString(), "")
            val cased = StringBuilder()
            for (c in f.word.indices) {
                if (c >= plain.length) break
                cased.append(if (f.word[c].isLowerCase()) plain[c].lowercaseChar() else plain[c].uppercaseChar())
            }
            var replacement = cased.toString()
            val s = f.start + offset
            val e = f.end + offset
            if ((nVowels > 1 || stressSingleVowel) && putStress) {
                replacement = replacement.substring(0, stressIdx) + STRESS + replacement.substring(stressIdx)
                offset += 1
            }
            out.replace(s, e, replacement)
        }
        return out.toString()
    }

    private companion object {
        val OUTPUTS = listOf("logit")
    }
}
