package io.github.kazeevn.silerotts.core

import java.nio.ByteBuffer
import java.nio.charset.StandardCharsets

internal const val VOWELS = "аоуыэиеяёю"
internal const val STRESS = '+'

/**
 * Port of `models/accentor.py` (AccentorNgram) from the Silero package: a
 * fastText-style n-gram embedding bag + two small MLPs predicting the stressed
 * vowel and the position of "ё", plus an exception dictionary.
 */
class Accentor(accentorBin: ByteBuffer, private val ngrams: StringHashTable, exceptionsTsv: String, private val ngramMaxLen: Int) {
    private val vocabSize: Int
    private val dim: Int
    private val scale: Float
    private val zeroPoint: Float
    private val embedding: ByteBuffer
    private val stressMlp: Mlp
    private val yoMlp: Mlp
    private val unk = ngrams["UNK"]
    private val exceptions = HashMap<String, IntArray>()

    init {
        val b = accentorBin.littleEndian()
        val magic = ByteArray(4).also { b.get(it) }
        require(String(magic, StandardCharsets.US_ASCII) == "ACC1") { "not an accentor file" }
        vocabSize = b.getInt()
        dim = b.getInt()
        scale = b.getFloat()
        zeroPoint = b.getFloat()
        embedding = b.sliceAt(b.position(), vocabSize * dim)
        b.position(b.position() + vocabSize * dim + ((4 - vocabSize * dim % 4) % 4))
        stressMlp = Mlp.read(b)
        yoMlp = Mlp.read(b)
        for (line in exceptionsTsv.lineSequence()) {
            if (line.isEmpty()) continue
            val p = line.split('\t')
            exceptions[p[0]] = intArrayOf(p[1].toInt(), p[2].toInt())
        }
    }

    private class Mlp(val layers: List<Pair<Array<FloatArray>, FloatArray>>) {
        fun forward(x: FloatArray): FloatArray {
            var h = x
            for ((li, layer) in layers.withIndex()) {
                val (w, bias) = layer
                val out = FloatArray(w.size)
                for (o in w.indices) {
                    val row = w[o]
                    var acc = bias[o]
                    for (i in row.indices) acc += row[i] * h[i]
                    out[o] = if (li < layers.size - 1 && acc < 0f) 0f else acc
                }
                h = out
            }
            return h
        }

        companion object {
            fun read(b: ByteBuffer): Mlp {
                val n = b.getInt()
                val layers = (0 until n).map {
                    val out = b.getInt()
                    val inp = b.getInt()
                    val w = Array(out) { FloatArray(inp) { b.getFloat() } }
                    val bias = FloatArray(out) { b.getFloat() }
                    w to bias
                }
                return Mlp(layers)
            }
        }
    }

    /** Mean of the dequantized embeddings of all known n-grams of "<word>" (EmbeddingBag mode="mean"). */
    private fun embed(word: String): FloatArray {
        val t = "<$word>"
        val sum = FloatArray(dim)
        var count = 0
        fun add(id: Int) {
            val base = id * dim
            for (d in 0 until dim) sum[d] += embedding.get(base + d).toFloat()
            count++
        }
        val maxN = minOf(word.length + 3, ngramMaxLen)
        for (n in 1..maxN) {
            for (i in 0..t.length - n) {
                val id = ngrams.get(t, i, i + n, false)
                if (id >= 0) add(id)
            }
        }
        if (count == 0) add(unk)
        // sum of (q - zp) * scale over ids, divided by count
        for (d in 0 until dim) sum[d] = scale * (sum[d] - count * zeroPoint) / count
        return sum
    }

    private fun argmaxWithProb(logits: FloatArray): Pair<Int, Float> {
        var k = 0
        for (i in logits.indices) if (logits[i] > logits[k]) k = i
        var z = 0.0
        for (v in logits) z += Math.exp((v - logits[k]).toDouble())
        return k to (1.0 / z).toFloat()
    }

    data class Options(
        val putStress: Boolean = true,
        val putYo: Boolean = true,
        val stressSingleVowel: Boolean = true,
        val skipStressWords: Set<String> = emptySet(),
        val skipYoWords: Set<String> = emptySet(),
    )

    fun accentuate(sentence: String, opt: Options = Options()): String {
        if (!(opt.putStress || opt.putYo)) return sentence
        val out = StringBuilder(sentence.length + sentence.length / 4)
        for ((rawWord0, cleanWord, need) in tokenize(sentence)) {
            var rawWord = rawWord0
            val low = rawWord.lowercase()
            if (!need) {
                out.append(rawWord)
                continue
            }
            val haveStress = STRESS in low
            val haveYo = 'ё' in low
            if (haveStress && haveYo) {
                out.append(rawWord)
                continue
            }
            if (!haveStress && haveYo && opt.putStress) {
                if ((low.count { it in VOWELS } == 1 && !opt.stressSingleVowel) || cleanWord.replace('ё', 'е') in opt.skipStressWords) {
                    out.append(rawWord)
                    continue
                }
                var shift = 0
                val sb = StringBuilder(rawWord)
                for ((i, c) in low.withIndex()) if (c == 'ё') {
                    sb.insert(i + shift, STRESS)
                    shift++
                }
                out.append(sb)
                continue
            }
            exceptions[cleanWord]?.let {
                out.append(exception(it, rawWord, haveStress))
                continue
            }
            val emb = embed(cleanWord)
            val (sIdx, sProb) = argmaxWithProb(stressMlp.forward(emb))
            val (yIdx, yProb) = argmaxWithProb(yoMlp.forward(emb))
            var stressed = intArrayOf(sIdx)
            val passedStress = sProb > (if (opt.putStress) 0.5f else 1f)
            var setStress = passedStress && !haveStress && cleanWord.replace('ё', 'е') !in opt.skipStressWords
            val passedYo = yProb > (if (opt.putYo) 0.5f else 1f)
            val setYo = passedYo && cleanWord.replace('ё', 'е') !in opt.skipYoWords
            if (haveStress) {
                stressed = low.split(STRESS).map { part -> part.count { it in VOWELS } }.toIntArray()
            }
            val vowelIds = low.indices.filter { low[it] in VOWELS }
            val yeIds = low.indices.filter { low[it] == 'е' }
            var stressPos = stressed.filter { it < vowelIds.size }.map { vowelIds[it] }
            val yoPos = if (yIdx > 0 && yIdx - 1 < yeIds.size) listOf(yeIds[yIdx - 1]) else emptyList()
            if (vowelIds.isEmpty()) {
                out.append(rawWord)
                continue
            }
            for (p in yoPos) {
                if (p in stressPos && setYo && low[p] == 'е') {
                    rawWord = rawWord.substring(0, p) + (if (rawWord[p].isLowerCase()) 'ё' else 'Ё') + rawWord.substring(p + 1)
                }
            }
            if (vowelIds.size == 1) {
                stressPos = listOf(vowelIds[0])
                setStress = opt.stressSingleVowel && opt.putStress
            }
            if (!haveStress && setStress) {
                val sb = StringBuilder(rawWord)
                for ((i, p) in stressPos.withIndex()) sb.insert(p + i, STRESS)
                rawWord = sb.toString()
            }
            out.append(rawWord)
        }
        return out.toString()
    }

    private fun exception(exc: IntArray, rawWord: String, haveStress: Boolean): String {
        val (excStress, excYo) = exc[0] to exc[1]
        fun yo(w: String, p: Int) = w.substring(0, p) + (if (w[p].isLowerCase()) 'ё' else 'Ё') + w.substring(p + 1)
        if (haveStress) {
            val user = rawWord.indices.filter { rawWord[it] == STRESS }
            var w = rawWord.replace(STRESS.toString(), "")
            if (excYo != -1 && (excYo + 1) in user) w = yo(w, excYo)
            val sb = StringBuilder(w)
            for (p in user) sb.insert(p, STRESS)
            return sb.toString()
        }
        val w = if (excYo != -1) yo(rawWord, excYo) else rawWord
        return w.substring(0, excStress) + STRESS + w.substring(excStress)
    }

    internal data class Token(val raw: String, val clean: String, val needsPrediction: Boolean)

    companion object {
        private const val SEPARATORS = ".,!?;:<>=()/\\"

        private fun isSeparator(c: Char) = c.isWhitespace() || c in SEPARATORS

        private fun cleanForModel(t: String): String {
            val sb = StringBuilder()
            for (c in t.lowercase()) if (c in 'а'..'я' || c == 'ё') sb.append(c)
            return sb.toString()
        }

        /** Equivalent of `re.split(r'([\s.,!?;:<>=()/\\]+)', sentence)` + hyphen handling. */
        internal fun tokenize(sentence: String): List<Token> {
            val out = ArrayList<Token>()
            var i = 0
            while (i < sentence.length) {
                val sep = isSeparator(sentence[i])
                var j = i
                while (j < sentence.length && isSeparator(sentence[j]) == sep) j++
                val run = sentence.substring(i, j)
                if (sep) {
                    out.add(Token(run, "", false))
                } else {
                    val parts = run.split('-')
                    for ((k, part) in parts.withIndex()) {
                        val last = k == parts.size - 1
                        val raw = if (last) part else "$part-"
                        val clean = cleanForModel(raw)
                        // backward compatibility for "что-то", "кому-то"...
                        val mask = !(last && parts.size > 1 && part == "то")
                        out.add(Token(raw, clean, mask && clean.isNotEmpty()))
                    }
                }
                i = j
            }
            return out
        }
    }
}
