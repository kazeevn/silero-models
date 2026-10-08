package io.github.kazeevn.silerotts.core

import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.nio.charset.StandardCharsets

/** Read-only access to the exported model files (see tools/optimize_models.py). */
interface AssetSource {
    /**
     * Returns a direct buffer with the whole file. Implementations memory-map the
     * file when possible so that the frontend tables are read in place without
     * copying.
     */
    fun map(name: String): ByteBuffer

    fun text(name: String): String {
        val buf = map(name).duplicate()
        return StandardCharsets.UTF_8.decode(buf).toString()
    }
}

class DirectoryAssetSource(private val dir: File) : AssetSource {
    override fun map(name: String): ByteBuffer =
        RandomAccessFile(File(dir, name), "r").use { f ->
            f.channel.map(FileChannel.MapMode.READ_ONLY, 0, f.length())
        }
}

internal fun ByteBuffer.littleEndian(): ByteBuffer = duplicate().order(ByteOrder.LITTLE_ENDIAN)

/** Little-endian view of `length` bytes at absolute `offset`. */
internal fun ByteBuffer.sliceAt(offset: Int, length: Int): ByteBuffer = slice(offset, length).order(ByteOrder.LITTLE_ENDIAN)

/** Constants exported to config.json. */
class ModelConfig(json: Map<String, Any?>) {
    val symbols: String = json["symbols"] as String
    val symbolToId: Map<Char, Int> =
        (json["symbol_to_id"] as Map<*, *>).entries.associate { (k, v) -> (k as String)[0] to (v as Number).toInt() }
    /** Speaker name -> embedding id, in model order. */
    val speakers: Map<String, Int> =
        (json["speakers"] as Map<*, *>).entries.associateTo(LinkedHashMap()) { (k, v) -> k as String to (v as Number).toInt() }
    val meanStdCoef: FloatArray = (json["mean_std_coef"] as List<*>).map { (it as Number).toFloat() }.toFloatArray()
    val nFft: Int = (json["n_fft"] as Number).toInt()
    val hopLength: Int = (json["hop_length"] as Number).toInt()
    val winLength: Int = (json["win_length"] as Number).toInt()
    val pqmf: Map<Int, FloatArray> = (json["pqmf"] as Map<*, *>).entries.associate { (k, v) ->
        (k as String).toInt() to (v as List<*>).map { (it as Number).toFloat() }.toFloatArray()
    }
    val pqmfTaps: Int = (json["pqmf_taps"] as Number).toInt()
    val ngramMaxLen: Int = (json["ngram_max_len"] as Number).toInt()
    val bert: BertIds = (json["bert"] as Map<*, *>).let { b ->
        fun i(k: String) = (b[k] as Number).toInt()
        BertIds(i("pad"), i("cls"), i("sep"), i("unk"), i("homo_start"), i("homo_end"), i("max_len"))
    }

    /** Static signature sizes of the networks. */
    val textSizes: SignatureSizes = sizes(json, "text", "L")
    val decoderSizes: SignatureSizes = sizes(json, "decoder", "T")
    val vocoderSizes: SignatureSizes = sizes(json, "vocoder", "W")
    val bertSizes: SignatureSizes = sizes(json, "homosolver", "S")
    /** Frames of context around a vocoder window (>= the vocoder's receptive field). */
    val vocoderContext: Int = (json["vocoder_context"] as Number).toInt()
    val nMels: Int = (json["n_mels"] as Number).toInt()
    /** Width of the encoder output (decoder input) per token / frame. */
    val hidden: Int = (json["hidden"] as Number).toInt()

    /** Characters the acoustic model accepts after the control symbols _, ~ and |. */
    val textAlphabet: String = symbols.substring(3)

    data class BertIds(val pad: Int, val cls: Int, val sep: Int, val unk: Int, val homoStart: Int, val homoEnd: Int, val maxLen: Int)

    companion object {
        private fun sizes(json: Map<String, Any?>, network: String, prefix: String): SignatureSizes {
            val list = (json["sizes"] as Map<*, *>)[network] as List<*>
            return SignatureSizes(prefix, list.map { (it as Number).toInt() }.toIntArray())
        }

        fun load(assets: AssetSource): ModelConfig {
            @Suppress("UNCHECKED_CAST")
            return ModelConfig(Json.parse(assets.text("config.json")) as Map<String, Any?>)
        }
    }
}

/** Minimal JSON reader (objects, arrays, strings, numbers, booleans, null). */
internal object Json {
    fun parse(s: String): Any? {
        val p = Parser(s)
        val v = p.value()
        p.ws()
        require(p.i == s.length) { "trailing data at ${p.i}" }
        return v
    }

    private class Parser(val s: String) {
        var i = 0

        fun ws() {
            while (i < s.length && s[i].isWhitespace()) i++
        }

        fun value(): Any? {
            ws()
            require(i < s.length) { "unexpected end of JSON" }
            return when (val c = s[i]) {
                '{' -> obj()
                '[' -> arr()
                '"' -> str()
                't' -> lit("true", true)
                'f' -> lit("false", false)
                'n' -> lit("null", null)
                else -> if (c == '-' || c.isDigit()) num() else throw IllegalArgumentException("bad JSON at $i")
            }
        }

        private fun lit(word: String, v: Any?): Any? {
            require(s.startsWith(word, i)) { "bad literal at $i" }
            i += word.length
            return v
        }

        private fun num(): Number {
            val start = i
            while (i < s.length && (s[i].isDigit() || s[i] in "+-.eE")) i++
            val t = s.substring(start, i)
            return if (t.any { it == '.' || it == 'e' || it == 'E' }) t.toDouble() else t.toLong()
        }

        private fun str(): String {
            i++ // opening quote
            val sb = StringBuilder()
            while (true) {
                val c = s[i++]
                when (c) {
                    '"' -> return sb.toString()
                    '\\' -> {
                        when (val e = s[i++]) {
                            'n' -> sb.append('\n')
                            't' -> sb.append('\t')
                            'r' -> sb.append('\r')
                            'b' -> sb.append('\b')
                            'f' -> sb.append('\u000c')
                            'u' -> {
                                sb.append(s.substring(i, i + 4).toInt(16).toChar())
                                i += 4
                            }
                            else -> sb.append(e)
                        }
                    }
                    else -> sb.append(c)
                }
            }
        }

        private fun arr(): List<Any?> {
            i++
            val out = ArrayList<Any?>()
            ws()
            if (s[i] == ']') {
                i++
                return out
            }
            while (true) {
                out.add(value())
                ws()
                when (s[i++]) {
                    ',' -> continue
                    ']' -> return out
                    else -> throw IllegalArgumentException("bad array at ${i - 1}")
                }
            }
        }

        private fun obj(): Map<String, Any?> {
            i++
            val out = LinkedHashMap<String, Any?>()
            ws()
            if (s[i] == '}') {
                i++
                return out
            }
            while (true) {
                ws()
                val k = str()
                ws()
                require(s[i++] == ':') { "expected ':' at ${i - 1}" }
                out[k] = value()
                ws()
                when (s[i++]) {
                    ',' -> continue
                    '}' -> return out
                    else -> throw IllegalArgumentException("bad object at ${i - 1}")
                }
            }
        }
    }
}

/**
 * Read-only open-addressing hash table String -> Int written by
 * tools/export_models.py (write_hash_table), used for the 126k accentor
 * n-grams and the BERT wordpiece vocabulary. It is read in place from the
 * mapped asset, so it costs no heap and no load time.
 */
class StringHashTable(buffer: ByteBuffer) {
    private val nSlots: Int
    private val slots: java.nio.IntBuffer
    private val values: java.nio.IntBuffer
    private val offsets: java.nio.IntBuffer
    private val blob: java.nio.CharBuffer
    val size: Int

    init {
        val b = buffer.littleEndian()
        val magic = ByteArray(4).also { b.get(it) }
        require(String(magic, StandardCharsets.US_ASCII) == "SHT1") { "not a SHT1 table" }
        nSlots = b.getInt()
        size = b.getInt()
        var off = 12
        slots = b.sliceAt(off, 4 * nSlots).asIntBuffer()
        off += 4 * nSlots
        values = b.sliceAt(off, 4 * size).asIntBuffer()
        off += 4 * size
        offsets = b.sliceAt(off, 4 * (size + 1)).asIntBuffer()
        off += 4 * (size + 1)
        blob = b.sliceAt(off, 2 * offsets.get(size)).asCharBuffer()
    }

    operator fun get(key: CharSequence): Int = get(key, 0, key.length, false)

    /**
     * Looks up `key[from, to)`, optionally prefixed with "##" (wordpiece
     * continuation), without allocating. Returns -1 when absent.
     */
    fun get(key: CharSequence, from: Int, to: Int, hashPrefix: Boolean): Int {
        var h = FNV_OFFSET
        if (hashPrefix) {
            h = (h xor '#'.code) * FNV_PRIME
            h = (h xor '#'.code) * FNV_PRIME
        }
        for (i in from until to) h = (h xor key[i].code) * FNV_PRIME
        val len = to - from + if (hashPrefix) 2 else 0
        val mask = nSlots - 1
        var s = h and mask
        while (true) {
            val e = slots.get(s)
            if (e < 0) return -1
            val a = offsets.get(e)
            if (offsets.get(e + 1) - a == len && matches(a, key, from, to, hashPrefix)) return values.get(e)
            s = (s + 1) and mask
        }
    }

    private fun matches(a: Int, key: CharSequence, from: Int, to: Int, hashPrefix: Boolean): Boolean {
        var p = a
        if (hashPrefix) {
            if (blob.get(p) != '#' || blob.get(p + 1) != '#') return false
            p += 2
        }
        for (i in from until to) if (blob.get(p++) != key[i]) return false
        return true
    }

    private companion object {
        const val FNV_OFFSET = 0x811C9DC5.toInt()
        const val FNV_PRIME = 0x01000193
    }
}
