package io.github.kazeevn.silerotts.core

import java.io.Closeable
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/** Per-request synthesis parameters. */
data class SynthesisParams(
    val speaker: String = "xenia",
    /** 48000 (native), 24000 or 8000. */
    val sampleRate: Int = 24000,
    /** Speed factor, 1.0 = normal. */
    val rate: Float = 1f,
    /** Pitch factor, 1.0 = normal (0 = monotone). */
    val pitch: Float = 1f,
    val putStress: Boolean = true,
    val putStressHomo: Boolean = true,
    val putYo: Boolean = true,
    val putYoHomo: Boolean = true,
)

fun interface AudioSink {
    /** Receives mono float samples in [-1, 1]. Return false to abort synthesis. */
    fun write(samples: FloatArray, count: Int): Boolean
}

/** Timing info of the last [SileroEngine.synthesizeChunk] call (milliseconds). */
data class ChunkStats(
    var frontendMs: Double = 0.0,
    /** Duration / pitch predictors and encoder (text.tflite). */
    var textMs: Double = 0.0,
    /** Length regulator and decoder (decoder.tflite). */
    var decoderMs: Double = 0.0,
    var vocoderMs: Double = 0.0,
    var firstAudioMs: Double = 0.0,
    var audioSeconds: Double = 0.0,
)

/**
 * On-device Silero v5_5_ru engine: LiteRT for the networks, everything else
 * ported from the original Python package (see tools/reference_pipeline.py,
 * which this class mirrors step by step).
 */
class SileroEngine(assets: AssetSource, loader: NetworkLoader) : Closeable {
    val config = ModelConfig.load(assets)
    val speakers: List<String> get() = config.speakers.keys.toList()

    private val networks = ArrayList<Network>()
    private val textNet: Network
    private val decoder: Network
    private val vocoder: Network
    private val homo: HomoSolver
    private val accentor: Accentor
    val lastStats = ChunkStats()

    init {
        try {
            fun load(name: String) = loader.load(name).also { networks.add(it) }
            textNet = load("text")
            decoder = load("decoder")
            vocoder = load("vocoder")
            val vocab = StringHashTable(assets.map("bert_vocab.bin"))
            homo = HomoSolver(
                load("homosolver"), BertEmbeddings(assets.map("bert_emb.bin")), config.bertSizes,
                BertTokenizer(vocab, config.bert), assets.text("homodict.tsv"), config.bert,
            )
            accentor = Accentor(assets.map("accentor.bin"), StringHashTable(assets.map("ngrams.bin")), assets.text("exceptions.tsv"), config.ngramMaxLen)
        } catch (e: Throwable) {
            close()
            throw e
        }
    }

    private val allowed: Set<Char> = config.textAlphabet.toSet()

    /** `prepare_tts_model_input` + `prepare_text_input` for plain text. */
    fun clean(text: String): String {
        val t = text.trim()
        val sb = StringBuilder(t.length)
        for ((i, c) in t.withIndex()) {
            if (c == '\n') sb.append(if (i > 0 && t[i - 1] in ",;:.!?") " " else ". ") else sb.append(c)
        }
        val lower = sb.toString().lowercase().replace('—', '–').replace('‑', '-')
        val out = StringBuilder(lower.length)
        var space = false
        for (c in lower) {
            if (c !in allowed) continue
            if (c == ' ') {
                space = true
                continue
            }
            if (space && out.isNotEmpty()) out.append(' ')
            space = false
            out.append(c)
        }
        return out.toString()
    }

    /** Homograph resolution followed by the n-gram accentor (SileroStress.__call__). */
    fun accentuate(cleanText: String, p: SynthesisParams): String {
        val solved = homo.solve(cleanText, putStress = p.putStressHomo, putYo = p.putYoHomo)
        return accentor.accentuate(
            solved,
            Accentor.Options(
                putStress = p.putStress,
                putYo = p.putYo,
                skipStressWords = if (!p.putStressHomo) homo.homodict.keys else emptySet(),
                skipYoWords = if (!p.putYoHomo) homo.yoHomographs else emptySet(),
            ),
        )
    }

    fun hasSpeech(cleanText: String) = cleanText.any { it in 'а'..'я' || it == 'ё' }

    internal fun tokens(accented: String): IntArray {
        val seq = "|$accented~"
        return IntArray(seq.length) { config.symbolToId.getValue(seq[it]) }
    }

    /** Duration post-processing of MultiTTSModel.forward (first [n] log-durations -> frames). */
    internal fun durations(logDur: FloatArray, n: Int, rate: Float): IntArray {
        val d = FloatArray(n) { max(exp(logDur[it]) - 1f, 0f) }
        for (i in 0 until n) d[i] = Math.rint(d[i].toDouble()).toFloat()
        d[0] = min(d[0], 5f)
        if (rate != 1f) {
            for (i in 0 until n) {
                val orig = d[i]
                d[i] = Math.rint((d[i] / rate).toDouble()).toFloat()
                // Unlike the original, keep at least one frame per sound when
                // speeding up, otherwise fast speech (screen readers) drops phonemes.
                if (rate > 1f && orig >= 1f && d[i] < 1f) d[i] = 1f
            }
        }
        d[0] = min(d[0], 5f)
        d[n - 1] = min(d[n - 1], 7f)
        d[n - 2] = 13f
        d[n - 3] = min(d[n - 3], 13f)
        return IntArray(n) { d[it].toInt() }
    }

    /**
     * Full text-to-speech: normalization, sentence chunking and synthesis of
     * every chunk, streamed to [sink]. Returns false if the sink aborted.
     */
    fun synthesize(text: String, p: SynthesisParams, sink: AudioSink): Boolean {
        // Slow speech makes long sentences exceed the largest decoder signature
        // (28.8 s): split them more finely.
        val maxLen = (MAX_CHUNK_CHARS * p.rate).toInt().coerceIn(MIN_CHUNK_CHARS, MAX_CHUNK_CHARS)
        val chunks = TextNormalizer.chunks(TextNormalizer.normalize(text), maxLen)
        for (chunk in chunks) {
            if (!synthesizeChunk(chunk.text, p, sink, chunk.sentence)) return false
            if (chunk.pauseAfterMs > 0) {
                val silence = FloatArray(p.sampleRate * chunk.pauseAfterMs / 1000)
                if (!sink.write(silence, silence.size)) return false
            }
        }
        return true
    }

    /**
     * Synthesizes one chunk (a sentence or a part of it).
     * @param text raw chunk text (after [TextNormalizer]).
     * @param typeText text used for intonation classification (defaults to [text]).
     * @return false if the sink aborted.
     */
    fun synthesizeChunk(text: String, p: SynthesisParams, sink: AudioSink, typeText: String = text): Boolean {
        val t0 = System.nanoTime()
        val stats = lastStats
        stats.firstAudioMs = 0.0
        val cleanText = clean(text)
        if (!hasSpeech(cleanText)) return true
        val spk = config.speakers[p.speaker] ?: config.speakers.values.first()
        val accented = accentuate(cleanText, p)
        val tokens = tokens(accented)
        if (tokens.size > config.textSizes.max) return synthesizeHalves(text, p, sink, typeText)
        val typeIds = SentenceType.typeIds(typeText, tokens.size)
        val t1 = System.nanoTime()
        stats.frontendMs = (t1 - t0) / 1e6

        // durations, and the encoder output with the pitch applied
        // (update_pitch_coef: scale, shift = (coef - 1) * mean_std_coef)
        val n = tokens.size
        val textSize = config.textSizes.fit(n)
        val coef = p.pitch.coerceIn(0f, 2.5f)
        val shift = ((if (coef == 0f) 1f else coef) - 1f) * config.meanStdCoef[spk]
        val encoded = textNet.run(
            config.textSizes.name(textSize),
            mapOf(
                "tokens" to tokens.copyOf(textSize),
                "type_ids" to typeIds.copyOf(textSize),
                "speaker" to intArrayOf(spk),
                "mask" to lengthMask(n, textSize),
                "pitch_scale" to floatArrayOf(coef),
                "pitch_shift" to floatArrayOf(shift),
            ),
            TEXT_OUTPUTS,
        )
        val durs = durations(encoded.getValue("log_dur"), n, p.rate.coerceIn(0.2f, 4f))
        val total = durs.sum()
        if (total > config.decoderSizes.max) return synthesizeHalves(text, p, sink, typeText)
        val t2 = System.nanoTime()
        stats.textMs = (t2 - t1) / 1e6

        // length regulator: every frame gets the encoder output of its token
        val h = config.hidden
        val hidden = encoded.getValue("hidden")
        val frameSize = config.decoderSizes.fit(total)
        val frames = FloatArray(frameSize * h)
        var f = 0
        for ((i, d) in durs.withIndex()) {
            repeat(d) { System.arraycopy(hidden, i * h, frames, f++ * h, h) }
        }
        val mel = decoder.run(
            config.decoderSizes.name(frameSize),
            mapOf("frames" to frames, "mask" to lengthMask(total, frameSize)),
            DECODER_OUTPUTS,
        ).getValue("mel")
        val t3 = System.nanoTime()
        stats.decoderMs = (t3 - t2) / 1e6
        stats.audioSeconds = total * config.hopLength / 48000.0
        val ok = vocode(mel, total, p.sampleRate, sink, t0)
        stats.vocoderMs = (System.nanoTime() - t3) / 1e6
        return ok
    }

    /** Fallback for chunks too long for the largest signature: synthesize the halves. */
    private fun synthesizeHalves(text: String, p: SynthesisParams, sink: AudioSink, typeText: String): Boolean {
        val mid = text.length / 2
        var cut = text.lastIndexOf(' ', mid)
        if (cut <= 0) cut = text.indexOf(' ', mid)
        if (cut <= 0) cut = mid
        require(cut in 1 until text.length) { "cannot split \"$text\"" }
        return synthesizeChunk(text.substring(0, cut), p, sink, typeText) &&
            synthesizeChunk(text.substring(cut), p, sink, typeText)
    }

    /**
     * Runs the vocoder over the mel spectrogram ([total] frames, time-major) in
     * windows so that audio can be played before the whole sentence is vocoded.
     * Every window gets [ModelConfig.vocoderContext] frames of context on both
     * sides, which covers the receptive field of the ConvNeXt backbone (9
     * convolutions with kernel 7 = 27 frames), so the output is identical to a
     * single pass. The first window uses the smallest signature for a low
     * latency, the following ones the largest (mirrored in reference_pipeline.py).
     */
    private fun vocode(mel: FloatArray, total: Int, sampleRate: Int, sink: AudioSink, t0: Long): Boolean {
        val nMels = config.nMels
        val nBins = config.nFft / 2 + 1
        val sizes = config.vocoderSizes
        val context = config.vocoderContext
        val pending = SampleBuffer()
        val decimator = when (sampleRate) {
            48000 -> null
            24000 -> Decimator(config.pqmf.getValue(2), 2, pending::append)
            8000 -> Decimator(config.pqmf.getValue(6), 6, pending::append)
            else -> throw IllegalArgumentException("unsupported sample rate $sampleRate")
        }
        val istft = StreamingIstft(config.nFft, config.hopLength, total) { buf, n ->
            if (decimator != null) decimator.push(buf, n) else pending.append(buf, n)
        }
        var first = true
        fun flush(): Boolean {
            if (pending.size == 0) return true
            // the original soft-clips with tanh only when the peak exceeds 1; we
            // hard-limit so that streamed windows stay consistent
            for (i in 0 until pending.size) pending.data[i] = pending.data[i].coerceIn(-1f, 1f)
            if (first) {
                lastStats.firstAudioMs = (System.nanoTime() - t0) / 1e6
                first = false
            }
            val ok = sink.write(pending.data, pending.size)
            pending.size = 0
            return ok
        }

        var a = 0
        while (a < total) {
            // output frames [a, b) from input frames [s, e)
            val cap = if (a == 0) sizes.min else sizes.max
            val s = max(0, a - context)
            val b: Int
            val e: Int
            if (total - s <= cap) {
                b = total
                e = total
            } else {
                e = s + cap
                b = e - context
            }
            val size = sizes.fit(e - s)
            val window = FloatArray(size * nMels)
            System.arraycopy(mel, s * nMels, window, 0, (e - s) * nMels)
            val out = vocoder.run(sizes.name(size), mapOf("mel" to window, "mask" to lengthMask(e - s, size)), VOCODER_OUTPUTS)
            val re = out.getValue("re")
            val im = out.getValue("im")
            for (t in a until b) istft.push(re, im, (t - s) * nBins, 1)
            if (b == total) decimator?.finish()
            if (!flush()) return false
            a = b
        }
        return true
    }

    private class SampleBuffer {
        var data = FloatArray(32768)
        var size = 0

        fun append(x: FloatArray, n: Int) {
            if (size + n > data.size) data = data.copyOf(maxOf(size + n, data.size * 2))
            System.arraycopy(x, 0, data, size, n)
            size += n
        }
    }

    override fun close() {
        for (n in networks) runCatching { n.close() }
        networks.clear()
    }

    companion object {
        /** Sentences longer than this are split at commas etc. (at normal speed). */
        const val MAX_CHUNK_CHARS = 250
        const val MIN_CHUNK_CHARS = 60
        private val TEXT_OUTPUTS = listOf("log_dur", "hidden")
        private val DECODER_OUTPUTS = listOf("mel")
        private val VOCODER_OUTPUTS = listOf("re", "im")
    }
}
