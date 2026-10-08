package io.github.kazeevn.silerotts.core

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.Closeable
import java.nio.FloatBuffer
import java.nio.LongBuffer
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

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
    var predictorsMs: Double = 0.0,
    var acousticMs: Double = 0.0,
    var vocoderMs: Double = 0.0,
    var firstAudioMs: Double = 0.0,
    var audioSeconds: Double = 0.0,
)

/**
 * On-device Silero v5_5_ru engine: ONNX Runtime for the networks, everything
 * else ported from the original Python package (see tools/reference_pipeline.py,
 * which this class mirrors step by step).
 */
class SileroEngine(
    assets: AssetSource,
    private val env: OrtEnvironment,
    sessionOptions: () -> OrtSession.SessionOptions,
) : Closeable {
    val config = ModelConfig.load(assets)
    val speakers: List<String> get() = config.speakers.keys.toList()

    private val predictors: OrtSession
    private val acoustic: OrtSession
    private val vocoder: OrtSession
    private val homo: HomoSolver
    private val accentor: Accentor
    val lastStats = ChunkStats()

    init {
        fun session(name: String) = sessionOptions().use { so -> env.createSession(assets.map("$name.onnx"), so) }
        predictors = session("predictors")
        acoustic = session("acoustic")
        vocoder = session("vocoder")
        val vocab = StringHashTable(assets.map("bert_vocab.bin"))
        homo = HomoSolver(env, session("homosolver"), BertTokenizer(vocab, config.bert), assets.text("homodict.tsv"), config.bert)
        accentor = Accentor(assets.map("accentor.bin"), StringHashTable(assets.map("ngrams.bin")), assets.text("exceptions.tsv"), config.ngramMaxLen)
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

    internal fun tokens(accented: String): LongArray {
        val seq = "|$accented~"
        return LongArray(seq.length) { config.symbolToId.getValue(seq[it]).toLong() }
    }

    /** Duration post-processing of MultiTTSModel.forward (log-durations -> frames). */
    internal fun durations(logDur: FloatArray, rate: Float): IntArray {
        val n = logDur.size
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

    /** update_pitch_coef with a constant coefficient. */
    internal fun applyPitch(pitch: FloatArray, coef: Float, speakerId: Int) {
        val c = if (coef == 0f) 1f else coef
        val shift = (c - 1f) * config.meanStdCoef[speakerId]
        for (i in pitch.indices) {
            var p = pitch[i]
            if (abs(p) < 0.001f) p = 0f
            val pc = p * coef
            pitch[i] = if (pc == 0f) 0f else pc + shift
        }
    }

    /**
     * Full text-to-speech: normalization, sentence chunking and synthesis of
     * every chunk, streamed to [sink]. Returns false if the sink aborted.
     */
    fun synthesize(text: String, p: SynthesisParams, sink: AudioSink): Boolean {
        val chunks = TextNormalizer.chunks(TextNormalizer.normalize(text))
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
        val typeIds = SentenceType.typeIds(typeText, tokens.size)
        val t1 = System.nanoTime()
        stats.frontendMs = (t1 - t0) / 1e6

        val L = tokens.size.toLong()
        val (logDur, pitch) = OnnxTensor.createTensor(env, LongBuffer.wrap(tokens), longArrayOf(1, L)).use { tTok ->
            OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(spk.toLong())), longArrayOf(1)).use { tSpk ->
                OnnxTensor.createTensor(env, LongBuffer.wrap(typeIds), longArrayOf(1, L)).use { tType ->
                    predictors.run(mapOf("tokens" to tTok, "speaker" to tSpk, "type_ids" to tType)).use { r ->
                        (r[0] as OnnxTensor).floats() to (r[1] as OnnxTensor).floats()
                    }
                }
            }
        }
        val durs = durations(logDur, p.rate.coerceIn(0.2f, 4f))
        applyPitch(pitch, p.pitch.coerceIn(0f, 2.5f), spk)
        val total = durs.sum()
        val frameIdx = LongArray(total)
        var f = 0
        for ((i, d) in durs.withIndex()) repeat(d) { frameIdx[f++] = i.toLong() }
        val t2 = System.nanoTime()
        stats.predictorsMs = (t2 - t1) / 1e6

        val mel: FloatArray = OnnxTensor.createTensor(env, LongBuffer.wrap(tokens), longArrayOf(1, L)).use { tTok ->
            OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(spk.toLong())), longArrayOf(1)).use { tSpk ->
                OnnxTensor.createTensor(env, FloatBuffer.wrap(pitch), longArrayOf(1, L)).use { tPitch ->
                    OnnxTensor.createTensor(env, LongBuffer.wrap(frameIdx), longArrayOf(total.toLong())).use { tIdx ->
                        acoustic.run(mapOf("tokens" to tTok, "speaker" to tSpk, "pitch" to tPitch, "frame_idx" to tIdx)).use { r ->
                            (r[0] as OnnxTensor).floats()
                        }
                    }
                }
            }
        }
        val t3 = System.nanoTime()
        stats.acousticMs = (t3 - t2) / 1e6
        stats.audioSeconds = total * config.hopLength / 48000.0
        val ok = vocode(mel, total, p.sampleRate, sink, t0)
        stats.vocoderMs = (System.nanoTime() - t3) / 1e6
        return ok
    }

    private fun OnnxTensor.floats(): FloatArray {
        val fb = floatBuffer
        return FloatArray(fb.remaining()).also { fb.get(it) }
    }

    /**
     * Runs the vocoder over the mel spectrogram in windows so that audio can be
     * played before the whole sentence is vocoded. Every window gets
     * [CONTEXT] frames of context on both sides, which covers the receptive
     * field of the ConvNeXt backbone (9 convolutions with kernel 7 = 27 frames),
     * so the output is identical to a single pass.
     */
    private fun vocode(mel: FloatArray, total: Int, sampleRate: Int, sink: AudioSink, t0: Long): Boolean {
        val nMels = mel.size / total
        val nBins = config.nFft / 2 + 1
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
            val w = if (a == 0) FIRST_WINDOW else WINDOW
            // avoid a tiny last window
            val b = if (total - (a + w) < w / 2) total else a + w
            val s = max(0, a - CONTEXT)
            val e = min(total, b + CONTEXT)
            val len = e - s
            val window = FloatArray(nMels * len)
            for (c in 0 until nMels) System.arraycopy(mel, c * total + s, window, c * len, len)
            val (re, im) = OnnxTensor.createTensor(env, FloatBuffer.wrap(window), longArrayOf(1, nMels.toLong(), len.toLong())).use { tMel ->
                vocoder.run(mapOf("mel" to tMel)).use { r -> (r[0] as OnnxTensor).floats() to (r[1] as OnnxTensor).floats() }
            }
            check(re.size == nBins * len)
            for (t in a until b) istft.push(re, im, t - s, len)
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
        predictors.close()
        acoustic.close()
        vocoder.close()
        homo.close()
    }

    companion object {
        /** Vocoder receptive field is 27 frames on each side. */
        const val CONTEXT = 28
        /** 48 frames = 0.6 s: small first window for low latency... */
        const val FIRST_WINDOW = 48
        /** ...then larger windows to keep the context overhead low (~20%). */
        const val WINDOW = 256
    }
}
