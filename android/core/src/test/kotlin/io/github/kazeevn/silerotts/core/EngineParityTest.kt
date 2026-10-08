package io.github.kazeevn.silerotts.core

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import kotlin.math.log10
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * End-to-end parity with tools/reference_pipeline.py (which is itself verified
 * against the original torch package). Needs the exported assets and the
 * vectors written by `reference_pipeline.py --vectors`; skipped otherwise.
 */
class EngineParityTest {
    private val assetsDir = File(System.getProperty("silero.assets") ?: "")
    private val vectors = System.getProperty("silero.vectors")?.let { File(it) }

    private fun engine(): SileroEngine {
        val env = OrtEnvironment.getEnvironment()
        return SileroEngine(DirectoryAssetSource(assetsDir), env) {
            OrtSession.SessionOptions().apply { setIntraOpNumThreads(4) }
        }
    }

    @Test
    fun matchesReferencePipeline() {
        if (!File(assetsDir, "config.json").exists() || vectors == null || !vectors.exists()) {
            println("SKIPPED: set -Psilero.assets and -Psilero.vectors")
            return
        }
        @Suppress("UNCHECKED_CAST")
        val cases = Json.parse(vectors.readText()) as List<Map<String, Any?>>
        engine().use { engine ->
            var worstSnr = Double.MAX_VALUE
            for ((i, c) in cases.withIndex()) {
                val text = c["text"] as String
                val p = SynthesisParams(
                    speaker = c["speaker"] as String,
                    sampleRate = (c["sample_rate"] as Number).toInt(),
                    rate = (c["rate"] as Number).toFloat(),
                    pitch = (c["pitch"] as Number).toFloat(),
                )
                val clean = engine.clean(text)
                val accented = engine.accentuate(clean, p)
                assertEquals(c["accented"], accented, "accented text of case $i")
                val tokens = engine.tokens(accented)
                assertEquals((c["tokens"] as List<*>).map { (it as Number).toLong() }, tokens.toList(), "tokens of case $i")

                val audio = ArrayList<Float>()
                engine.synthesizeChunk(text, p, { buf, n -> for (k in 0 until n) audio.add(buf[k]); true })
                val ref = (c["audio"] as List<*>).map { (it as Number).toFloat() }
                assertEquals(ref.size, audio.size, "audio length of case $i")
                var num = 0.0
                var den = 0.0
                var maxErr = 0f
                for (k in ref.indices) {
                    val d = (audio[k] - ref[k]).toDouble()
                    num += ref[k].toDouble() * ref[k]
                    den += d * d
                    maxErr = maxOf(maxErr, kotlin.math.abs(audio[k] - ref[k]))
                }
                val snr = 10 * log10(num / maxOf(den, 1e-20))
                worstSnr = minOf(worstSnr, snr)
                val s = engine.lastStats
                println(
                    "case %2d %-8s snr %.1f dB  max err %.2e | frontend %.0f ms, predictors %.0f ms, acoustic %.0f ms, vocoder %.0f ms, first audio %.0f ms, audio %.2f s"
                        .format(i, p.speaker, snr, maxErr, s.frontendMs, s.predictorsMs, s.acousticMs, s.vocoderMs, s.firstAudioMs, s.audioSeconds)
                )
            }
            assertTrue(worstSnr > 50, "worst SNR $worstSnr dB")
        }
    }
}
