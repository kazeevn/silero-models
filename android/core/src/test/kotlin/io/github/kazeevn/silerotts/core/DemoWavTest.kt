package io.github.kazeevn.silerotts.core

import java.io.DataOutputStream
import java.io.File
import java.io.FileOutputStream
import kotlin.test.Test

/**
 * Writes WAV files through the full engine (normalization + synthesis) for
 * listening tests: ./gradlew :core:test --tests '*DemoWav*' -Psilero.wavOut=DIR
 */
class DemoWavTest {
    private val outDir = System.getProperty("silero.wavOut")?.let { File(it) }

    private fun writeWav(file: File, samples: FloatArray, n: Int, sampleRate: Int) {
        DataOutputStream(FileOutputStream(file).buffered()).use { out ->
            fun le32(v: Int) = out.writeInt(Integer.reverseBytes(v))
            fun le16(v: Int) = out.writeShort(java.lang.Short.reverseBytes(v.toShort()).toInt())
            out.writeBytes("RIFF"); le32(36 + 2 * n); out.writeBytes("WAVE")
            out.writeBytes("fmt "); le32(16); le16(1); le16(1); le32(sampleRate); le32(2 * sampleRate); le16(2); le16(16)
            out.writeBytes("data"); le32(2 * n)
            for (i in 0 until n) le16((samples[i] * 32767f).toInt().coerceIn(-32768, 32767))
        }
    }

    @Test
    fun writeDemoWavs() {
        val loaded = if (outDir == null) null else DesktopLiteRt.engineOrNull()
        if (outDir == null || loaded == null) {
            println("SKIPPED: set -Psilero.wavOut, -Psilero.assets and -Psilero.litert")
            return
        }
        outDir.mkdirs()
        val (runtime, engine) = loaded
        runtime.use {
            engine.use {
                val demos = listOf(
                    "xenia" to "Привет! Это системный синтезатор речи Silero на телефоне. Сегодня 8 октября 2026 года, на улице +15°C, " +
                        "а курс доллара — 81,5 руб. Встреча в 10:30, не опаздывайте!",
                    "aidar" to "Мой замок стоит на горе, а на двери висит замок. Вы уже прочитали все 3 главы?",
                    "baya" to "Скидка 20% действует до 31 декабря. Подробности на сайте https://example.com или по телефону.",
                    "kseniya" to "В 1990-х годах в России было непросто. Кто-то уехал, а кто-то остался.",
                    "eugene" to "Скачайте обновление iOS 17 через Wi-Fi: размер — 2,5 GB. Готово? Тогда нажмите «OK».",
                )
                for ((speaker, text) in demos) {
                    val buf = ArrayList<Float>()
                    val p = SynthesisParams(speaker = speaker, sampleRate = 24000)
                    val t0 = System.nanoTime()
                    engine.synthesize(text, p) { s, n -> for (i in 0 until n) buf.add(s[i]); true }
                    val ms = (System.nanoTime() - t0) / 1e6
                    val arr = buf.toFloatArray()
                    writeWav(File(outDir, "silero_$speaker.wav"), arr, arr.size, p.sampleRate)
                    println("%-8s %.2f s audio in %.0f ms: %s".format(speaker, arr.size / 24000.0, ms, TextNormalizer.normalize(text)))
                }
            }
        }
    }
}
