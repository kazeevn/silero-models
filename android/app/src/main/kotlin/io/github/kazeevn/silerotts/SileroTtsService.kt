package io.github.kazeevn.silerotts

import android.media.AudioFormat
import android.speech.tts.SynthesisCallback
import android.speech.tts.SynthesisRequest
import android.speech.tts.TextToSpeech
import android.speech.tts.TextToSpeechService
import android.speech.tts.Voice
import android.util.Log
import io.github.kazeevn.silerotts.core.AudioSink
import java.util.Locale

/** System text-to-speech engine backed by Silero v5_5_ru. */
class SileroTtsService : TextToSpeechService() {
    @Volatile private var stopped = false

    override fun onCreate() {
        super.onCreate()
        // start loading the models right away so the first request does not wait long
        EngineHolder.load(this)
    }

    private fun isRussian(lang: String?) = lang != null && (lang.equals("rus", true) || lang.equals("ru", true))

    override fun onIsLanguageAvailable(lang: String?, country: String?, variant: String?): Int = when {
        !isRussian(lang) -> TextToSpeech.LANG_NOT_SUPPORTED
        country.isNullOrEmpty() -> TextToSpeech.LANG_AVAILABLE
        country.equals("RUS", true) || country.equals("RU", true) -> TextToSpeech.LANG_COUNTRY_AVAILABLE
        else -> TextToSpeech.LANG_AVAILABLE
    }

    override fun onGetLanguage(): Array<String> = arrayOf("rus", "RUS", "")

    override fun onLoadLanguage(lang: String?, country: String?, variant: String?): Int {
        val r = onIsLanguageAvailable(lang, country, variant)
        if (r >= TextToSpeech.LANG_AVAILABLE) EngineHolder.load(this)
        return r
    }

    override fun onStop() {
        stopped = true
    }

    override fun onGetVoices(): List<Voice> = SPEAKERS.map { voice(it) }

    override fun onIsValidVoiceName(voiceName: String?): Int =
        if (speakerOf(voiceName) != null) TextToSpeech.SUCCESS else TextToSpeech.ERROR

    override fun onLoadVoice(voiceName: String?): Int {
        if (speakerOf(voiceName) == null) return TextToSpeech.ERROR
        EngineHolder.load(this)
        return TextToSpeech.SUCCESS
    }

    override fun onGetDefaultVoiceNameFor(lang: String?, country: String?, variant: String?): String? =
        if (isRussian(lang)) voiceName(EngineHolder.prefs(this).getString(Prefs.SPEAKER, Prefs.DEFAULT_SPEAKER)!!) else null

    override fun onSynthesizeText(request: SynthesisRequest, callback: SynthesisCallback) {
        stopped = false
        val text = request.charSequenceText?.toString() ?: request.text ?: ""
        if (!isRussian(request.language) && request.voiceName.isNullOrEmpty()) {
            callback.error(TextToSpeech.ERROR_INVALID_REQUEST)
            return
        }
        val engine = try {
            EngineHolder.engine(this)
        } catch (e: Exception) {
            Log.e(TAG, "engine failed to load", e)
            callback.error(TextToSpeech.ERROR_SERVICE)
            return
        }
        // speech rate / pitch: 100 = normal
        val params = Prefs.params(
            this,
            speaker = speakerOf(request.voiceName),
            rate = request.speechRate / 100f,
            pitch = request.pitch / 100f,
        )
        if (callback.start(params.sampleRate, AudioFormat.ENCODING_PCM_16BIT, 1) != TextToSpeech.SUCCESS) return
        val maxBytes = callback.maxBufferSize
        var bytes = ByteArray(0)
        val sink = AudioSink { samples, n ->
            if (stopped) return@AudioSink false
            if (bytes.size < 2 * n) bytes = ByteArray(2 * n)
            for (i in 0 until n) {
                val v = (samples[i] * 32767f).toInt().coerceIn(-32768, 32767)
                bytes[2 * i] = v.toByte()
                bytes[2 * i + 1] = (v shr 8).toByte()
            }
            var off = 0
            while (off < 2 * n) {
                val len = minOf(maxBytes, 2 * n - off)
                if (callback.audioAvailable(bytes, off, len) != TextToSpeech.SUCCESS) return@AudioSink false
                off += len
            }
            !stopped
        }
        try {
            synchronized(EngineHolder.lock) { engine.synthesize(text, params, sink) }
            callback.done()
        } catch (e: Exception) {
            Log.e(TAG, "synthesis failed", e)
            callback.error(TextToSpeech.ERROR_SYNTHESIS)
        }
    }

    companion object {
        private const val TAG = "SileroTts"
        val SPEAKERS = listOf("xenia", "baya", "kseniya", "aidar", "eugene")
        private val LOCALE = Locale("ru", "RU")

        fun voiceName(speaker: String) = "ru-RU-x-silero-$speaker"

        fun speakerOf(voiceName: String?): String? =
            voiceName?.removePrefix("ru-RU-x-silero-")?.takeIf { it in SPEAKERS && voiceName.startsWith("ru-RU-x-silero-") }

        fun voice(speaker: String) =
            Voice(voiceName(speaker), LOCALE, Voice.QUALITY_VERY_HIGH, Voice.LATENCY_NORMAL, false, emptySet())
    }
}
