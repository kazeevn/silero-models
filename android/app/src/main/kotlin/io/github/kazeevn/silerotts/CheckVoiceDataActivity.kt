package io.github.kazeevn.silerotts

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.speech.tts.TextToSpeech

/** Answers the framework's CHECK_TTS_DATA query: the models are bundled, Russian is always installed. */
class CheckVoiceDataActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val data = Intent()
            .putStringArrayListExtra(TextToSpeech.Engine.EXTRA_AVAILABLE_VOICES, arrayListOf("rus-RUS"))
            .putStringArrayListExtra(TextToSpeech.Engine.EXTRA_UNAVAILABLE_VOICES, arrayListOf())
        setResult(TextToSpeech.Engine.CHECK_VOICE_DATA_PASS, data)
        finish()
    }
}
