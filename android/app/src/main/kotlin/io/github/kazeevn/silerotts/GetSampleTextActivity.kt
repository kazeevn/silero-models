package io.github.kazeevn.silerotts

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.speech.tts.TextToSpeech

/** Sample sentence played by the system "Listen to an example" button. */
class GetSampleTextActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setResult(TextToSpeech.LANG_AVAILABLE, Intent().putExtra("sampleText", getString(R.string.sample_text)))
        finish()
    }
}
