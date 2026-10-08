package io.github.kazeevn.silerotts

import android.app.Activity
import android.content.Intent
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import android.view.View
import android.view.WindowInsets
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.Spinner
import android.widget.Switch
import android.widget.TextView
import io.github.kazeevn.silerotts.core.AudioSink
import io.github.kazeevn.silerotts.core.TextNormalizer
import kotlin.concurrent.thread

/** Engine settings, a test field and an on-device benchmark. */
class SettingsActivity : Activity() {
    @Volatile private var stopRequested = false
    private lateinit var status: TextView
    private lateinit var result: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_settings)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            // edge-to-edge is enforced for targetSdk 35+ on Android 15+
            findViewById<View>(R.id.root).setOnApplyWindowInsetsListener { v, insets ->
                val bars = insets.getInsets(WindowInsets.Type.systemBars() or WindowInsets.Type.ime())
                v.setPadding(bars.left, bars.top, bars.right, bars.bottom)
                insets
            }
        }
        status = findViewById(R.id.status)
        result = findViewById(R.id.result)
        val prefs = EngineHolder.prefs(this)

        val speakers = SileroTtsService.SPEAKERS
        setupSpinner(R.id.speaker, speakers, speakers.indexOf(prefs.getString(Prefs.SPEAKER, Prefs.DEFAULT_SPEAKER)).coerceAtLeast(0)) {
            prefs.edit().putString(Prefs.SPEAKER, speakers[it]).apply()
        }
        val rates = Prefs.SAMPLE_RATES
        setupSpinner(R.id.sample_rate, rates.map { "${it / 1000} kHz" + if (it == 48000) " (native)" else "" },
            rates.indexOf(prefs.getInt(Prefs.SAMPLE_RATE, Prefs.DEFAULT_SAMPLE_RATE)).coerceAtLeast(0)) {
            prefs.edit().putInt(Prefs.SAMPLE_RATE, rates[it]).apply()
        }
        val threadOptions = listOf(0) + (1..Runtime.getRuntime().availableProcessors())
        setupSpinner(R.id.threads, threadOptions.map { if (it == 0) getString(R.string.threads_auto, EngineHolder.performanceCores()) else "$it" },
            threadOptions.indexOf(prefs.getInt(Prefs.THREADS, 0)).coerceAtLeast(0)) {
            if (prefs.getInt(Prefs.THREADS, 0) != threadOptions[it]) {
                prefs.edit().putInt(Prefs.THREADS, threadOptions[it]).apply()
                watchLoading()
            }
        }
        for ((id, key) in listOf(R.id.stress to Prefs.STRESS, R.id.homographs to Prefs.HOMOGRAPHS, R.id.yo to Prefs.YO)) {
            findViewById<Switch>(id).apply {
                isChecked = prefs.getBoolean(key, true)
                setOnCheckedChangeListener { _, checked -> prefs.edit().putBoolean(key, checked).apply() }
            }
        }
        findViewById<Button>(R.id.speak).setOnClickListener { speak(findViewById<EditText>(R.id.text).text.toString()) }
        findViewById<Button>(R.id.stop).setOnClickListener { stopRequested = true }
        findViewById<Button>(R.id.benchmark).setOnClickListener { benchmark() }
        findViewById<Button>(R.id.tts_settings).setOnClickListener {
            runCatching { startActivity(Intent("com.android.settings.TTS_SETTINGS")) }
        }
        watchLoading()
    }

    private fun setupSpinner(id: Int, items: List<String>, selected: Int, onSelect: (Int) -> Unit) {
        findViewById<Spinner>(id).apply {
            adapter = ArrayAdapter(this@SettingsActivity, android.R.layout.simple_spinner_dropdown_item, items)
            setSelection(selected, false)
            onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, rowId: Long) = onSelect(position)
                override fun onNothingSelected(parent: AdapterView<*>?) {}
            }
        }
    }

    private fun watchLoading() {
        status.setText(R.string.status_loading)
        val future = EngineHolder.load(this)
        thread(name = "silero-status") {
            val text = try {
                future.get()
                getString(R.string.status_ready, EngineHolder.loadTimeMs, EngineHolder.currentThreads())
            } catch (e: Exception) {
                getString(R.string.status_error, (e.cause ?: e).toString())
            }
            runOnUiThread { status.text = text }
        }
    }

    private fun speak(text: String) {
        stopRequested = false
        thread(name = "silero-speak") {
            val engine = runCatching { EngineHolder.engine(this) }.getOrElse { return@thread }
            val params = Prefs.params(this)
            val track = AudioTrack.Builder()
                .setAudioAttributes(AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA).setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
                .setAudioFormat(AudioFormat.Builder().setEncoding(AudioFormat.ENCODING_PCM_FLOAT).setSampleRate(params.sampleRate).setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build())
                .setTransferMode(AudioTrack.MODE_STREAM)
                .setBufferSizeInBytes(params.sampleRate * 4 / 2)
                .build()
            try {
                track.play()
                synchronized(EngineHolder.lock) {
                    engine.synthesize(text, params, AudioSink { samples, n ->
                        !stopRequested && track.write(samples, 0, n, AudioTrack.WRITE_BLOCKING) >= 0 && !stopRequested
                    })
                }
                if (!stopRequested) {
                    // let the buffered tail play out
                    SystemClock.sleep(500)
                }
            } finally {
                track.stop()
                track.release()
            }
        }
    }

    private fun benchmark() {
        result.setText(R.string.benchmark_running)
        val text = getString(R.string.sample_text) + " Съешь же ещё этих мягких французских булок, да выпей чаю. " +
            "В недрах тундры выдры в гетрах тырят в вёдра ядра кедров."
        thread(name = "silero-bench") {
            val report = try {
                val engine = EngineHolder.engine(this)
                val params = Prefs.params(this)
                val chunks = TextNormalizer.chunks(TextNormalizer.normalize(text))
                val lines = StringBuilder()
                synchronized(EngineHolder.lock) {
                    // warm-up (first runs allocate ORT buffers)
                    engine.synthesizeChunk(chunks[0].text, params, { _, _ -> true }, chunks[0].sentence)
                    var audio = 0.0
                    var frontend = 0.0
                    var predictors = 0.0
                    var acoustic = 0.0
                    var vocoder = 0.0
                    var firstAudio = 0.0
                    val t0 = SystemClock.elapsedRealtimeNanos()
                    for ((i, c) in chunks.withIndex()) {
                        engine.synthesizeChunk(c.text, params, { _, _ -> true }, c.sentence)
                        val s = engine.lastStats
                        if (i == 0) firstAudio = s.firstAudioMs
                        audio += s.audioSeconds
                        frontend += s.frontendMs
                        predictors += s.predictorsMs
                        acoustic += s.acousticMs
                        vocoder += s.vocoderMs
                    }
                    val total = (SystemClock.elapsedRealtimeNanos() - t0) / 1e6
                    lines.append("threads        ${EngineHolder.currentThreads()}\n")
                    lines.append("model load     ${EngineHolder.loadTimeMs} ms\n")
                    lines.append("first audio    %.0f ms\n".format(firstAudio))
                    lines.append("audio          %.2f s @ %d Hz\n".format(audio, params.sampleRate))
                    lines.append("synthesis      %.0f ms\n".format(total))
                    lines.append("real-time      x%.1f (RTF %.3f)\n".format(audio * 1000 / total, total / 1000 / audio))
                    lines.append("  frontend     %.0f ms\n  predictors   %.0f ms\n  acoustic     %.0f ms\n  vocoder      %.0f ms\n"
                        .format(frontend, predictors, acoustic, vocoder))
                }
                lines.toString()
            } catch (e: Exception) {
                e.toString()
            }
            runOnUiThread { result.text = report }
        }
    }

    override fun onDestroy() {
        stopRequested = true
        super.onDestroy()
    }
}
