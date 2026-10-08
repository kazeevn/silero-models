package io.github.kazeevn.silerotts

import android.content.Context
import android.content.SharedPreferences
import android.content.res.AssetManager
import android.os.SystemClock
import android.util.Log
import io.github.kazeevn.silerotts.core.AssetSource
import io.github.kazeevn.silerotts.core.SileroEngine
import io.github.kazeevn.silerotts.core.SynthesisParams
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.Executors
import java.util.concurrent.Future

/** Memory-maps uncompressed assets (see noCompress in build.gradle.kts) in place. */
class ApkAssetSource(private val assets: AssetManager, private val dir: String) : AssetSource {
    override fun map(name: String): ByteBuffer {
        assets.openFd("$dir/$name").use { fd ->
            FileInputStream(fd.fileDescriptor).use { s ->
                return s.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
            }
        }
    }
}

/** Process-wide engine, loaded once in the background and shared by the service and the UI. */
object EngineHolder {
    private const val TAG = "SileroTts"
    private val loader = Executors.newSingleThreadExecutor { r -> Thread(r, "silero-loader") }
    @Volatile private var future: Future<SileroEngine>? = null
    @Volatile private var loadedThreads = 0
    @Volatile var loadTimeMs: Long = 0
        private set

    /** Synthesis is single-threaded per engine; the service and the settings screen take turns. */
    val lock = Any()

    fun prefs(context: Context): SharedPreferences =
        context.applicationContext.getSharedPreferences("settings", Context.MODE_PRIVATE)

    @Synchronized
    fun load(context: Context): Future<SileroEngine> {
        val threads = threads(context)
        future?.let { f ->
            if (loadedThreads == threads && !(f.isDone && runCatching { f.get() }.isFailure)) return f
            // thread count changed or the last attempt failed: reload
            loader.submit { synchronized(lock) { runCatching { f.get().close() } } }
        }
        val app = context.applicationContext
        loadedThreads = threads
        return loader.submit<SileroEngine> {
            val t0 = SystemClock.elapsedRealtime()
            val engine = SileroEngine(ApkAssetSource(app.assets, LiteRtLoader.ASSET_DIR), LiteRtLoader(app, threads))
            loadTimeMs = SystemClock.elapsedRealtime() - t0
            // allocate the buffers of the small signatures used by the first
            // sentence, so the first request does not pay for it
            try {
                synchronized(lock) { engine.synthesizeChunk(WARM_UP, SynthesisParams(), { _, _ -> true }) }
            } catch (e: Throwable) {
                engine.close()
                throw e
            }
            Log.i(TAG, "models loaded in $loadTimeMs ms (+ warm-up ${SystemClock.elapsedRealtime() - t0 - loadTimeMs} ms) with $threads threads")
            engine
        }.also { future = it }
    }

    fun engine(context: Context): SileroEngine = load(context).get()

    fun currentThreads() = loadedThreads

    private const val WARM_UP = "Привет."

    /** Thread count from the settings, "auto" = number of non-LITTLE cores. */
    fun threads(context: Context): Int {
        val v = prefs(context).getInt(Prefs.THREADS, 0)
        return if (v > 0) v else performanceCores()
    }

    /**
     * Cores whose max frequency is above the slowest cluster's (on the Pixel 8a's
     * Tensor G3: 4x Cortex-A715 + 1x Cortex-X3; the 4 Cortex-A510 are skipped).
     */
    fun performanceCores(): Int {
        val freqs = (0 until Runtime.getRuntime().availableProcessors()).mapNotNull { cpu ->
            runCatching { File("/sys/devices/system/cpu/cpu$cpu/cpufreq/cpuinfo_max_freq").readText().trim().toLong() }.getOrNull()
        }
        val n = if (freqs.size < 2 || freqs.toSet().size < 2) {
            Runtime.getRuntime().availableProcessors() / 2
        } else {
            val slowest = freqs.min()
            freqs.count { it > slowest }
        }
        return n.coerceIn(1, 6)
    }
}

object Prefs {
    const val SPEAKER = "speaker"
    const val SAMPLE_RATE = "sample_rate"
    const val THREADS = "threads"
    const val STRESS = "stress"
    const val HOMOGRAPHS = "homographs"
    const val YO = "yo"

    const val DEFAULT_SPEAKER = "xenia"
    const val DEFAULT_SAMPLE_RATE = 24000
    val SAMPLE_RATES = listOf(24000, 48000, 8000)

    fun params(context: Context, speaker: String? = null, rate: Float = 1f, pitch: Float = 1f): SynthesisParams {
        val p = EngineHolder.prefs(context)
        val homographs = p.getBoolean(HOMOGRAPHS, true)
        return SynthesisParams(
            speaker = speaker ?: p.getString(SPEAKER, DEFAULT_SPEAKER)!!,
            sampleRate = p.getInt(SAMPLE_RATE, DEFAULT_SAMPLE_RATE),
            rate = rate,
            pitch = pitch,
            putStress = p.getBoolean(STRESS, true),
            putStressHomo = homographs,
            putYo = p.getBoolean(YO, true),
            putYoHomo = homographs,
        )
    }
}
