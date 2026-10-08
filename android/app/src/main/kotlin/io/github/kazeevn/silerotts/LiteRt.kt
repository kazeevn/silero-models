package io.github.kazeevn.silerotts

import android.content.Context
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.LiteRtException
import com.google.ai.edge.litert.TensorBuffer
import io.github.kazeevn.silerotts.core.Network
import io.github.kazeevn.silerotts.core.NetworkLoader
import java.io.File

/** A network on the LiteRT CompiledModel API; the tensor buffers of each signature are allocated once. */
class LiteRtNetwork(private val model: CompiledModel) : Network {
    private class Buffers(val inputs: Map<String, TensorBuffer>, val outputs: Map<String, TensorBuffer>)

    private val buffers = HashMap<String, Buffers>()

    override fun run(signature: String, inputs: Map<String, Any>, outputs: List<String>): Map<String, FloatArray> {
        val b = buffers.getOrPut(signature) {
            Buffers(
                inputs.keys.associateWith { model.createInputBuffer(it, signature) },
                outputs.associateWith { model.createOutputBuffer(it, signature) },
            )
        }
        for ((name, value) in inputs) {
            val buffer = b.inputs[name] ?: throw IllegalArgumentException("$signature: unexpected input $name")
            when (value) {
                is FloatArray -> buffer.writeFloat(value)
                is IntArray -> buffer.writeInt(value)
                else -> throw IllegalArgumentException("$signature: unsupported type of input $name")
            }
        }
        model.run(b.inputs, b.outputs, signature)
        return outputs.associateWith { b.outputs.getValue(it).readFloat() }
    }

    override fun close() {
        for (b in buffers.values) (b.inputs.values + b.outputs.values).forEach { it.close() }
        buffers.clear()
        model.close()
    }
}

/**
 * Loads the networks with the LiteRT CompiledModel API on the CPU (XNNPACK).
 *
 * The .tflite files are copied out of the APK once per app version and loaded
 * by path, so that LiteRT memory-maps them (loading from the AssetManager
 * copies the whole model to the native heap). XNNPACK keeps its packed weights
 * in a file-backed cache next to them: all signatures of a network share one
 * memory-mapped copy (otherwise every signature packs its own, several times
 * the memory), and later starts skip the packing.
 */
class LiteRtLoader(private val context: Context, private val threads: Int) : NetworkLoader {
    private val dir = modelDir(context)

    override fun load(name: String): Network {
        val model = File(dir, "$name.tflite")
        if (!model.exists()) {
            val tmp = File(dir, "$name.tflite.tmp")
            context.assets.open("$ASSET_DIR/$name.tflite").use { input -> tmp.outputStream().use { input.copyTo(it, 1 shl 16) } }
            check(tmp.renameTo(model)) { "cannot write $model" }
        }
        val cache = File(dir, "$name.xnnpack")
        val options = CompiledModel.Options(Accelerator.CPU).apply {
            cpuOptions = CompiledModel.CpuOptions(numThreads = threads, xnnPackWeightCachePath = cache.path)
        }
        val compiled = try {
            CompiledModel.create(model.path, options, environment)
        } catch (e: LiteRtException) {
            // e.g. a weight cache truncated by a crash while it was written
            Log.w(TAG, "$name: rebuilding the XNNPACK weight cache", e)
            cache.delete()
            CompiledModel.create(model.path, options, environment)
        }
        return LiteRtNetwork(compiled)
    }

    companion object {
        private const val TAG = "SileroTts"
        const val ASSET_DIR = "silero"

        private val environment: Environment by lazy { Environment.create() }

        /** Per-version directory for the extracted models and weight caches; older ones are deleted. */
        private fun modelDir(context: Context): File {
            val info = context.packageManager.getPackageInfo(context.packageName, 0)
            val root = File(context.noBackupFilesDir, "litert")
            val dir = File(root, "${info.longVersionCode}-${info.lastUpdateTime}")
            root.listFiles()?.filter { it != dir }?.forEach { it.deleteRecursively() }
            check(dir.isDirectory || dir.mkdirs()) { "cannot create $dir" }
            return dir
        }
    }
}
