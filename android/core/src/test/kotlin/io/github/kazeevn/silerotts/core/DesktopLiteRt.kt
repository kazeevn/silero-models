package io.github.kazeevn.silerotts.core

import com.sun.jna.Library
import com.sun.jna.Memory
import com.sun.jna.Native
import com.sun.jna.Pointer
import com.sun.jna.ptr.LongByReference
import com.sun.jna.ptr.PointerByReference
import java.io.Closeable
import java.io.File

/**
 * Runs the exported .tflite networks on the desktop through the LiteRT C API
 * (`libLiteRt.so` of the ai-edge-litert Python wheel): the same CompiledModel
 * runtime and XNNPACK kernels as the Android app, so the JVM tests exercise the
 * engine with the real model files.
 */
class DesktopLiteRt(library: File, private val modelDir: File) : NetworkLoader, Closeable {
    @Suppress("FunctionName")
    private interface C : Library {
        fun LiteRtGetStatusString(status: Int): String
        fun LiteRtCreateEnvironment(numOptions: Int, options: Pointer?, env: PointerByReference): Int
        fun LiteRtDestroyEnvironment(env: Pointer)
        fun LiteRtCreateModelFromFile(env: Pointer, filename: String, model: PointerByReference): Int
        fun LiteRtDestroyModel(model: Pointer)
        fun LiteRtCreateOptions(options: PointerByReference): Int
        fun LiteRtSetOptionsHardwareAccelerators(options: Pointer, accelerators: Int): Int
        fun LiteRtDestroyOptions(options: Pointer)
        fun LiteRtCreateCompiledModel(env: Pointer, model: Pointer, options: Pointer, compiled: PointerByReference): Int
        fun LiteRtDestroyCompiledModel(compiled: Pointer)
        fun LiteRtGetNumModelSignatures(model: Pointer, n: LongByReference): Int
        fun LiteRtGetModelSignature(model: Pointer, index: Long, signature: PointerByReference): Int
        fun LiteRtGetSignatureKey(signature: Pointer, key: PointerByReference): Int
        fun LiteRtGetNumSignatureInputs(signature: Pointer, n: LongByReference): Int
        fun LiteRtGetSignatureInputName(signature: Pointer, index: Long, name: PointerByReference): Int
        fun LiteRtGetSignatureInputTensor(signature: Pointer, name: String, tensor: PointerByReference): Int
        fun LiteRtGetNumSignatureOutputs(signature: Pointer, n: LongByReference): Int
        fun LiteRtGetSignatureOutputName(signature: Pointer, index: Long, name: PointerByReference): Int
        fun LiteRtGetSignatureOutputTensor(signature: Pointer, name: String, tensor: PointerByReference): Int
        fun LiteRtGetRankedTensorType(tensor: Pointer, type: Pointer): Int
        fun LiteRtGetCompiledModelInputBufferRequirements(compiled: Pointer, signature: Long, input: Long, requirements: PointerByReference): Int
        fun LiteRtGetCompiledModelOutputBufferRequirements(compiled: Pointer, signature: Long, output: Long, requirements: PointerByReference): Int
        fun LiteRtCreateManagedTensorBufferFromRequirements(env: Pointer, type: Pointer, requirements: Pointer, buffer: PointerByReference): Int
        fun LiteRtLockTensorBuffer(buffer: Pointer, address: PointerByReference, mode: Int): Int
        fun LiteRtUnlockTensorBuffer(buffer: Pointer): Int
        fun LiteRtDestroyTensorBuffer(buffer: Pointer)
        fun LiteRtRunCompiledModel(compiled: Pointer, signature: Long, numInputs: Long, inputs: Pointer, numOutputs: Long, outputs: Pointer): Int
    }

    private val c: C = Native.load(library.absolutePath, C::class.java)
    private val env: Pointer = ref { c.LiteRtCreateEnvironment(0, null, it) }

    private fun check(status: Int) {
        if (status != 0) throw IllegalStateException("LiteRT: ${c.LiteRtGetStatusString(status)}")
    }

    private fun ref(call: (PointerByReference) -> Int): Pointer {
        val r = PointerByReference()
        check(call(r))
        return r.value
    }

    private fun count(call: (LongByReference) -> Int): Long {
        val r = LongByReference()
        check(call(r))
        return r.value
    }

    override fun load(name: String): Network = DesktopNetwork(File(modelDir, "$name.tflite"))

    override fun close() = c.LiteRtDestroyEnvironment(env)

    private class Tensor(val name: String, val buffer: Pointer, val elements: Int, val isFloat: Boolean)

    private class Signature(val index: Long, val inputs: List<Tensor>, val outputs: List<Tensor>) {
        val inputArray = pointers(inputs)
        val outputArray = pointers(outputs)

        private fun pointers(tensors: List<Tensor>) = Memory(8L * tensors.size).also { m ->
            tensors.forEachIndexed { i, t -> m.setPointer(8L * i, t.buffer) }
        }
    }

    private inner class DesktopNetwork(file: File) : Network {
        private val model = ref { c.LiteRtCreateModelFromFile(env, file.path, it) }
        private val compiled: Pointer
        private val signatures = HashMap<String, Signature>()

        init {
            val options = ref { c.LiteRtCreateOptions(it) }
            try {
                check(c.LiteRtSetOptionsHardwareAccelerators(options, ACCELERATOR_CPU))
                compiled = ref { c.LiteRtCreateCompiledModel(env, model, options, it) }
            } finally {
                c.LiteRtDestroyOptions(options)
            }
            for (i in 0 until count { c.LiteRtGetNumModelSignatures(model, it) }) {
                val sig = ref { c.LiteRtGetModelSignature(model, i, it) }
                val key = ref { c.LiteRtGetSignatureKey(sig, it) }.getString(0)
                val inputs = (0 until count { c.LiteRtGetNumSignatureInputs(sig, it) }).map { k ->
                    val name = ref { c.LiteRtGetSignatureInputName(sig, k, it) }.getString(0)
                    tensor(name, ref { c.LiteRtGetSignatureInputTensor(sig, name, it) },
                        ref { c.LiteRtGetCompiledModelInputBufferRequirements(compiled, i, k, it) })
                }
                val outputs = (0 until count { c.LiteRtGetNumSignatureOutputs(sig, it) }).map { k ->
                    val name = ref { c.LiteRtGetSignatureOutputName(sig, k, it) }.getString(0)
                    tensor(name, ref { c.LiteRtGetSignatureOutputTensor(sig, name, it) },
                        ref { c.LiteRtGetCompiledModelOutputBufferRequirements(compiled, i, k, it) })
                }
                signatures[key] = Signature(i, inputs, outputs)
            }
        }

        private fun tensor(name: String, tensor: Pointer, requirements: Pointer): Tensor {
            // LiteRtRankedTensorType: element type, then LiteRtLayout (rank:7
            // bit-field, int32 dimensions[8], uint32 strides[8])
            val type = Memory(72)
            check(c.LiteRtGetRankedTensorType(tensor, type))
            val rank = type.getInt(4) and 0x7f
            var elements = 1
            for (d in 0 until rank) elements *= type.getInt(8L + 4 * d)
            val elementType = type.getInt(0)
            require(elementType == FLOAT32 || elementType == INT32) { "$name: unsupported element type $elementType" }
            val buffer = ref { c.LiteRtCreateManagedTensorBufferFromRequirements(env, type, requirements, it) }
            return Tensor(name, buffer, elements, elementType == FLOAT32)
        }

        override fun run(signature: String, inputs: Map<String, Any>, outputs: List<String>): Map<String, FloatArray> {
            val sig = signatures[signature] ?: throw IllegalArgumentException("no signature $signature")
            for (t in sig.inputs) {
                val value = inputs[t.name] ?: throw IllegalArgumentException("$signature: missing input ${t.name}")
                val address = ref { c.LiteRtLockTensorBuffer(t.buffer, it, LOCK_WRITE) }
                try {
                    when (value) {
                        is FloatArray -> {
                            require(t.isFloat && value.size == t.elements) { "$signature/${t.name}: ${value.size} floats for ${t.elements}" }
                            address.write(0, value, 0, value.size)
                        }
                        is IntArray -> {
                            require(!t.isFloat && value.size == t.elements) { "$signature/${t.name}: ${value.size} ints for ${t.elements}" }
                            address.write(0, value, 0, value.size)
                        }
                        else -> throw IllegalArgumentException("${t.name}: unsupported input type")
                    }
                } finally {
                    check(c.LiteRtUnlockTensorBuffer(t.buffer))
                }
            }
            check(c.LiteRtRunCompiledModel(compiled, sig.index, sig.inputs.size.toLong(), sig.inputArray,
                sig.outputs.size.toLong(), sig.outputArray))
            return outputs.associateWith { name ->
                val t = sig.outputs.firstOrNull { it.name == name } ?: throw IllegalArgumentException("$signature: no output $name")
                val address = ref { c.LiteRtLockTensorBuffer(t.buffer, it, LOCK_READ) }
                try {
                    address.getFloatArray(0, t.elements)
                } finally {
                    check(c.LiteRtUnlockTensorBuffer(t.buffer))
                }
            }
        }

        override fun close() {
            for (s in signatures.values) for (t in s.inputs + s.outputs) c.LiteRtDestroyTensorBuffer(t.buffer)
            signatures.clear()
            c.LiteRtDestroyCompiledModel(compiled)
            c.LiteRtDestroyModel(model)
        }
    }

    companion object {
        private const val ACCELERATOR_CPU = 1
        private const val FLOAT32 = 1
        private const val INT32 = 2
        private const val LOCK_READ = 0
        private const val LOCK_WRITE = 1

        /** Engine on the assets in `-Psilero.assets` with `-Psilero.litert`, or null (tests are skipped). */
        fun engineOrNull(): Pair<DesktopLiteRt, SileroEngine>? {
            val assets = File(System.getProperty("silero.assets") ?: return null)
            val lib = File(System.getProperty("silero.litert") ?: return null)
            if (!File(assets, "config.json").exists() || !lib.exists()) return null
            val runtime = DesktopLiteRt(lib, assets)
            return runtime to SileroEngine(DirectoryAssetSource(assets), runtime)
        }
    }
}
