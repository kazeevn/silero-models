package io.github.kazeevn.silerotts.core

import java.io.Closeable

/**
 * A network converted by tools/export_models.py: a LiteRT model whose
 * signatures ("L64", "T384", ...) are the same network at different static
 * input sizes, sharing one copy of the weights. Inputs are zero-padded to the
 * signature size and a mask marks the valid positions.
 *
 * The app runs it with the LiteRT CompiledModel API, the desktop tests with the
 * LiteRT C API, so this module stays a plain JVM library.
 */
interface Network : Closeable {
    /**
     * Runs [signature]. [inputs] maps every input name to an [IntArray] or a
     * [FloatArray] holding exactly the input's number of elements. Returns the
     * float outputs listed in [outputs], by name.
     */
    fun run(signature: String, inputs: Map<String, Any>, outputs: List<String>): Map<String, FloatArray>
}

fun interface NetworkLoader {
    /** Loads `<name>.tflite`. */
    fun load(name: String): Network
}

/** The static signature sizes of one network (config.json "sizes"). */
class SignatureSizes(private val prefix: String, private val sizes: IntArray) {
    init {
        require(sizes.isNotEmpty() && sizes.toList() == sizes.sorted()) { "bad signature sizes ${sizes.toList()}" }
    }

    val min: Int get() = sizes.first()
    val max: Int get() = sizes.last()

    /** Smallest size that fits [n] elements. */
    fun fit(n: Int): Int {
        for (s in sizes) if (s >= n) return s
        throw IllegalArgumentException("$n exceeds the largest signature $prefix$max")
    }

    fun name(size: Int) = prefix + size
}

/** 1 for the first [n] of [size] positions, 0 for the padding. */
internal fun lengthMask(n: Int, size: Int) = FloatArray(size).also { it.fill(1f, 0, n) }
