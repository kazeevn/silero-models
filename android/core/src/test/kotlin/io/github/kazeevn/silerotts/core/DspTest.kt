package io.github.kazeevn.silerotts.core

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

class DspTest {
    private fun naiveDft(re: DoubleArray, im: DoubleArray, sign: Double): Pair<DoubleArray, DoubleArray> {
        val n = re.size
        val or = DoubleArray(n)
        val oi = DoubleArray(n)
        for (k in 0 until n) {
            var sr = 0.0
            var si = 0.0
            for (t in 0 until n) {
                val a = sign * 2 * PI * k * t / n
                sr += re[t] * cos(a) - im[t] * sin(a)
                si += re[t] * sin(a) + im[t] * cos(a)
            }
            or[k] = sr
            oi[k] = si
        }
        return or to oi
    }

    @Test
    fun complexFftMatchesNaiveDft() {
        val rnd = Random(1)
        for (n in listOf(1, 2, 3, 4, 5, 6, 8, 12, 15, 16, 30, 49, 60, 77, 120, 1200)) {
            for (inverse in listOf(false, true)) {
                val re = DoubleArray(n) { rnd.nextDouble(-1.0, 1.0) }
                val im = DoubleArray(n) { rnd.nextDouble(-1.0, 1.0) }
                val (er, ei) = naiveDft(re, im, if (inverse) 1.0 else -1.0)
                ComplexFft(n, inverse).transform(re, im)
                var err = 0.0
                for (k in 0 until n) err = maxOf(err, abs(re[k] - er[k]), abs(im[k] - ei[k]))
                assertTrue(err < 1e-9 * n, "n=$n inverse=$inverse err=$err")
            }
        }
    }

    @Test
    fun inverseRealFftMatchesDefinition() {
        val rnd = Random(2)
        for (n in listOf(8, 12, 2400)) {
            val bins = n / 2 + 1
            val re = FloatArray(bins) { rnd.nextFloat() - 0.5f }
            val im = FloatArray(bins) { rnd.nextFloat() - 0.5f }
            val out = FloatArray(n)
            InverseRealFft(n).transform(re, im, 0, 1, out)
            var err = 0.0
            for (t in 0 until n) {
                // Hermitian extension; imaginary parts of DC / Nyquist ignored
                var s = re[0].toDouble() + re[n / 2] * (if (t % 2 == 0) 1.0 else -1.0)
                for (k in 1 until n / 2) {
                    val a = 2 * PI * k * t / n
                    s += 2 * (re[k] * cos(a) - im[k] * sin(a))
                }
                err = maxOf(err, abs(s / n - out[t]))
            }
            assertTrue(err < 1e-6, "n=$n err=$err")
        }
    }

    @Test
    fun decimatorMatchesStridedConvolution() {
        val rnd = Random(3)
        val h = FloatArray(63) { rnd.nextFloat() - 0.5f }
        val x = FloatArray(10007) { rnd.nextFloat() - 0.5f }
        for (stride in listOf(2, 6)) {
            val got = ArrayList<Float>()
            val d = Decimator(h, stride) { buf, n -> for (i in 0 until n) got.add(buf[i]) }
            var i = 0
            while (i < x.size) {
                val n = minOf(777, x.size - i)
                d.push(x.copyOfRange(i, i + n), n)
                i += n
            }
            d.finish()
            val padded = FloatArray(31) + x + FloatArray(31)
            val expected = (0..(padded.size - h.size) / stride).map { m ->
                var s = 0f
                for (k in h.indices) s += h[k] * padded[m * stride + k]
                s
            }
            assertTrue(got.size == expected.size, "stride=$stride size ${got.size} != ${expected.size}")
            val err = got.indices.maxOf { abs(got[it] - expected[it]) }
            assertTrue(err < 1e-5, "stride=$stride err=$err")
        }
    }
}
