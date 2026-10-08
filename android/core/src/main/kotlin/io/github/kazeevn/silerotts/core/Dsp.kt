package io.github.kazeevn.silerotts.core

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

/**
 * Mixed-radix (2, 3, 4, 5 + generic) complex FFT for a fixed size, used for the
 * 2400-point inverse real FFT of the Vocos ISTFT head (2400 is not a power of
 * two: it is done as a 1200-point complex FFT = 4*4*3*5*5).
 */
class ComplexFft(val n: Int, private val inverse: Boolean) {
    private val factors = factorize(n)
    private val twRe = DoubleArray(n)
    private val twIm = DoubleArray(n)
    private val bufRe = DoubleArray(n)
    private val bufIm = DoubleArray(n)
    private val maxRadix = factors.maxOrNull() ?: 1
    private val ar = DoubleArray(maxRadix)
    private val ai = DoubleArray(maxRadix)
    private val br = DoubleArray(maxRadix)
    private val bi = DoubleArray(maxRadix)
    private val sg = if (inverse) 1.0 else -1.0

    init {
        for (k in 0 until n) {
            twRe[k] = cos(2 * PI * k / n)
            twIm[k] = sg * sin(2 * PI * k / n)
        }
    }

    private fun factorize(n0: Int): IntArray {
        val out = ArrayList<Int>()
        var n = n0
        for (p in intArrayOf(4, 2, 3, 5)) while (n % p == 0) {
            out.add(p)
            n /= p
        }
        var p = 7
        while (n > 1) {
            while (n % p == 0) {
                out.add(p)
                n /= p
            }
            p += 2
        }
        return out.toIntArray()
    }

    /**
     * In-place unnormalized transform (Stockham autosort, decimation in
     * frequency): at a stage with remaining length `len`, stride `s` and radix p
     * (m = len / p), y[q + s*(p*j + k)] = w_len^(j*k) * sum_r x[q + s*(j + r*m)] w_p^(r*k).
     */
    fun transform(re: DoubleArray, im: DoubleArray) {
        var srcRe = re
        var srcIm = im
        var dstRe = bufRe
        var dstIm = bufIm
        var len = n
        var s = 1
        for (p in factors) {
            val m = len / p
            val twStep = n / len
            for (j in 0 until m) {
                for (q in 0 until s) {
                    for (r in 0 until p) {
                        val idx = q + s * (j + r * m)
                        ar[r] = srcRe[idx]
                        ai[r] = srcIm[idx]
                    }
                    dft(p)
                    val outBase = q + s * p * j
                    dstRe[outBase] = br[0]
                    dstIm[outBase] = bi[0]
                    for (k in 1 until p) {
                        val t = (j * k * twStep) % n
                        val wr = twRe[t]
                        val wi = twIm[t]
                        dstRe[outBase + s * k] = br[k] * wr - bi[k] * wi
                        dstIm[outBase + s * k] = br[k] * wi + bi[k] * wr
                    }
                }
            }
            len = m
            s *= p
            val tr = srcRe; srcRe = dstRe; dstRe = tr
            val ti = srcIm; srcIm = dstIm; dstIm = ti
        }
        if (srcRe !== re) {
            System.arraycopy(srcRe, 0, re, 0, n)
            System.arraycopy(srcIm, 0, im, 0, n)
        }
    }

    private val sqrt3by2 = sin(PI / 3)
    private val c51 = cos(2 * PI / 5)
    private val c52 = cos(4 * PI / 5)
    private val s51 = sin(2 * PI / 5)
    private val s52 = sin(4 * PI / 5)

    /** b = DFT_p(a) with kernel exp(sg * 2 pi i r k / p). */
    private fun dft(p: Int) {
        when (p) {
            2 -> {
                br[0] = ar[0] + ar[1]; bi[0] = ai[0] + ai[1]
                br[1] = ar[0] - ar[1]; bi[1] = ai[0] - ai[1]
            }
            3 -> {
                val sr = ar[1] + ar[2]; val si = ai[1] + ai[2]
                val dr = ar[1] - ar[2]; val di = ai[1] - ai[2]
                val mr = ar[0] - 0.5 * sr; val mi = ai[0] - 0.5 * si
                // (a1 - a2) * i*sg*sin(2pi/3)
                val tr = -sg * sqrt3by2 * di; val ti = sg * sqrt3by2 * dr
                br[0] = ar[0] + sr; bi[0] = ai[0] + si
                br[1] = mr + tr; bi[1] = mi + ti
                br[2] = mr - tr; bi[2] = mi - ti
            }
            4 -> {
                val t0r = ar[0] + ar[2]; val t0i = ai[0] + ai[2]
                val t1r = ar[0] - ar[2]; val t1i = ai[0] - ai[2]
                val t2r = ar[1] + ar[3]; val t2i = ai[1] + ai[3]
                val dr = ar[1] - ar[3]; val di = ai[1] - ai[3]
                // (a1 - a3) * i*sg
                val t3r = -sg * di; val t3i = sg * dr
                br[0] = t0r + t2r; bi[0] = t0i + t2i
                br[1] = t1r + t3r; bi[1] = t1i + t3i
                br[2] = t0r - t2r; bi[2] = t0i - t2i
                br[3] = t1r - t3r; bi[3] = t1i - t3i
            }
            5 -> {
                val s1r = ar[1] + ar[4]; val s1i = ai[1] + ai[4]
                val d1r = ar[1] - ar[4]; val d1i = ai[1] - ai[4]
                val s2r = ar[2] + ar[3]; val s2i = ai[2] + ai[3]
                val d2r = ar[2] - ar[3]; val d2i = ai[2] - ai[3]
                val m1r = ar[0] + c51 * s1r + c52 * s2r; val m1i = ai[0] + c51 * s1i + c52 * s2i
                val m2r = ar[0] + c52 * s1r + c51 * s2r; val m2i = ai[0] + c52 * s1i + c51 * s2i
                // i*sg*(sin(2pi/5) d1 + sin(4pi/5) d2) and i*sg*(sin(4pi/5) d1 - sin(2pi/5) d2)
                val n1r = -sg * (s51 * d1i + s52 * d2i); val n1i = sg * (s51 * d1r + s52 * d2r)
                val n2r = -sg * (s52 * d1i - s51 * d2i); val n2i = sg * (s52 * d1r - s51 * d2r)
                br[0] = ar[0] + s1r + s2r; bi[0] = ai[0] + s1i + s2i
                br[1] = m1r + n1r; bi[1] = m1i + n1i
                br[4] = m1r - n1r; bi[4] = m1i - n1i
                br[2] = m2r + n2r; bi[2] = m2i + n2i
                br[3] = m2r - n2r; bi[3] = m2i - n2i
            }
            else -> {
                val step = n / p
                for (k in 0 until p) {
                    var sr = 0.0
                    var si = 0.0
                    for (r in 0 until p) {
                        val t = ((r * k) % p) * step
                        sr += ar[r] * twRe[t] - ai[r] * twIm[t]
                        si += ar[r] * twIm[t] + ai[r] * twRe[t]
                    }
                    br[k] = sr
                    bi[k] = si
                }
            }
        }
    }
}

/**
 * torch.fft.irfft(X, n) for even n via an n/2-point complex FFT. The imaginary
 * parts of the DC and Nyquist bins are ignored, as in pocketfft / torch.
 */
class InverseRealFft(val n: Int) {
    private val half = n / 2
    private val fft = ComplexFft(half, inverse = true)
    private val zr = DoubleArray(half)
    private val zi = DoubleArray(half)
    private val wr = DoubleArray(half) { cos(2 * PI * it / n) }
    private val wi = DoubleArray(half) { sin(2 * PI * it / n) }

    init {
        require(n % 2 == 0)
    }

    /** re/im hold bins 0..n/2 read with the given stride from offset; writes n samples to out. */
    fun transform(re: FloatArray, im: FloatArray, offset: Int, stride: Int, out: FloatArray) {
        for (k in 0 until half) {
            val xr = re[offset + k * stride].toDouble()
            val xi = if (k == 0) 0.0 else im[offset + k * stride].toDouble()
            val j = half - k // X[k + M] = conj(X[M - k])
            val yr = re[offset + j * stride].toDouble()
            val yi = if (j == half) 0.0 else -im[offset + j * stride].toDouble()
            val er = xr + yr
            val ei = xi + yi
            val dr = xr - yr
            val di = xi - yi
            // O = d * e^{2 pi i k / n}
            val or = dr * wr[k] - di * wi[k]
            val oi = dr * wi[k] + di * wr[k]
            // Z = E + i*O
            zr[k] = er - oi
            zi[k] = ei + or
        }
        fft.transform(zr, zi)
        val scale = 1.0 / n
        for (m in 0 until half) {
            out[2 * m] = (zr[m] * scale).toFloat()
            out[2 * m + 1] = (zi[m] * scale).toFloat()
        }
    }
}

/**
 * Streaming version of the Vocos ISTFT head with padding="same":
 * y = overlap_add(irfft(S_t) * w)[pad:-pad] / overlap_add(w^2)[pad:-pad].
 * Frames must be pushed in order; finished samples are emitted as soon as no
 * later frame can overlap them.
 */
class StreamingIstft(private val nFft: Int, private val hop: Int, private val totalFrames: Int, private val sink: (FloatArray, Int) -> Unit) {
    private val window = FloatArray(nFft) { (0.5 - 0.5 * cos(2 * PI * it / nFft)).toFloat() } // periodic Hann
    private val window2 = FloatArray(nFft) { window[it] * window[it] }
    private val irfft = InverseRealFft(nFft)
    private val frame = FloatArray(nFft)
    private val acc = FloatArray(nFft)
    private val env = FloatArray(nFft)
    private val pad = (nFft - hop) / 2
    private val paddedLen = (totalFrames - 1) * hop + nFft
    private val out = FloatArray(nFft)
    private var base = 0 // padded position of acc[0]
    private var pushed = 0

    /** Adds frame `pushed` given its spectrum (bins 0..nFft/2 at re/im[offset + k*stride]). */
    fun push(re: FloatArray, im: FloatArray, offset: Int, stride: Int) {
        check(pushed < totalFrames)
        irfft.transform(re, im, offset, stride, frame)
        for (i in 0 until nFft) {
            acc[i] += frame[i] * window[i]
            env[i] += window2[i]
        }
        pushed++
        if (pushed < totalFrames) {
            emit(hop)
            System.arraycopy(acc, hop, acc, 0, nFft - hop)
            System.arraycopy(env, hop, env, 0, nFft - hop)
            acc.fill(0f, nFft - hop, nFft)
            env.fill(0f, nFft - hop, nFft)
            base += hop
        } else {
            emit(nFft)
        }
    }

    /** Emits acc[0, count) restricted to the un-padded range [pad, paddedLen - pad). */
    private fun emit(count: Int) {
        var n = 0
        for (i in 0 until count) {
            val p = base + i
            if (p < pad || p >= paddedLen - pad) continue
            out[n++] = acc[i] / env[i]
        }
        if (n > 0) sink(out, n)
    }
}

/**
 * Streaming PQMF analysis band 0 (conv1d with stride N and padding taps/2),
 * i.e. what the original vocoder does to produce 24 kHz (N=2) or 8 kHz (N=6)
 * from its 48 kHz output.
 */
class Decimator(private val h: FloatArray, private val stride: Int, private val sink: (FloatArray, Int) -> Unit) {
    private val pad = (h.size - 1) / 2
    private var buf = FloatArray(4096)
    private var len = 0 // valid samples in buf
    private var bufStart = 0L // absolute (padded) index of buf[0]
    private var next = 0L // next output index
    private val out = FloatArray(4096)

    init {
        // left zero padding
        len = pad
    }

    fun push(x: FloatArray, n: Int) {
        ensure(len + n)
        System.arraycopy(x, 0, buf, len, n)
        len += n
        drain()
    }

    fun finish() {
        ensure(len + pad)
        buf.fill(0f, len, len + pad)
        len += pad
        drain()
    }

    private fun ensure(cap: Int) {
        if (cap > buf.size) buf = buf.copyOf(maxOf(cap, buf.size * 2))
    }

    private fun drain() {
        var n = 0
        while (true) {
            val start = (next * stride - bufStart).toInt()
            if (start + h.size > len) break
            var acc = 0f
            for (k in h.indices) acc += h[k] * buf[start + k]
            out[n++] = acc
            next++
            if (n == out.size) {
                sink(out, n)
                n = 0
            }
        }
        if (n > 0) sink(out, n)
        // drop consumed samples
        val keepFrom = (next * stride - bufStart).toInt().coerceAtMost(len)
        if (keepFrom > 0) {
            System.arraycopy(buf, keepFrom, buf, 0, len - keepFrom)
            len -= keepFrom
            bufStart += keepFrom
        }
    }
}
