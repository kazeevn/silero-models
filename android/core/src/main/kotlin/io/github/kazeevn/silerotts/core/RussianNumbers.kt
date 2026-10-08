package io.github.kazeevn.silerotts.core

/** Russian number verbalization: cardinals, ordinals, fractions and noun agreement. */
object RussianNumbers {
    enum class Gender { MASC, FEM, NEUT }

    /** Grammatical form of an ordinal numeral. */
    enum class Ord { MASC_NOM, FEM_NOM, NEUT_NOM, PL_NOM, MASC_GEN, MASC_DAT, MASC_PREP, MASC_INSTR, FEM_GEN, FEM_ACC, PL_GEN, PL_INSTR }

    private val UNITS = arrayOf("ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять")
    private val TEENS = arrayOf(
        "десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать",
        "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать",
    )
    private val TENS = arrayOf("", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят", "восемьдесят", "девяносто")
    private val HUNDREDS = arrayOf("", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот", "восемьсот", "девятьсот")

    private class Scale(val value: Long, val gender: Gender, val one: String, val few: String, val many: String, val ordStem: String)

    private val SCALES = listOf(
        Scale(1_000_000_000_000L, Gender.MASC, "триллион", "триллиона", "триллионов", "триллионн"),
        Scale(1_000_000_000L, Gender.MASC, "миллиард", "миллиарда", "миллиардов", "миллиардн"),
        Scale(1_000_000L, Gender.MASC, "миллион", "миллиона", "миллионов", "миллионн"),
        Scale(1_000L, Gender.FEM, "тысяча", "тысячи", "тысяч", "тысячн"),
    )

    /** Picks the noun form agreeing with n: 1 стол, 2 стола, 5 столов. */
    fun plural(n: Long, one: String, few: String, many: String): String {
        val a = Math.abs(n) % 100
        if (a in 11..14) return many
        return when (a % 10) {
            1L -> one
            2L, 3L, 4L -> few
            else -> many
        }
    }

    private fun triad(n: Int, gender: Gender, out: MutableList<String>) {
        val h = n / 100
        val rest = n % 100
        if (h > 0) out.add(HUNDREDS[h])
        if (rest in 10..19) {
            out.add(TEENS[rest - 10])
            return
        }
        val t = rest / 10
        val u = rest % 10
        if (t > 0) out.add(TENS[t])
        if (u > 0) out.add(
            when {
                u == 1 && gender == Gender.FEM -> "одна"
                u == 1 && gender == Gender.NEUT -> "одно"
                u == 2 && gender == Gender.FEM -> "две"
                else -> UNITS[u]
            }
        )
    }

    fun cardinal(n0: Long, gender: Gender = Gender.MASC): String {
        if (n0 == 0L) return UNITS[0]
        val out = ArrayList<String>()
        var n = n0
        if (n < 0) {
            out.add("минус")
            n = -n
        }
        for (s in SCALES) {
            val k = (n / s.value).toInt()
            if (k > 0) {
                if (k >= 1000) {
                    // beyond the largest scale: recurse
                    out.add(cardinal(k.toLong(), s.gender))
                } else if (k != 1 || out.isNotEmpty()) {
                    // a leading 1 is not pronounced: "тысяча двести", "миллион"
                    triad(k, s.gender, out)
                }
                out.add(plural(k.toLong(), s.one, s.few, s.many))
                n %= s.value
            }
        }
        if (n > 0) triad(n.toInt(), gender, out)
        return out.joinToString(" ")
    }

    // ordinal stems; "ой"-stressed ones and "трет" are handled in ending()
    private val ORD_UNITS = arrayOf("", "перв", "втор", "трет", "четвёрт", "пят", "шест", "седьм", "восьм", "девят")
    private val ORD_TEENS = arrayOf(
        "десят", "одиннадцат", "двенадцат", "тринадцат", "четырнадцат",
        "пятнадцат", "шестнадцат", "семнадцат", "восемнадцат", "девятнадцат",
    )
    private val ORD_TENS = arrayOf("", "", "двадцат", "тридцат", "сороков", "пятидесят", "шестидесят", "семидесят", "восьмидесят", "девяност")
    private val ORD_HUNDREDS = arrayOf("", "сот", "двухсот", "трёхсот", "четырёхсот", "пятисот", "шестисот", "семисот", "восьмисот", "девятисот")
    // genitive prefixes for compound ordinals: двухтысячный, пятимиллионный
    private val GEN_PREFIX = arrayOf("", "", "двух", "трёх", "четырёх", "пяти", "шести", "семи", "восьми", "девяти")
    private val STRESSED_OI = setOf("втор", "шест", "седьм", "восьм", "сороков")

    private fun ending(stem: String, form: Ord): String {
        if (stem == "трет") return when (form) {
            Ord.MASC_NOM -> "третий"
            Ord.FEM_NOM -> "третья"
            Ord.NEUT_NOM -> "третье"
            Ord.PL_NOM -> "третьи"
            Ord.MASC_GEN -> "третьего"
            Ord.MASC_DAT -> "третьему"
            Ord.MASC_PREP -> "третьем"
            Ord.MASC_INSTR -> "третьим"
            Ord.FEM_GEN -> "третьей"
            Ord.FEM_ACC -> "третью"
            Ord.PL_GEN -> "третьих"
            Ord.PL_INSTR -> "третьими"
        }
        return stem + when (form) {
            Ord.MASC_NOM -> if (stem in STRESSED_OI) "ой" else "ый"
            Ord.FEM_NOM -> "ая"
            Ord.NEUT_NOM -> "ое"
            Ord.PL_NOM -> "ые"
            Ord.MASC_GEN -> "ого"
            Ord.MASC_DAT -> "ому"
            Ord.MASC_PREP -> "ом"
            Ord.MASC_INSTR -> "ым"
            Ord.FEM_GEN -> "ой"
            Ord.FEM_ACC -> "ую"
            Ord.PL_GEN -> "ых"
            Ord.PL_INSTR -> "ыми"
        }
    }

    /** Ordinal numeral: only the last component is declined ("две тысячи двадцать четвёртом"). */
    fun ordinal(n: Long, form: Ord): String {
        if (n <= 0L) return cardinal(n)
        val parts = ArrayList<String>()
        if (n % 1000 == 0L) {
            // round scales: тысячный, двухтысячный, "миллион двухтысячный"
            val s = SCALES.first { n % it.value == 0L && (n / it.value) % 1000 != 0L }
            val head = (n / s.value) % 1000
            val higher = n - head * s.value
            if (higher > 0) parts.add(cardinal(higher))
            when (head) {
                1L -> parts.add(ending(s.ordStem, form))
                in 2L..9L -> parts.add(ending(GEN_PREFIX[head.toInt()] + s.ordStem, form))
                else -> {
                    // e.g. 20000: proper form is a compound ("двадцатитысячный"); approximate
                    parts.add(cardinal(head))
                    parts.add(ending(s.ordStem, form))
                }
            }
            return parts.joinToString(" ")
        }
        val last3 = n % 1000
        if (last3 % 100 == 0L) {
            if (n - last3 > 0) parts.add(cardinal(n - last3))
            parts.add(ending(ORD_HUNDREDS[(last3 / 100).toInt()], form))
            return parts.joinToString(" ")
        }
        val last2 = n % 100
        if (n - last2 > 0) parts.add(cardinal(n - last2))
        val stem = when {
            last2 < 10 -> ORD_UNITS[last2.toInt()]
            last2 < 20 -> ORD_TEENS[(last2 - 10).toInt()]
            last2 % 10 == 0L -> ORD_TENS[(last2 / 10).toInt()]
            else -> {
                parts.add(TENS[(last2 / 10).toInt()])
                ORD_UNITS[(last2 % 10).toInt()]
            }
        }
        parts.add(ending(stem, form))
        return parts.joinToString(" ")
    }

    /** Ordinal form from a written suffix such as "-й", "-го", "-ая", "-х". */
    fun ordFromSuffix(suffix: String): Ord? = when (suffix.lowercase()) {
        "й", "ый", "ой", "ий" -> Ord.MASC_NOM
        "я", "ая", "ья", "яя" -> Ord.FEM_NOM
        "е", "ое", "ье", "ее" -> Ord.NEUT_NOM
        "ые", "ие" -> Ord.PL_NOM
        "го", "ого", "его" -> Ord.MASC_GEN
        "му", "ому", "ему" -> Ord.MASC_DAT
        "м", "ом", "ем" -> Ord.MASC_PREP
        "ым", "им" -> Ord.MASC_INSTR
        "ей" -> Ord.FEM_GEN
        "ю", "ую", "ью" -> Ord.FEM_ACC
        "х", "ых", "их" -> Ord.PL_GEN
        "ми", "ыми", "ими" -> Ord.PL_INSTR
        else -> null
    }

    private val FRACTION_NAMES = arrayOf(
        null,
        Triple("десятая", "десятых", "десятых"),
        Triple("сотая", "сотых", "сотых"),
        Triple("тысячная", "тысячных", "тысячных"),
    )

    /** "3,14" -> "три целых четырнадцать сотых" (up to 3 decimals, digit by digit beyond). */
    fun decimal(intPart: String, fracPart: String): String {
        val whole = intPart.toLongOrNull() ?: return digits(intPart) + " запятая " + digits(fracPart)
        if (fracPart.length > 3) return cardinal(whole) + " запятая " + digitsGrouped(fracPart)
        val frac = fracPart.toLong()
        val (one, few, many) = FRACTION_NAMES[fracPart.length]!!
        return cardinal(whole, Gender.FEM) + " " + plural(whole, "целая", "целых", "целых") + " " +
            cardinal(frac, Gender.FEM) + " " + plural(frac, one, few, many)
    }

    fun digits(s: String): String = s.filter { it.isDigit() }.map { UNITS[it - '0'] }.joinToString(" ")

    /** Leading zeros digit by digit, the rest as a number: "007" -> "ноль ноль семь". */
    fun digitsGrouped(s: String): String {
        val zeros = s.takeWhile { it == '0' }
        val rest = s.substring(zeros.length)
        val parts = zeros.map { UNITS[0] }.toMutableList()
        if (rest.isNotEmpty()) parts.add(if (rest.length <= 15) cardinal(rest.toLong()) else digits(rest))
        return parts.joinToString(" ")
    }
}
