package io.github.kazeevn.silerotts.core

import io.github.kazeevn.silerotts.core.RussianNumbers.Gender
import io.github.kazeevn.silerotts.core.RussianNumbers.Ord
import java.text.Normalizer

/**
 * Turns arbitrary text (as sent by apps to a system TTS engine) into the
 * Cyrillic-only text the Silero model can read: numbers, dates, times, units,
 * currencies, Latin words, acronyms and symbols are verbalized in Russian.
 * The original model silently drops everything outside its alphabet.
 *
 * A "+" directly before a vowel is kept as a manual stress mark (Silero
 * convention); a combining acute accent (за́мок) is converted to one.
 */
object TextNormalizer {
    private const val CYR_VOWELS = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"
    private const val L = "\\p{L}"
    private const val NOT_WORD_BEFORE = "(?<![\\p{L}\\p{N}])"
    private const val NOT_WORD_AFTER = "(?![\\p{L}\\p{N}])"

    private class Noun(val one: String, val few: String, val many: String, val gender: Gender = Gender.MASC) {
        fun forNumber(n: Long) = RussianNumbers.plural(n, one, few, many)
    }

    private val RUBLE = Noun("рубль", "рубля", "рублей")
    private val KOPEK = Noun("копейка", "копейки", "копеек", Gender.FEM)
    private val DOLLAR = Noun("доллар", "доллара", "долларов")
    private val EURO = Noun("евро", "евро", "евро", Gender.NEUT)
    private val POUND = Noun("фунт", "фунта", "фунтов")
    private val YUAN = Noun("юань", "юаня", "юаней")
    private val PERCENT = Noun("процент", "процента", "процентов")
    private val DEGREE = Noun("градус", "градуса", "градусов")

    private val CURRENCY_SYMBOLS = mapOf('₽' to RUBLE, '$' to DOLLAR, '€' to EURO, '£' to POUND, '¥' to YUAN)

    /** Abbreviated units / nouns read after a number. */
    private val UNITS_RU: Map<String, Noun> = mapOf(
        "руб" to RUBLE, "р" to RUBLE, "коп" to KOPEK,
        "тыс" to Noun("тысяча", "тысячи", "тысяч", Gender.FEM),
        "млн" to Noun("миллион", "миллиона", "миллионов"),
        "млрд" to Noun("миллиард", "миллиарда", "миллиардов"),
        "трлн" to Noun("триллион", "триллиона", "триллионов"),
        "км" to Noun("километр", "километра", "километров"),
        "м" to Noun("метр", "метра", "метров"),
        "см" to Noun("сантиметр", "сантиметра", "сантиметров"),
        "мм" to Noun("миллиметр", "миллиметра", "миллиметров"),
        "кг" to Noun("килограмм", "килограмма", "килограммов"),
        "гр" to Noun("грамм", "грамма", "граммов"),
        "мг" to Noun("миллиграмм", "миллиграмма", "миллиграммов"),
        "л" to Noun("литр", "литра", "литров"),
        "мл" to Noun("миллилитр", "миллилитра", "миллилитров"),
        "ч" to Noun("час", "часа", "часов"),
        "мин" to Noun("минута", "минуты", "минут", Gender.FEM),
        "сек" to Noun("секунда", "секунды", "секунд", Gender.FEM),
        "мс" to Noun("миллисекунда", "миллисекунды", "миллисекунд", Gender.FEM),
        "шт" to Noun("штука", "штуки", "штук", Gender.FEM),
        "кб" to Noun("килобайт", "килобайта", "килобайт"),
        "мб" to Noun("мегабайт", "мегабайта", "мегабайт"),
        "гб" to Noun("гигабайт", "гигабайта", "гигабайт"),
        "тб" to Noun("терабайт", "терабайта", "терабайт"),
        "кбит" to Noun("килобит", "килобита", "килобит"),
        "мбит" to Noun("мегабит", "мегабита", "мегабит"),
        "гц" to Noun("герц", "герца", "герц"),
        "кгц" to Noun("килогерц", "килогерца", "килогерц"),
        "мгц" to Noun("мегагерц", "мегагерца", "мегагерц"),
        "ггц" to Noun("гигагерц", "гигагерца", "гигагерц"),
        "вт" to Noun("ватт", "ватта", "ватт"),
        "квт" to Noun("киловатт", "киловатта", "киловатт"),
        "мач" to Noun("миллиампер-час", "миллиампер-часа", "миллиампер-часов"),
        "км/ч" to Noun("километр в час", "километра в час", "километров в час"),
        "г" to Noun("грамм", "грамма", "граммов"),
    )
    private val UNITS: Map<String, Noun> = UNITS_RU + mapOf(
        // Latin unit spellings
        "kb" to "кб", "mb" to "мб", "gb" to "гб", "tb" to "тб", "hz" to "гц", "khz" to "кгц", "mhz" to "мгц", "ghz" to "ггц",
        "kg" to "кг", "km" to "км", "cm" to "см", "mm" to "мм", "ml" to "мл", "mah" to "мач", "w" to "вт", "kw" to "квт",
        "ms" to "мс", "mbps" to "мбит", "kbps" to "кбит",
    ).mapValues { (_, v) -> UNITS_RU.getValue(v) }

    private val MONTHS = listOf("января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября", "декабря")

    private val ABBREVIATIONS = listOf(
        "т.е." to "то есть", "т. е." to "то есть", "т.к." to "так как", "т. к." to "так как",
        "т.д." to "так далее", "т. д." to "так далее", "т.п." to "тому подобное", "т. п." to "тому подобное",
        "т.н." to "так называемый", "т. н." to "так называемый", "и др." to "и другие", "и пр." to "и прочее",
        "напр." to "например", "см." to "смотри", "прим." to "примечание",
        "г-н" to "господин", "г-жа" to "госпожа", "ул." to "улица", "пр-т" to "проспект",
        "др." to "другие", "стр." to "страница", "рис." to "рисунок", "тел." to "телефон",
    )

    private val LATIN_LETTER_NAMES = mapOf(
        'a' to "эй", 'b' to "би", 'c' to "си", 'd' to "ди", 'e' to "и", 'f' to "эф", 'g' to "джи", 'h' to "эйч",
        'i' to "ай", 'j' to "джей", 'k' to "кей", 'l' to "эл", 'm' to "эм", 'n' to "эн", 'o' to "оу", 'p' to "пи",
        'q' to "кью", 'r' to "ар", 's' to "эс", 't' to "ти", 'u' to "ю", 'v' to "ви", 'w' to "дабл ю", 'x' to "икс",
        'y' to "уай", 'z' to "зед",
    )

    private val CYR_LETTER_NAMES = mapOf(
        'б' to "бэ", 'в' to "вэ", 'г' to "гэ", 'д' to "дэ", 'ж' to "жэ", 'з' to "зэ", 'й' to "й", 'к' to "ка",
        'л' to "эл", 'м' to "эм", 'н' to "эн", 'п' to "пэ", 'р' to "эр", 'с' to "эс", 'т' to "тэ", 'ф' to "эф",
        'х' to "ха", 'ц' to "цэ", 'ч' to "чэ", 'ш' to "ша", 'щ' to "ща", 'ъ' to "твёрдый знак", 'ь' to "мягкий знак",
        'а' to "а", 'е' to "е", 'ё' to "ё", 'и' to "и", 'о' to "о", 'у' to "у", 'ы' to "ы", 'э' to "э", 'ю' to "ю", 'я' to "я",
    )

    /** Acronyms with vowels that are spelled letter by letter. */
    private val SPELLED_ACRONYMS = setOf(
        "США", "МГУ", "ЕС", "ФИО", "ИП", "ИНН", "ОАО", "ООО", "ЗАО", "ПАО", "НКО", "ЧП", "ИТ", "ЕГЭ", "ОГЭ", "СНГ",
        "ЦРУ", "ФБР", "МВФ", "ЛДПР", "КПРФ", "МИД", "СИЗО", "ГИБДД", "ИИ", "ПО", "ОС", "ПК", "АЭС", "ГЭС", "ТЭЦ", "ВИЧ",
    )

    private val SYMBOL_WORDS = mapOf(
        '&' to " и ", '=' to " равно ", '@' to " собака ", '№' to " номер ", '§' to " параграф ", '#' to " решётка ",
        '×' to " на ", '÷' to " разделить на ", '<' to " меньше ", '>' to " больше ", '≈' to " примерно ",
        '≤' to " меньше или равно ", '≥' to " больше или равно ", '~' to " около ", '*' to " ", '_' to " ", '|' to " ",
        '/' to " ", '\\' to " ", '^' to " ", '`' to " ", '{' to " ", '}' to " ", '[' to " ", ']' to " ",
    )

    fun normalize(input: String): String {
        var t = unicode(input)
        t = Regex("(?i)\\b(?:https?://|www\\.)\\S+").replace(t, " ссылка ")
        t = Regex("[\\w.+-]+@[\\w-]+\\.[\\w.]+").replace(t, " электронный адрес ")
        t = abbreviations(t)
        // split letters glued to digits: "iPhone15" -> "iPhone 15", "5G" -> "5 G"
        t = Regex("(?<=\\p{L})(?=\\d)|(?<=\\d)(?=[A-Za-z])").replace(t, " ")
        t = numbers(t)
        t = symbols(t)
        t = latin(t)
        t = acronyms(t)
        return cleanup(t)
    }

    private fun unicode(input: String): String {
        // combining acute accent after a vowel -> "+" before it
        val nfd = Normalizer.normalize(input, Normalizer.Form.NFD)
        val sb = StringBuilder(nfd.length)
        for (c in nfd) {
            when (c) {
                '́' -> {
                    val last = sb.length - 1
                    if (last >= 0 && sb[last].lowercaseChar() in "аеиоуыэюяaeiouy") sb.insert(last, '+')
                }
                '̀' -> {}
                else -> sb.append(c)
            }
        }
        var t = Normalizer.normalize(sb, Normalizer.Form.NFC)
        val out = StringBuilder(t.length)
        for (c in t) {
            out.append(
                when (c) {
                    ' ', ' ', ' ', ' ', ' ', '\t' -> ' '
                    '­', '​', '‌', '‍', '﻿' -> ""
                    '−', '‐', '‑' -> '-'
                    '–', '—', '―' -> " – "
                    '‘', '’', '‚', '′', '`' -> '\''
                    '“', '”', '„', '″' -> '"'
                    '\r' -> '\n'
                    else -> c
                }
            )
        }
        t = out.toString()
        t = t.replace("...", "…")
        return t
    }

    private fun abbreviations(t0: String): String {
        var t = t0
        for ((abbr, full) in ABBREVIATIONS) {
            val re = Regex("(?<!\\d\\s?)" + NOT_WORD_BEFORE + Regex.escape(abbr).replace(" ", "\\E\\s*\\Q"), RegexOption.IGNORE_CASE)
            val src = t
            t = re.replace(src) { m ->
                val keepCap = m.value[0].isUpperCase()
                (if (keepCap) full.replaceFirstChar { it.uppercase() } else full) + if (endsSentence(src, m)) "." else ""
            }
        }
        return t
    }

    private fun num(s: String) = s.replace(Regex("[\\s]"), "")

    private val NEXT_SENTENCE = Regex("\\s+[\\p{Lu}«\"]")

    /** Keeps a period that ended both an abbreviation and the sentence ("100 руб. Потом"). */
    private fun endsSentence(text: String, m: MatchResult): Boolean {
        if (!m.value.endsWith(".")) return false
        val after = m.range.last + 1
        return after >= text.length || NEXT_SENTENCE.matchesAt(text, after)
    }

    private fun numbers(t0: String): String {
        var t = t0
        // thousands separators: "1 000 000" -> "1000000"
        t = Regex("(?<![\\d,.])\\d{1,3}(?:[  ]\\d{3})+(?![\\d])").replace(t) { num(it.value) }
        // negative numbers (before digits get verbalized)
        t = Regex("(?<![\\p{L}\\p{N}])-(?=\\d)").replace(t, " минус ")
        // phone-like "+7" and arithmetic plus
        t = Regex("\\+(?=\\s?\\d)").replace(t, " плюс ")
        t = Regex("(?<=\\d)\\s*\\+\\s*").replace(t, " плюс ")

        // numeric dates dd.mm.yyyy / dd.mm.yy
        t = Regex("$NOT_WORD_BEFORE(\\d{1,2})\\.(\\d{1,2})\\.(\\d{4}|\\d{2})(?![\\d.]\\d)").replace(t) { m ->
            val d = m.groupValues[1].toInt()
            val mo = m.groupValues[2].toInt()
            if (d !in 1..31 || mo !in 1..12) return@replace m.value
            val y = m.groupValues[3].toLong()
            " " + RussianNumbers.ordinal(d.toLong(), Ord.NEUT_NOM) + " " + MONTHS[mo - 1] + " " +
                RussianNumbers.ordinal(y, Ord.MASC_GEN) + " года "
        }
        // "1 января", "21-го марта"
        val months = MONTHS.joinToString("|")
        t = Regex("(?:(?<=^|[^\\p{L}])(до|с|со|от|после|около|для|из|без|кроме|к|ко|по)\\s+)?$NOT_WORD_BEFORE(\\d{1,2})(?:-?(го|е|ое|му))?\\s+($months)", RegexOption.IGNORE_CASE).replace(t) { m ->
            val prep = m.groupValues[1].lowercase()
            val form = when {
                m.groupValues[3] == "го" || prep in setOf("до", "с", "со", "от", "после", "около", "для", "из", "без", "кроме") -> Ord.MASC_GEN
                m.groupValues[3] == "му" || prep in setOf("к", "ко", "по") -> Ord.MASC_DAT
                else -> Ord.NEUT_NOM
            }
            (if (prep.isNotEmpty()) m.groupValues[1] + " " else "") +
                RussianNumbers.ordinal(m.groupValues[2].toLong(), form) + " " + m.groupValues[4]
        }
        // years: "2024 год", "в 2024 году", "2024 г."
        val beforeYears = t
        t = Regex("$NOT_WORD_BEFORE(?:(\\d{1,4})(?:-?(?:м|го|й))?\\s*(годах|годам|годами|годом|году|года|год)|(\\d{4})\\s*(гг\\.|г\\.))(?![\\p{L}])", RegexOption.IGNORE_CASE).replace(beforeYears) { m ->
            val y = (m.groupValues[1].ifEmpty { m.groupValues[3] }).toLong()
            val word = m.groupValues[2].ifEmpty { m.groupValues[4] }.lowercase()
            val (form, noun) = when (word) {
                "год" -> Ord.MASC_NOM to "год"
                "года", "г." -> Ord.MASC_GEN to "года"
                "году" -> Ord.MASC_PREP to "году"
                "годом" -> Ord.MASC_INSTR to "годом"
                "годах" -> Ord.PL_GEN to "годах"
                "годам" -> Ord.PL_GEN to "годам"
                "годами" -> Ord.PL_INSTR to "годами"
                else -> Ord.PL_NOM to "годы"
            }
            RussianNumbers.ordinal(y, form) + " " + noun + if (endsSentence(beforeYears, m)) "." else ""
        }
        // time hh:mm(:ss)
        t = Regex("$NOT_WORD_BEFORE([01]?\\d|2[0-3]):([0-5]\\d)(?::([0-5]\\d))?(?!\\d)").replace(t) { m ->
            val parts = mutableListOf(RussianNumbers.cardinal(m.groupValues[1].toLong()))
            for (g in listOf(m.groupValues[2], m.groupValues[3])) {
                if (g.isEmpty()) continue
                parts.add(if (g == "00") "ноль ноль" else if (g[0] == '0') "ноль " + RussianNumbers.cardinal(g.toLong()) else RussianNumbers.cardinal(g.toLong()))
            }
            parts.joinToString(" ")
        }
        // ordinals with a suffix: "1-й", "XX" is not handled
        t = Regex("$NOT_WORD_BEFORE(\\d{1,9})-(ый|ой|ий|ая|ья|яя|ое|ье|ее|ые|ие|ого|его|ому|ему|ом|ем|ым|им|ых|их|ую|ью|ыми|ими|го|му|й|я|е|м|х|ю|ми|ей)$NOT_WORD_AFTER").replace(t) { m ->
            val form = RussianNumbers.ordFromSuffix(m.groupValues[2]) ?: return@replace m.value
            RussianNumbers.ordinal(m.groupValues[1].toLong(), form)
        }
        // currency symbol before the number: "$5", "€ 10,50"
        t = Regex("([₽$€£¥])\\s?(\\d+(?:[.,]\\d+)?)").replace(t) { m -> amount(m.groupValues[2], CURRENCY_SYMBOLS.getValue(m.groupValues[1][0])) }
        // number + currency symbol / percent / degrees
        t = Regex("(\\d+(?:[.,]\\d+)?)\\s?([₽$€£¥%])").replace(t) { m ->
            val c = m.groupValues[2][0]
            amount(m.groupValues[1], if (c == '%') PERCENT else CURRENCY_SYMBOLS.getValue(c))
        }
        t = Regex("(\\d+(?:[.,]\\d+)?)\\s?°\\s?([CСFF])?(?![\\p{L}])").replace(t) { m ->
            amount(m.groupValues[1], DEGREE) + when (m.groupValues[2]) {
                "C", "С" -> " Цельсия"
                "F" -> " Фаренгейта"
                else -> ""
            }
        }
        // number + unit abbreviation: "5 км", "10 тыс. руб."
        val unitAlt = UNITS.keys.sortedByDescending { it.length }.joinToString("|") { Regex.escape(it) }
        val beforeUnits = t
        t = Regex("(\\d+(?:[.,]\\d+)?)\\s?($unitAlt)\\.?(?![\\p{L}])", RegexOption.IGNORE_CASE).replace(beforeUnits) { m ->
            amount(m.groupValues[1], UNITS.getValue(m.groupValues[2].lowercase())) + if (endsSentence(beforeUnits, m)) "." else ""
        }
        // version numbers / IPs: "1.2.3" -> "один точка два точка три"
        t = Regex("\\d+(?:\\.\\d+){2,}").replace(t) { m -> m.value.split('.').joinToString(" точка ") { RussianNumbers.digitsGrouped(it) } }
        // decimals
        t = Regex("(\\d+)[.,](\\d+)").replace(t) { m -> RussianNumbers.decimal(m.groupValues[1], m.groupValues[2]) }
        // remaining integers; "1"/"2" agree with the gender of the next word
        t = Regex("\\d+(?=(\\s+([А-ЯЁа-яё]+))?)").replace(t) { m ->
            val s = m.value
            if (s.length > 1 && s[0] == '0' || s.length > 15) return@replace RussianNumbers.digitsGrouped(s)
            val n = s.toLong()
            agreeWithNext(n, m.groups[2]?.value?.lowercase())
        }
        return t
    }

    /**
     * Heuristic agreement of numerals ending in 1 / 2 with the following noun
     * ("одна минута", "одну минуту", "одно окно", "две недели").
     */
    private fun agreeWithNext(n: Long, next: String?): String {
        val last2 = n % 100
        val last = n % 10
        if (next == null || last2 in 11..19 || (last != 1L && last != 2L)) return RussianNumbers.cardinal(n)
        val end = next.last()
        return when (last) {
            1L -> when (end) {
                'а', 'я' -> RussianNumbers.cardinal(n, Gender.FEM)
                'у', 'ю' -> RussianNumbers.cardinal(n, Gender.FEM).removeSuffix("одна") + "одну"
                'о', 'е' -> RussianNumbers.cardinal(n, Gender.NEUT)
                else -> RussianNumbers.cardinal(n)
            }
            else -> if (end == 'ы' || end == 'и') RussianNumbers.cardinal(n, Gender.FEM) else RussianNumbers.cardinal(n)
        }
    }

    /** "5,5" + noun -> "пять целых пять десятых процента"; integers agree with the noun. */
    private fun amount(number: String, noun: Noun): String {
        val parts = number.split(',', '.')
        if (parts.size == 2) return RussianNumbers.decimal(parts[0], parts[1]) + " " + noun.few
        val n = number.toLongOrNull() ?: return RussianNumbers.digits(number) + " " + noun.many
        return RussianNumbers.cardinal(n, noun.gender) + " " + noun.forNumber(n)
    }

    private fun symbols(t0: String): String {
        val sb = StringBuilder(t0.length)
        for ((i, c) in t0.withIndex()) {
            when {
                c == '+' -> {
                    // keep as stress mark only directly before a Cyrillic vowel
                    val next = t0.getOrNull(i + 1)
                    if (next != null && next in CYR_VOWELS) sb.append('+') else sb.append(' ')
                }
                c == '%' -> sb.append(" процентов ")
                c == '°' -> sb.append(" градусов ")
                c in CURRENCY_SYMBOLS -> sb.append(' ').append(CURRENCY_SYMBOLS.getValue(c).many).append(' ')
                c in SYMBOL_WORDS -> sb.append(SYMBOL_WORDS.getValue(c))
                c == '(' || c == ')' -> sb.append(", ")
                else -> sb.append(c)
            }
        }
        return sb.toString()
    }

    /** Common English words / brands with established Russian pronunciation. */
    private val ENGLISH_WORDS = mapOf(
        "ok" to "окей", "okay" to "окей", "wifi" to "вай фай", "ios" to "ай оу эс", "iphone" to "айфон", "ipad" to "айпэд",
        "android" to "андроид", "google" to "гугл", "youtube" to "ютуб", "whatsapp" to "вотсап", "telegram" to "телеграм",
        "email" to "имейл", "bluetooth" to "блютус", "windows" to "виндоус", "apple" to "эппл", "microsoft" to "майкрософт",
        "samsung" to "самсунг", "pixel" to "пиксель", "online" to "онлайн", "offline" to "офлайн", "facebook" to "фейсбук",
        "instagram" to "инстаграм", "twitter" to "твиттер", "skype" to "скайп", "zoom" to "зум", "chrome" to "хром",
        "linux" to "линукс", "github" to "гитхаб", "ai" to "эй ай", "chatgpt" to "чат джи пи ти", "claude" to "клод",
        "hello" to "хеллоу", "the" to "зе", "silero" to "силеро", "usb" to "ю эс би", "wi" to "вай", "fi" to "фай",
    )

    private fun latin(t0: String): String {
        // hyphenated / compound brand names first: "Wi-Fi", "e-mail"
        val t = Regex("(?i)\\b(wi)-?(fi)\\b|\\b(e)-(mail)\\b").replace(t0) { m -> if (m.groupValues[1].isNotEmpty()) "wifi" else "email" }
        return Regex("[A-Za-z]+(?:'[A-Za-z]+)?").replace(t) { m -> latinWord(m.value) }
    }

    private fun latinWord(w: String): String {
        ENGLISH_WORDS[w.lowercase()]?.let { return it }
        return when {
            w.length == 1 -> LATIN_LETTER_NAMES[w[0].lowercaseChar()] ?: ""
            (w.all { it.isUpperCase() } && w.length <= 5) || !w.any { it.lowercaseChar() in "aeiouy" } ->
                w.lowercase().mapNotNull { LATIN_LETTER_NAMES[it] }.joinToString(" ")
            else -> Transliterator.english(w.lowercase())
        }
    }

    private fun acronyms(t: String): String = Regex("$NOT_WORD_BEFORE[А-ЯЁ]{2,6}$NOT_WORD_AFTER").replace(t) { m ->
        val w = m.value
        val noVowels = w.none { it in CYR_VOWELS }
        if (w in SPELLED_ACRONYMS || noVowels) w.lowercase().map { CYR_LETTER_NAMES[it] ?: it.toString() }.joinToString(" ") else w
    }

    private fun cleanup(t0: String): String {
        val sb = StringBuilder(t0.length)
        for (c in t0) {
            sb.append(
                when {
                    c.isLetter() || c in " \n.,!?:;-–…+\"'«»" -> c
                    c.isWhitespace() -> ' '
                    else -> ' '
                }
            )
        }
        var t = sb.toString()
        t = Regex("[ ]+").replace(t, " ")
        t = Regex(" ([.,!?:;…])").replace(t, "$1")
        t = Regex("([,;:])[,;:\\s]*(?=[,;:])").replace(t, "")
        t = Regex("(^|\\n)[ ,;:]+").replace(t, "$1")
        return t.lines().joinToString("\n") { it.trim() }.trim()
    }

    /** A synthesis unit: a sentence (or a part of a long one). */
    data class Chunk(val text: String, val sentence: String, val pauseAfterMs: Int)

    private val SENTENCE_END = Regex("(?<=[.!?…])\\s+")

    /**
     * Splits normalized text into sentences; sentences longer than [maxLen]
     * characters are split at commas / dashes / semicolons (or spaces), keeping
     * the whole sentence for intonation classification.
     */
    fun chunks(normalized: String, maxLen: Int = 250): List<Chunk> {
        val out = ArrayList<Chunk>()
        val paragraphs = normalized.split('\n').map { it.trim() }.filter { it.isNotEmpty() }
        for ((pi, para) in paragraphs.withIndex()) {
            val sentences = SENTENCE_END.split(para).map { it.trim() }.filter { it.isNotEmpty() }
            for ((si, s0) in sentences.withIndex()) {
                val s = if (s0.last() in ".!?…,;:–-") s0 else "$s0."
                val parts = splitLong(s, maxLen)
                for ((k, part) in parts.withIndex()) {
                    val lastOfParagraph = si == sentences.size - 1 && k == parts.size - 1 && pi < paragraphs.size - 1
                    out.add(Chunk(part, s, if (lastOfParagraph) 250 else 0))
                }
            }
        }
        return out
    }

    private fun splitLong(s: String, maxLen: Int): List<String> {
        if (s.length <= maxLen) return listOf(s)
        val out = ArrayList<String>()
        var rest = s
        while (rest.length > maxLen) {
            val window = rest.substring(0, maxLen)
            var cut = maxOf(window.lastIndexOf(", "), window.lastIndexOf("; "), window.lastIndexOf(" – "), window.lastIndexOf(": "))
            if (cut < maxLen / 3) cut = window.lastIndexOf(' ')
            if (cut <= 0) cut = maxLen
            var head = rest.substring(0, cut + 1).trim()
            if (head.last() !in ".!?…,;:–-") head = "$head,"
            out.add(head)
            rest = rest.substring(cut + 1).trim().trimStart('–', ' ')
        }
        if (rest.isNotEmpty()) out.add(rest)
        return out
    }
}

/** Rough English-to-Russian transliteration for Latin words inside Russian text. */
internal object Transliterator {
    private val MULTI = listOf(
        "tion" to "шн", "sion" to "жн", "ough" to "оу", "augh" to "о", "igh" to "ай", "sch" to "ск", "tch" to "ч",
        "sh" to "ш", "ch" to "ч", "zh" to "ж", "th" to "т", "ph" to "ф", "kh" to "х", "ck" to "к", "qu" to "кв", "wh" to "в",
        "oo" to "у", "ee" to "и", "ea" to "и", "ou" to "ау", "ow" to "оу", "ay" to "эй", "ai" to "эй", "ey" to "эй",
        "oa" to "оу", "oi" to "ой", "oy" to "ой", "au" to "о", "aw" to "о", "ew" to "ью", "ie" to "и", "ng" to "нг",
        "x" to "кс", "j" to "дж",
    )
    private val SINGLE = mapOf(
        'a' to "а", 'b' to "б", 'c' to "к", 'd' to "д", 'e' to "е", 'f' to "ф", 'g' to "г", 'h' to "х", 'i' to "и",
        'k' to "к", 'l' to "л", 'm' to "м", 'n' to "н", 'o' to "о", 'p' to "п", 'q' to "к", 'r' to "р", 's' to "с",
        't' to "т", 'u' to "у", 'v' to "в", 'w' to "в", 'y' to "и", 'z' to "з",
    )

    fun english(w0: String): String {
        var w = w0.replace("'", "")
        // silent final e: "make" -> "мейк" is too ambitious; just drop it
        if (w.length > 3 && w.endsWith("e") && w[w.length - 2] !in "aeiouy") w = w.dropLast(1)
        val sb = StringBuilder()
        var i = 0
        loop@ while (i < w.length) {
            for ((k, v) in MULTI) {
                if (w.startsWith(k, i)) {
                    sb.append(v)
                    i += k.length
                    continue@loop
                }
            }
            val c = w[i]
            val next = w.getOrNull(i + 1)
            sb.append(
                when {
                    c == 'c' && next != null && next in "eiy" -> "с"
                    c == 'g' && next != null && next in "ei" && i > 0 -> "дж"
                    c == 'y' && i == 0 -> "й"
                    c == 'e' && i == 0 -> "э"
                    else -> SINGLE[c] ?: ""
                }
            )
            i++
        }
        return sb.toString()
    }
}
