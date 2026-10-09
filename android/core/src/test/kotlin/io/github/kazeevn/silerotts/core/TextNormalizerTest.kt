package io.github.kazeevn.silerotts.core

import io.github.kazeevn.silerotts.core.RussianNumbers.Gender
import io.github.kazeevn.silerotts.core.RussianNumbers.Ord
import kotlin.test.Test
import kotlin.test.assertEquals

class TextNormalizerTest {
    @Test
    fun cardinals() {
        assertEquals("ноль", RussianNumbers.cardinal(0))
        assertEquals("двадцать один", RussianNumbers.cardinal(21))
        assertEquals("сто двенадцать", RussianNumbers.cardinal(112))
        assertEquals("тысяча двести тридцать четыре", RussianNumbers.cardinal(1234))
        assertEquals("две тысячи двадцать четыре", RussianNumbers.cardinal(2024))
        assertEquals("пять тысяч", RussianNumbers.cardinal(5000))
        assertEquals("миллион два", RussianNumbers.cardinal(1_000_002))
        assertEquals("двадцать одна", RussianNumbers.cardinal(21, Gender.FEM))
        assertEquals("минус семь", RussianNumbers.cardinal(-7))
    }

    @Test
    fun ordinals() {
        assertEquals("первый", RussianNumbers.ordinal(1, Ord.MASC_NOM))
        assertEquals("второй", RussianNumbers.ordinal(2, Ord.MASC_NOM))
        assertEquals("третьего", RussianNumbers.ordinal(3, Ord.MASC_GEN))
        assertEquals("двадцать первое", RussianNumbers.ordinal(21, Ord.NEUT_NOM))
        assertEquals("сороковой", RussianNumbers.ordinal(40, Ord.MASC_NOM))
        assertEquals("сотый", RussianNumbers.ordinal(100, Ord.MASC_NOM))
        assertEquals("две тысячи двадцать четвёртом", RussianNumbers.ordinal(2024, Ord.MASC_PREP))
        assertEquals("двухтысячного", RussianNumbers.ordinal(2000, Ord.MASC_GEN))
        assertEquals("тысяча девятьсот девяностых", RussianNumbers.ordinal(1990, Ord.PL_GEN))
        assertEquals("тысячный", RussianNumbers.ordinal(1000, Ord.MASC_NOM))
    }

    @Test
    fun decimals() {
        assertEquals("три целых четырнадцать сотых", RussianNumbers.decimal("3", "14"))
        assertEquals("одна целая пять десятых", RussianNumbers.decimal("1", "5"))
    }

    private fun n(s: String) = TextNormalizer.normalize(s)

    @Test
    fun normalization() {
        val cases = listOf(
            "У меня 5 яблок." to "У меня пять яблок.",
            "Встреча 1 января в 10:30." to "Встреча первое января в десять тридцать.",
            "Это было в 2024 году." to "Это было в две тысячи двадцать четвёртом году.",
            "Скидка 15% на всё!" to "Скидка пятнадцать процентов на всё!",
            "Цена \$5." to "Цена пять долларов.",
            "Стоит 100 руб." to "Стоит сто рублей.",
            "Стоит 100 руб. Потом." to "Стоит сто рублей. Потом.",
            "Расстояние 3,5 км." to "Расстояние три целых пять десятых километра.",
            "Температура -5°C." to "Температура минус пять градусов Цельсия.",
            "Дата: 01.02.2024." to "Дата: первое февраля две тысячи двадцать четвёртого года.",
            "В 90-х годах." to "В девяностых годах.",
            "Он пришёл 2-ым." to "Он пришёл вторым.",
            "Во 2-м классе." to "Во втором классе.",
            "Открой USB порт." to "Открой ю эс би порт.",
            "Я из ФСБ." to "Я из эф эс бэ.",
            "Жил-был кот, т.е. котик." to "Жил-был кот, то есть котик.",
            "Длина 5 см." to "Длина пять сантиметров.",
            "Ск+оро будет за́мок." to "Ск+оро будет з+амок.",
            "Подожди 1 минуту." to "Подожди одну минуту.",
            "Прошла 21 неделя и 2 недели." to "Прошла двадцать одна неделя и две недели.",
            "Осталось 1 окно и 2 стула." to "Осталось одно окно и два стула.",
            "Версия 1.2.3 вышла." to "Версия один точка два точка три вышла.",
            "Агент 007." to "Агент ноль ноль семь.",
            "Смотри https://example.com сейчас." to "Смотри ссылка сейчас.",
            "Привет, Google!" to "Привет, гугл!",
            "2+2=4" to "два плюс два равно четыре",
            "Вес 200 г." to "Вес двести граммов.",
            "Скидка до 31 декабря." to "Скидка до тридцать первого декабря.",
            "В 1990-х годах." to "В тысяча девятьсот девяностых годах.",
            "Включи Wi-Fi, OK?" to "Включи вай фай, окей?",
            "Это было в 2024 г. Потом всё." to "Это было в две тысячи двадцать четвёртого года. Потом всё.",
            "Купил 2 GB памяти." to "Купил два гигабайта памяти.",
        )
        val failures = cases.mapNotNull { (input, expected) ->
            val got = n(input)
            println("$input  ->  $got")
            if (got != expected) "'$input': expected '$expected', got '$got'" else null
        }
        assertEquals(emptyList(), failures)
    }

    @Test
    fun chunking() {
        val text = "Первое предложение. Второе предложение?\nНовый абзац без точки"
        val chunks = TextNormalizer.chunks(TextNormalizer.normalize(text))
        assertEquals(listOf("Первое предложение.", "Второе предложение?", "Новый абзац без точки."), chunks.map { it.text })
        assertEquals(listOf(0, 250, 0), chunks.map { it.pauseAfterMs })
        val long = (1..40).joinToString(", ") { "слово номер $it" } + "."
        val parts = TextNormalizer.chunks(TextNormalizer.normalize(long), maxLen = 120)
        parts.forEach { assert(it.text.length <= 121) { it.text } }
        assert(parts.size > 3)
        assert(parts.all { it.sentence == parts[0].sentence })
    }
}
