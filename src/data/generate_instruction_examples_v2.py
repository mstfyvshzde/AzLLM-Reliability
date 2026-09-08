"""Generate diverse instruction-following examples for adaptation v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class InstructionExample:
    """Represent one instruction-following example."""

    example_id: str
    language: str
    category: str
    question: str
    reference_answer: str
    metadata: dict[str, Any]


def _make_example(
    *,
    index: int,
    language: str,
    category: str,
    question: str,
    answer: str,
) -> InstructionExample:
    """Create one instruction-following example."""

    return InstructionExample(
        example_id=f"inst_v2_{language}_{index:04d}",
        language=language,
        category=category,
        question=question,
        reference_answer=answer,
        metadata={
            "generation_method": "diverse_parameterized",
            "version": "2.0",
        },
    )


AZ_NAMES = [
    "Aysel",
    "Murad",
    "Leyla",
    "Nərgiz",
    "Kamran",
    "Elvin",
    "Zəhra",
    "Orxan",
    "Nigar",
    "Tural",
]

AZ_CITIES = [
    "Bakı",
    "Gəncə",
    "Şəki",
    "Quba",
    "Lənkəran",
    "Şamaxı",
    "Qəbələ",
    "Naxçıvan",
    "Mingəçevir",
    "Sumqayıt",
]

AZ_WORD_GROUPS = [
    ["alma", "qələm", "dəniz"],
    ["kitab", "bulud", "masa"],
    ["çiçək", "avtobus", "qapı"],
    ["məktəb", "telefon", "ağac"],
    ["körpü", "dəftər", "çay"],
    ["otaq", "pəncərə", "şəhər"],
    ["kağız", "bağ", "saat"],
    ["kompüter", "stul", "yol"],
    ["limon", "çanta", "dağ"],
    ["velosiped", "fincan", "çörək"],
]


def generate_az_instruction_examples(
    count: int = 160,
) -> list[InstructionExample]:
    """Generate diverse Azerbaijani instruction examples."""

    if count != 160:
        raise ValueError(
            "Azerbaijani v2 pool must contain exactly 160 examples."
        )

    examples: list[InstructionExample] = []
    index = 1

    # 1. Format following — 10
    separators = [
        " | ",
        " / ",
        " - ",
        " :: ",
        " + ",
        " ~ ",
        " > ",
        " ; ",
        " # ",
        " : ",
    ]

    for i in range(10):
        words = AZ_WORD_GROUPS[i]
        separator = separators[i]

        question = (
            "Aşağıdakı üç sözü sırasını dəyişmədən "
            f"yalnız '{separator.strip()}' işarəsi ilə ayır: "
            + ", ".join(words)
            + "."
        )

        answer = separator.join(words)

        examples.append(
            _make_example(
                index=index,
                language="az",
                category="format_following",
                question=question,
                answer=answer,
            )
        )
        index += 1

    # 2. Ordering — 10
    number_sets = [
        [41, 9, 27, 13],
        [72, 18, 55, 31],
        [6, 44, 21, 37],
        [93, 17, 68, 42],
        [25, 4, 39, 16],
        [81, 33, 57, 12],
        [48, 7, 29, 64],
        [14, 52, 36, 8],
        [61, 23, 45, 19],
        [76, 11, 58, 34],
    ]

    for i, numbers in enumerate(number_sets):
        if i % 2 == 0:
            ordered = sorted(numbers)
            question = (
                "Ədədləri artan qaydada düz və cavabda "
                "yalnız vergüllə ayrılmış ədədləri yaz: "
                + ", ".join(map(str, numbers))
                + "."
            )
        else:
            ordered = sorted(
                numbers,
                reverse=True,
            )
            question = (
                "Verilən ədədləri böyükdən kiçiyə sırala; "
                "əlavə izah yazma: "
                + ", ".join(map(str, numbers))
                + "."
            )

        answer = ", ".join(map(str, ordered))

        examples.append(
            _make_example(
                index=index,
                language="az",
                category="ordering",
                question=question,
                answer=answer,
            )
        )
        index += 1

    # 3. Extraction — 10
    for i in range(10):
        name = AZ_NAMES[i]
        city = AZ_CITIES[i]
        hour = 8 + i

        if i % 2 == 0:
            question = (
                "Cümlədəki şəhər adını çıxar və yalnız "
                f"şəhəri yaz: {name} saat {hour}:00-da "
                f"{city} şəhərinə gedəcək."
            )
            answer = city
        else:
            question = (
                "Aşağıdakı cümlədən yalnız şəxsin adını "
                f"götür: {name} sabah {city} şəhərində "
                "seminarda iştirak edəcək."
            )
            answer = name

        examples.append(
            _make_example(
                index=index,
                language="az",
                category="extraction",
                question=question,
                answer=answer,
            )
        )
        index += 1

    # 4. Case transformation — 10
    sentences = [
        "sakit dəniz sahili",
        "yaşıl şəhər parkı",
        "kiçik taxta körpü",
        "səhər qatar səfəri",
        "yeni elmi layihə",
        "uzun dağ yolu",
        "maraqlı kitab sərgisi",
        "müasir kompüter sistemi",
        "açıq universitet tədbiri",
        "sürətli elektrik qatarı",
    ]

    for i, sentence in enumerate(sentences):
        if i % 2 == 0:
            question = (
                "Bu ifadəni tamamilə BÖYÜK hərflərlə yaz: "
                f"{sentence}"
            )
            answer = sentence.upper()
        else:
            original = sentence.upper()
            question = (
                "Bu ifadəni tamamilə kiçik hərflərlə yaz: "
                f"{original}"
            )
            answer = sentence.lower()

        examples.append(
            _make_example(
                index=index,
                language="az",
                category="transformation",
                question=question,
                answer=answer,
            )
        )
        index += 1

    # 5. Filtering — 10
    filtering_sets = [
        ([3, 8, 11, 14, 19], "cüt"),
        ([21, 34, 47, 52, 63], "tək"),
        ([5, 16, 20, 27, 40], "cüt"),
        ([12, 17, 26, 31, 44], "tək"),
        ([9, 18, 25, 32, 51], "cüt"),
        ([7, 22, 35, 48, 59], "tək"),
        ([10, 13, 24, 37, 46], "cüt"),
        ([15, 28, 33, 42, 57], "tək"),
        ([6, 19, 30, 41, 54], "cüt"),
        ([11, 20, 29, 38, 49], "tək"),
    ]

    for numbers, kind in filtering_sets:
        if kind == "cüt":
            selected = [
                n for n in numbers
                if n % 2 == 0
            ]
        else:
            selected = [
                n for n in numbers
                if n % 2 != 0
            ]

        question = (
            f"Bu siyahıdan yalnız {kind} ədədləri seç və "
            "artan sıra ilə yaz: "
            + ", ".join(map(str, numbers))
            + "."
        )

        answer = ", ".join(
            map(
                str,
                sorted(selected),
            )
        )

        examples.append(
            _make_example(
                index=index,
                language="az",
                category="filtering",
                question=question,
                answer=answer,
            )
        )
        index += 1

    # 6. Counting — 10
    texts = [
        "alma armud alma üzüm alma",
        "kitab qələm kitab dəftər kitab",
        "dəniz dağ meşə dağ dağ",
        "Bakı Gəncə Bakı Şəki Bakı",
        "qırmızı mavi yaşıl mavi mavi",
        "telefon masa telefon stul telefon",
        "çay su qəhvə çay çay",
        "ağac gül ağac kol ağac",
        "qatar avtobus qatar metro qatar",
        "kağız kitab kağız qələm kağız",
    ]

    targets = [
        "alma",
        "kitab",
        "dağ",
        "Bakı",
        "mavi",
        "telefon",
        "çay",
        "ağac",
        "qatar",
        "kağız",
    ]

    for text, target in zip(
        texts,
        targets,
        strict=True,
    ):
        count_value = text.split().count(
            target
        )

        question = (
            f"'{target}' sözünün bu siyahıda neçə dəfə "
            f"işləndiyini yalnız rəqəmlə yaz: {text}."
        )

        answer = str(count_value)

        examples.append(
            _make_example(
                index=index,
                language="az",
                category="counting",
                question=question,
                answer=answer,
            )
        )
        index += 1

    # 7. Multi-step arithmetic — 10
    arithmetic = [
        (7, 5, 4),
        (9, 6, 8),
        (12, 4, 7),
        (8, 9, 11),
        (15, 3, 6),
        (11, 7, 9),
        (14, 5, 13),
        (6, 12, 10),
        (13, 4, 15),
        (10, 8, 17),
    ]

    for a, b, subtract in arithmetic:
        product = a * b
        result = product - subtract

        question = (
            f"Əvvəl {a} ilə {b}-ni vur, sonra alınan "
            f"nəticədən {subtract} çıx. Son nəticəni "
            "bir cümlə ilə bildir."
        )

        answer = (
            f"{a} × {b} = {product}, "
            f"{product} - {subtract} = {result}; "
            f"son nəticə {result}-dir."
        )

        examples.append(
            _make_example(
                index=index,
                language="az",
                category="multi_step",
                question=question,
                answer=answer,
            )
        )
        index += 1

    # 8. Constraint following — 10
    facts = [
        ("Bakı", "Azərbaycan"),
        ("Mars", "planet"),
        ("Python", "proqramlaşdırma dili"),
        ("Xəzər", "dəniz"),
        ("Yupiter", "planet"),
        ("HTML", "işarələmə dili"),
        ("Ay", "peyk"),
        ("DNT", "genetik material"),
        ("metr", "uzunluq vahidi"),
        ("Saturn", "planet"),
    ]

    for i, (subject, description) in enumerate(facts):
        if i % 2 == 0:
            question = (
                "Cavabı yalnız iki sözlə yaz və nöqtə "
                f"işarəsi istifadə etmə: {subject} — "
                f"{description}."
            )
            answer = f"{subject} {description.split()[0]}"
        else:
            question = (
                "Verilən məlumatı maksimum üç sözlə "
                f"ifadə et: {subject} — {description}."
            )
            answer = f"{subject} {description}"

        examples.append(
            _make_example(
                index=index,
                language="az",
                category="constraint_following",
                question=question,
                answer=answer,
            )
        )
        index += 1

    # 9. Reformatting — 10
    scores = [
        71,
        84,
        66,
        93,
        78,
        88,
        74,
        91,
        69,
        86,
    ]

    for name, score in zip(
        AZ_NAMES,
        scores,
        strict=True,
    ):
        question = (
            "Məlumatı dəqiq bu formatda yaz: "
            "'Tələbə=<ad>; Bal=<bal>'. "
            f"Məlumat: {name}, {score}."
        )

        answer = (
            f"Tələbə={name}; Bal={score}"
        )

        examples.append(
            _make_example(
                index=index,
                language="az",
                category="reformatting",
                question=question,
                answer=answer,
            )
        )
        index += 1

    # 10. Selection — 10
    selections = [
        ("alma", "meyvə", "masa", "alma"),
        ("qatar", "nəqliyyat", "kitab", "qatar"),
        ("Bakı", "şəhər", "qələm", "Bakı"),
        ("telefon", "elektron cihaz", "çörək", "telefon"),
        ("gül", "bitki", "stul", "gül"),
        ("velosiped", "nəqliyyat", "dəftər", "velosiped"),
        ("su", "içki", "qapı", "su"),
        ("kompüter", "elektron cihaz", "alma", "kompüter"),
        ("Şəki", "şəhər", "masa", "Şəki"),
        ("kitab", "oxu materialı", "avtobus", "kitab"),
    ]

    for first, category, second, answer in selections:
        question = (
            f"Bu iki variantdan '{category}' kateqoriyasına "
            f"uyğun olanı seç: {first} / {second}. "
            "Yalnız seçilən sözü yaz."
        )

        examples.append(
            _make_example(
                index=index,
                language="az",
                category="selection",
                question=question,
                answer=answer,
            )
        )
        index += 1

    # 11. Punctuation — 10
    raw_sentences = [
        ("salam necəsən", "Salam, necəsən?"),
        ("bu kitab sənindir", "Bu kitab sənindir?"),
        ("ay nə gözəl mənzərədir", "Ay, nə gözəl mənzərədir!"),
        ("murad bura gəl", "Murad, bura gəl."),
        ("sən bakıya gedəcəksən", "Sən Bakıya gedəcəksən?"),
        ("təbrik edirəm sən qalib oldun", "Təbrik edirəm, sən qalib oldun!"),
        ("leysan başladı evə gedək", "Leysan başladı, evə gedək."),
        ("aysel qapını bağla", "Aysel, qapını bağla."),
        ("bu nə vaxt baş verdi", "Bu nə vaxt baş verdi?"),
        ("əla nəticə əldə etdik", "Əla, nəticə əldə etdik!"),
    ]

    for raw, corrected in raw_sentences:
        question = (
            "Durğu işarələrini və lazım olan böyük hərfləri "
            f"düzəlt: {raw}"
        )

        examples.append(
            _make_example(
                index=index,
                language="az",
                category="punctuation",
                question=question,
                answer=corrected,
            )
        )
        index += 1

    # 12. Replacement — 10
    replacement_data = [
        ("Mən alma aldım.", "alma", "armud"),
        ("Murad kitab oxuyur.", "kitab", "məqalə"),
        ("Avtobus gec gəldi.", "Avtobus", "Qatar"),
        ("Masa pəncərənin yanındadır.", "Masa", "Stul"),
        ("Leyla Bakıya getdi.", "Bakıya", "Gəncəyə"),
        ("Telefon masadadır.", "Telefon", "Kompüter"),
        ("Qələm çantadadır.", "Qələm", "Dəftər"),
        ("Ağac həyətdədir.", "Ağac", "Gül"),
        ("Samir çay içdi.", "çay", "qəhvə"),
        ("Uşaq məktəbə getdi.", "məktəbə", "parka"),
    ]

    for sentence, old, new in replacement_data:
        question = (
            f"Cümlədə yalnız '{old}' sözünü '{new}' ilə "
            f"əvəz et, başqa heç nəyi dəyişmə: {sentence}"
        )

        answer = sentence.replace(
            old,
            new,
        )

        examples.append(
            _make_example(
                index=index,
                language="az",
                category="replacement",
                question=question,
                answer=answer,
            )
        )
        index += 1

    # 13. Structured output — 10
    ages = [
        19,
        22,
        25,
        28,
        31,
        24,
        27,
        30,
        21,
        26,
    ]

    for name, city, age in zip(
        AZ_NAMES,
        AZ_CITIES,
        ages,
        strict=True,
    ):
        question = (
            "Məlumatı tək sətirdə bu quruluşla yaz: "
            "'ad | şəhər | yaş'. "
            f"Verilənlər: {name}; {city}; {age}."
        )

        answer = (
            f"{name} | {city} | {age}"
        )

        examples.append(
            _make_example(
                index=index,
                language="az",
                category="structured_output",
                question=question,
                answer=answer,
            )
        )
        index += 1

    # 14. Reverse ordering — 10
    word_sequences = [
        ["alma", "kitab", "qələm", "masa"],
        ["dəniz", "dağ", "meşə", "çay"],
        ["Bakı", "Şəki", "Quba", "Gəncə"],
        ["telefon", "saat", "çanta", "dəftər"],
        ["qatar", "metro", "avtobus", "velosiped"],
        ["gül", "ağac", "kol", "ot"],
        ["çörək", "pendir", "alma", "su"],
        ["otaq", "qapı", "pəncərə", "masa"],
        ["bulud", "yağış", "külək", "qar"],
        ["kompüter", "klaviatura", "siçan", "ekran"],
    ]

    for words in word_sequences:
        question = (
            "Bu sözlərin sırasını tam tərsinə çevir və "
            "vergüllə ayır: "
            + ", ".join(words)
            + "."
        )

        answer = ", ".join(
            reversed(words)
        )

        examples.append(
            _make_example(
                index=index,
                language="az",
                category="reverse_order",
                question=question,
                answer=answer,
            )
        )
        index += 1

    # 15. Classification — 10
    classification_data = [
        ("17", "tək"),
        ("24", "cüt"),
        ("31", "tək"),
        ("46", "cüt"),
        ("55", "tək"),
        ("68", "cüt"),
        ("73", "tək"),
        ("82", "cüt"),
        ("97", "tək"),
        ("104", "cüt"),
    ]

    for number, label in classification_data:
        question = (
            f"{number} ədədini yalnız 'cüt' və ya 'tək' "
            "sözlərindən biri ilə təsnif et."
        )

        examples.append(
            _make_example(
                index=index,
                language="az",
                category="classification",
                question=question,
                answer=label,
            )
        )
        index += 1

    # 16. Character constraint — 10
    short_answers = [
        ("Azərbaycanın paytaxtını yaz.", "Bakı"),
        ("2 ilə 3-ün cəmini yaz.", "5"),
        ("Qırmızının ingiliscə qarşılığını yaz.", "red"),
        ("Bir həftədə günlərin sayını yaz.", "7"),
        ("Yer planetinin peykini yaz.", "Ay"),
        ("Əlifbanın ilk hərfini yaz.", "A"),
        ("Onun yarısını yaz: 20.", "10"),
        ("Dördün kvadratını yaz.", "16"),
        ("Suyun kimyəvi formulunu yaz.", "H2O"),
        ("Üçbucağın tərəf sayını yaz.", "3"),
    ]

    for prompt, answer in short_answers:
        question = (
            f"{prompt} Yalnız cavabı ver, heç bir izah əlavə etmə."
        )

        examples.append(
            _make_example(
                index=index,
                language="az",
                category="minimal_response",
                question=question,
                answer=answer,
            )
        )
        index += 1

    _validate_unique(
        examples,
        expected_count=160,
    )

    return examples


EN_NAMES = [
    "Alice",
    "Daniel",
    "Emma",
    "Michael",
    "Sarah",
]

EN_CITIES = [
    "London",
    "Paris",
    "Berlin",
    "Rome",
    "Madrid",
]


def generate_en_instruction_examples(
    count: int = 50,
) -> list[InstructionExample]:
    """Generate diverse English replay examples."""

    if count != 50:
        raise ValueError(
            "English replay pool must contain exactly 50 examples."
        )

    examples: list[InstructionExample] = []
    index = 1

    # 1. Format following — 5
    word_sets = [
        ["river", "clock", "paper"],
        ["cloud", "garden", "train"],
        ["window", "coffee", "bridge"],
        ["forest", "phone", "chair"],
        ["pencil", "ocean", "school"],
    ]

    separators = [
        " | ",
        " / ",
        " :: ",
        " - ",
        " > ",
    ]

    for words, separator in zip(
        word_sets,
        separators,
        strict=True,
    ):
        question = (
            "Keep the words in the same order and separate "
            f"them only with '{separator.strip()}': "
            + ", ".join(words)
            + "."
        )

        answer = separator.join(words)

        examples.append(
            _make_example(
                index=index,
                language="en",
                category="format_following",
                question=question,
                answer=answer,
            )
        )
        index += 1

    # 2. Ordering — 5
    number_sets = [
        [43, 12, 28, 7],
        [66, 19, 51, 32],
        [8, 37, 24, 59],
        [91, 46, 17, 73],
        [35, 6, 48, 21],
    ]

    for i, numbers in enumerate(number_sets):
        if i % 2 == 0:
            ordered = sorted(numbers)
            question = (
                "Sort these numbers from smallest to largest "
                "and give only the ordered list: "
                + ", ".join(map(str, numbers))
                + "."
            )
        else:
            ordered = sorted(
                numbers,
                reverse=True,
            )
            question = (
                "Arrange the following values in descending "
                "order without explanation: "
                + ", ".join(map(str, numbers))
                + "."
            )

        answer = ", ".join(map(str, ordered))

        examples.append(
            _make_example(
                index=index,
                language="en",
                category="ordering",
                question=question,
                answer=answer,
            )
        )
        index += 1

    # 3. Extraction — 5
    for i in range(5):
        name = EN_NAMES[i]
        city = EN_CITIES[i]

        question = (
            "Extract only the city name from this sentence: "
            f"{name} will attend a conference in {city} "
            "next month."
        )

        examples.append(
            _make_example(
                index=index,
                language="en",
                category="extraction",
                question=question,
                answer=city,
            )
        )
        index += 1

    # 4. Multi-step — 5
    arithmetic = [
        (7, 4, 3),
        (9, 5, 6),
        (11, 3, 8),
        (8, 7, 9),
        (12, 4, 11),
    ]

    for a, b, subtract in arithmetic:
        product = a * b
        result = product - subtract

        question = (
            f"Multiply {a} by {b}, subtract {subtract}, "
            "and state the final result in one sentence."
        )

        answer = (
            f"{a} × {b} = {product}, "
            f"then {product} - {subtract} = {result}; "
            f"the final result is {result}."
        )

        examples.append(
            _make_example(
                index=index,
                language="en",
                category="multi_step",
                question=question,
                answer=answer,
            )
        )
        index += 1

    # 5. Reformatting — 5
    scores = [
        74,
        82,
        69,
        91,
        87,
    ]

    for name, score in zip(
        EN_NAMES,
        scores,
        strict=True,
    ):
        question = (
            "Rewrite the data using exactly this format: "
            "'Student=<name>; Score=<score>'. "
            f"Data: {name}, {score}."
        )

        answer = (
            f"Student={name}; Score={score}"
        )

        examples.append(
            _make_example(
                index=index,
                language="en",
                category="reformatting",
                question=question,
                answer=answer,
            )
        )
        index += 1

    # 6. Filtering — 5
    number_groups = [
        [7, 12, 19, 24, 31],
        [16, 23, 38, 45, 52],
        [9, 20, 33, 46, 57],
        [14, 27, 36, 49, 62],
        [5, 18, 29, 40, 53],
    ]

    for i, numbers in enumerate(number_groups):
        even = i % 2 == 0

        selected = [
            n
            for n in numbers
            if (n % 2 == 0) == even
        ]

        label = (
            "even"
            if even
            else "odd"
        )

        question = (
            f"Select only the {label} numbers and list them "
            "in ascending order: "
            + ", ".join(map(str, numbers))
            + "."
        )

        answer = ", ".join(
            map(
                str,
                sorted(selected),
            )
        )

        examples.append(
            _make_example(
                index=index,
                language="en",
                category="filtering",
                question=question,
                answer=answer,
            )
        )
        index += 1

    # 7. Replacement — 5
    replacements = [
        ("The train arrived early.", "train", "bus"),
        ("Alice opened the window.", "window", "door"),
        ("The book is on the desk.", "book", "phone"),
        ("Daniel drank tea.", "tea", "coffee"),
        ("The bicycle is outside.", "bicycle", "car"),
    ]

    for sentence, old, new in replacements:
        question = (
            f"Replace only '{old}' with '{new}' and keep "
            f"everything else unchanged: {sentence}"
        )

        answer = sentence.replace(
            old,
            new,
        )

        examples.append(
            _make_example(
                index=index,
                language="en",
                category="replacement",
                question=question,
                answer=answer,
            )
        )
        index += 1

    # 8. Structured output — 5
    ages = [
        21,
        25,
        29,
        33,
        27,
    ]

    for name, city, age in zip(
        EN_NAMES,
        EN_CITIES,
        ages,
        strict=True,
    ):
        question = (
            "Write the information on one line using exactly "
            "'name | city | age'. "
            f"Data: {name}; {city}; {age}."
        )

        answer = (
            f"{name} | {city} | {age}"
        )

        examples.append(
            _make_example(
                index=index,
                language="en",
                category="structured_output",
                question=question,
                answer=answer,
            )
        )
        index += 1

    # 9. Reverse ordering — 5
    sequences = [
        ["apple", "train", "book", "river"],
        ["cloud", "chair", "phone", "garden"],
        ["Berlin", "Rome", "Paris", "Madrid"],
        ["paper", "clock", "window", "bridge"],
        ["coffee", "school", "forest", "ocean"],
    ]

    for words in sequences:
        question = (
            "Reverse the order of these words and separate "
            "them with commas: "
            + ", ".join(words)
            + "."
        )

        answer = ", ".join(
            reversed(words)
        )

        examples.append(
            _make_example(
                index=index,
                language="en",
                category="reverse_order",
                question=question,
                answer=answer,
            )
        )
        index += 1

    # 10. Minimal response — 5
    minimal = [
        (
            "Give the capital of Italy.",
            "Rome",
        ),
        (
            "Write the result of 9 plus 6.",
            "15",
        ),
        (
            "Give the chemical formula for water.",
            "H2O",
        ),
        (
            "Write the number of sides in a pentagon.",
            "5",
        ),
        (
            "Name Earth's natural satellite.",
            "Moon",
        ),
    ]

    for prompt, answer in minimal:
        question = (
            f"{prompt} Give only the answer and no explanation."
        )

        examples.append(
            _make_example(
                index=index,
                language="en",
                category="minimal_response",
                question=question,
                answer=answer,
            )
        )
        index += 1

    _validate_unique(
        examples,
        expected_count=50,
    )

    return examples


def _validate_unique(
    examples: list[InstructionExample],
    expected_count: int,
) -> None:
    """Validate count, IDs, questions, and answers."""

    if len(examples) != expected_count:
        raise ValueError(
            f"Expected {expected_count} examples, "
            f"got {len(examples)}."
        )

    ids = [
        example.example_id
        for example in examples
    ]

    questions = [
        example.question.strip()
        for example in examples
    ]

    if len(ids) != len(set(ids)):
        raise ValueError(
            "Duplicate instruction example_id detected."
        )

    if len(questions) != len(set(questions)):
        raise ValueError(
            "Duplicate instruction question detected."
        )


def generate_instruction_examples(
) -> list[InstructionExample]:
    """Generate the complete v2 instruction pool."""

    examples = (
        generate_az_instruction_examples(
            count=160,
        )
        + generate_en_instruction_examples(
            count=50,
        )
    )

    _validate_unique(
        examples,
        expected_count=210,
    )

    return examples
