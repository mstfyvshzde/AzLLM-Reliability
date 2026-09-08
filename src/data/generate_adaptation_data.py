"""Deterministic synthetic adaptation verisi üretir.

Bu modül boş adaptation candidate kayıtlarını okuyarak
kategori ve subcategory bilgilerine göre Azerbaijani
instruction-response örnekleri üretir.

Generation deterministic olacak şekilde seed kullanır.
Üretilen örnekler frozen benchmark'tan bağımsız olmalıdır.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import yaml

from src.data.adaptation_record import AdaptationRecord

AZ_ALPHABET = (
    "a",
    "b",
    "c",
    "ç",
    "d",
    "e",
    "ə",
    "f",
    "g",
    "ğ",
    "h",
    "x",
    "ı",
    "i",
    "j",
    "k",
    "q",
    "l",
    "m",
    "n",
    "o",
    "ö",
    "p",
    "r",
    "s",
    "ş",
    "t",
    "u",
    "ü",
    "v",
    "y",
    "z",
)

AZ_ALPHABET_INDEX = {
    char: index
    for index, char in enumerate(AZ_ALPHABET)
}


def az_lower(
    text: str,
) -> str:
    """Azerbaijani büyük harflerini doğru şekilde küçültür."""

    translation = str.maketrans(
        {
            "I": "ı",
            "İ": "i",
        }
    )

    return text.translate(
        translation
    ).lower()


def az_sort_key(
    word: str,
) -> tuple[int, ...]:
    """Kelimeyi Azerbaijani alfabe sırasına göre sıralamak için key üretir."""

    normalized = az_lower(
        word.strip()
    )

    return tuple(
        AZ_ALPHABET_INDEX.get(
            char,
            len(AZ_ALPHABET),
        )
        for char in normalized
    )

def load_yaml(
    path: Path,
) -> dict[str, Any]:
    """YAML dosyasını yükler."""

    if not path.exists():
        raise FileNotFoundError(
            f"Config not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(
            "Config must be a mapping."
        )

    return data


def load_candidates(
    path: Path,
) -> list[dict[str, Any]]:
    """Candidate JSONL kayıtlarını yükler."""

    if not path.exists():
        raise FileNotFoundError(
            f"Candidate file not found: {path}"
        )

    rows: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            if not line.strip():
                continue

            row = json.loads(line)

            if not isinstance(row, dict):
                raise ValueError(
                    "Each candidate must be a JSON object."
                )

            rows.append(row)

    return rows


def choose_subcategory(
    category: str,
    category_config: dict[str, Any],
    index: int,
) -> str:
    """Subcategory seçimini dengeli ve deterministic yapar."""

    subcategories = category_config.get(
        "subcategories"
    )

    if not isinstance(
        subcategories,
        list,
    ):
        raise ValueError(
            f"Missing subcategories for category: {category}"
        )

    if not subcategories:
        raise ValueError(
            f"Empty subcategory list for category: {category}"
        )

    return str(
        subcategories[index % len(subcategories)]
    )


def generate_instruction_following(
    subcategory: str,
    rng: random.Random,
    variant_index: int,
) -> tuple[str, str]:
    """Instruction-following örneği üretir."""

    if subcategory == "format_following":
        number = 20 + variant_index

        instruction = (
            f"{number} ədədini yalnız mötərizə içində yaz. "
            "Əlavə mətn yazma."
        )

        response = f"({number})"

        return instruction, response

    if subcategory == "constraint_following":
        word_sets = [
            ["alma", "kitab", "dəniz"],
            ["bulud", "qələm", "ağac"],
            ["məktəb", "avtobus", "çiçək"],
            ["dəniz", "saat", "balıq"],
            ["telefon", "masa", "kitab"],
            ["şəhər", "kənd", "yol"],
            ["qapı", "pəncərə", "otaq"],
            ["su", "çay", "dəniz"],
            ["alma", "armud", "banan"],
            ["dəftər", "qələm", "kitab"],
        ]

        selected = word_sets[
            variant_index % len(word_sets)
        ]

        instruction = (
            "Bu sözləri əlifba sırası ilə yaz və yalnız "
            "vergüllə ayır: "
            + ", ".join(selected)
        )

        response = ", ".join(
            sorted(
                selected,
                key=az_sort_key
                )
        )

        return instruction, response

    if subcategory == "extraction":
        examples = [
            ("Aysel", "Bakı"),
            ("Murad", "Gəncə"),
            ("Nigar", "Şəki"),
            ("Tural", "Lənkəran"),
            ("Leyla", "Quba"),
            ("Kamran", "Sumqayıt"),
            ("Ayla", "Şamaxı"),
            ("Orxan", "Mingəçevir"),
            ("Zəhra", "Naxçıvan"),
            ("Samir", "Qəbələ"),
        ]

        name, city = examples[
            variant_index % len(examples)
        ]

        text = (
            f"{name} yay tətilində "
            f"{city} şəhərinə səfər etdi."
        )

        instruction = (
            f'Mətndən yalnız şəhərin adını çıxar: "{text}"'
        )

        response = city

        return instruction, response

    if subcategory == "transformation":
        words = [
            "MƏKTƏB",
            "KİTAB",
            "TƏLƏBƏ",
            "BAKI",
            "SƏHƏR",
            "DƏFTƏR",
            "QƏLƏM",
            "MÜƏLLİM",
            "ŞƏHƏR",
            "LAYİHƏ",
        ]

        word = words[
            variant_index % len(words)
        ]

        instruction = (
            f'"{word}" sözünü kiçik hərflərlə yaz.'
        )

        response = az_lower(word)

        return instruction, response

    if subcategory == "multi_step_instruction":
        base = 2 + variant_index

        numbers = [
            base,
            base + 5,
            base + 2,
        ]

        ordered = sorted(
            numbers,
            reverse=True,
        )

        instruction = (
            "Ədədləri böyükdən kiçiyə sırala və "
            "aralarını tire ilə ayır: "
            + ", ".join(
                str(number)
                for number in numbers
            )
        )

        response = "-".join(
            str(number)
            for number in ordered
        )

        return instruction, response

    raise ValueError(
        f"Unknown instruction subcategory: {subcategory}"
    )


def generate_semantic_understanding(
    subcategory: str,
    rng: random.Random,
    variant_index: int,
) -> tuple[str, str]:
    """Semantic-understanding örneği üretir."""

    if subcategory == "contextual_meaning":
        examples = [
            (
                "ağır",
                "Bu gün onun ağır işi var idi.",
                "çətin və yorucu",
            ),
            (
                "soyuq",
                "Onun münasibəti çox soyuq idi.",
                "emosional uzaq və laqeyd",
            ),
            (
                "parlaq",
                "O, parlaq bir tələbədir.",
                "çox bacarıqlı və uğurlu",
            ),
            (
                "şirin",
                "Onun şirin söhbəti hamını sevindirdi.",
                "xoş və mehriban",
            ),
            (
                "dərin",
                "Bu, dərin bir fikirdir.",
                "mürəkkəb və düşündürücü",
            ),
            (
                "isti",
                "Onlar bizi isti qarşıladılar.",
                "mehriban və səmimi",
            ),
            (
                "sərt",
                "Müəllimin sərt münasibəti hamını təəccübləndirdi.",
                "ciddi və tələbkar",
            ),
            (
                "yüngül",
                "Bu məsələ düşündüyümdən daha yüngül idi.",
                "asan",
            ),
            (
                "quru",
                "Onun çıxışı çox quru idi.",
                "emosiyasız və maraqsız",
            ),
            (
                "açıq",
                "Müəllim məsələni açıq izah etdi.",
                "aydın və anlaşılır",
            ),
        ]

        word, sentence, meaning = examples[
            variant_index % len(examples)
        ]

        instruction = (
            f'Cümlədə "{word}" sözünün mənasını müəyyən et: '
            f'"{sentence}"'
        )

        response = (
            f"Burada '{word}' sözü {meaning} "
            "mənasında işlənib."
        )

        return instruction, response

    if subcategory == "lexical_disambiguation":
        examples = [
            (
                "baş",
                "Dağın başında qar var idi.",
                "dağın ən yüksək hissəsi",
            ),
            (
                "göz",
                "İynənin gözündən sap keçirdi.",
                "iynədə olan kiçik dəlik",
            ),
            (
                "ayaq",
                "Masanın ayağı qırıldı.",
                "masanı saxlayan dayaq hissəsi",
            ),
            (
                "dil",
                "Saatın dili üçü göstərirdi.",
                "saatın göstəricisi",
            ),
            (
                "qanad",
                "Binanın yeni qanadı istifadəyə verildi.",
                "binanın ayrıca hissəsi",
            ),
            (
                "kök",
                "Problemin kökünü tapmaq lazımdır.",
                "əsas səbəb",
            ),
            (
                "ağız",
                "Mağaranın ağzı çox dar idi.",
                "giriş hissəsi",
            ),
            (
                "bel",
                "Dağın beli ilə yol uzanırdı.",
                "dağın orta hissəsi",
            ),
            (
                "üz",
                "Gölün üzü buz bağlamışdı.",
                "səth",
            ),
            (
                "qol",
                "Çayın bir qolu kənddən keçir.",
                "əsas çaydan ayrılan hissə",
            ),
        ]

        word, sentence, meaning = examples[
            variant_index % len(examples)
        ]

        instruction = (
            f'Cümlədə "{word}" sözünün hansı mənada '
            f'işləndiyini de: "{sentence}"'
        )

        response = (
            f"Burada '{word}' sözü {meaning} "
            "mənasında işlənib."
        )

        return instruction, response

    if subcategory == "paraphrase_understanding":
        examples = [
            (
                "Aysel görüşə gecikdi.",
                "Aysel görüşə vaxtında çatmadı.",
                "Bəli, cümlələr əsasən eyni mənanı ifadə edir.",
            ),
            (
                "Murad imtahanı uğurla keçdi.",
                "Murad imtahandan kəsildi.",
                "Xeyr, cümlələr əks mənalar ifadə edir.",
            ),
            (
                "Leyla layihəni tamamladı.",
                "Leyla layihəni bitirdi.",
                "Bəli, cümlələr eyni əsas mənanı ifadə edir.",
            ),
            (
                "Kamran qərarını dəyişdi.",
                "Kamran əvvəlki qərarında qalmadı.",
                "Bəli, cümlələr eyni əsas mənanı ifadə edir.",
            ),
            (
                "Nigar tədbirdə iştirak etmədi.",
                "Nigar tədbirə qatıldı.",
                "Xeyr, cümlələr əks mənalar ifadə edir.",
            ),
            (
                "Ayla problemi həll etdi.",
                "Ayla problemin həllini tapdı.",
                "Bəli, cümlələr eyni əsas mənanı ifadə edir.",
            ),
            (
                "Tural işə erkən gəldi.",
                "Tural işə gec gəldi.",
                "Xeyr, cümlələr əks mənalar ifadə edir.",
            ),
            (
                "Samir təklifi qəbul etdi.",
                "Samir təkliflə razılaşdı.",
                "Bəli, cümlələr eyni əsas mənanı ifadə edir.",
            ),
            (
                "Zəhra kitabı oxuyub qurtardı.",
                "Zəhra kitabı tamamladı.",
                "Bəli, cümlələr əsasən eyni mənanı ifadə edir.",
            ),
            (
                "Orxan səfəri təxirə saldı.",
                "Orxan səfəri planlaşdırılan vaxtda etdi.",
                "Xeyr, cümlələr eyni mənanı ifadə etmir.",
            ),
        ]

        first, second, response = examples[
            variant_index % len(examples)
        ]

        instruction = (
            "Bu iki cümlənin eyni mənanı ifadə edib-etmədiyini de: "
            f'"{first}" "{second}"'
        )

        return instruction, response

    if subcategory == "reference_resolution":
        examples = [
            ("Nigar", "kitabı"),
            ("Aysel", "dəftəri"),
            ("Leyla", "telefonu"),
            ("Murad", "qələmi"),
            ("Kamran", "çantanı"),
            ("Ayla", "jurnalı"),
            ("Tural", "açarı"),
            ("Zəhra", "məktubu"),
            ("Samir", "sənədi"),
            ("Orxan", "faylı"),
        ]

        name, object_name = examples[
            variant_index % len(examples)
        ]

        instruction = (
            f"{name} {object_name} masanın üstünə qoydu. "
            f'Sonra onu çantasına yerləşdirdi. '
            f'"Onu" sözü nəyə aiddir?'
        )

        response = object_name

        return instruction, response

    if subcategory == "discourse_understanding":
        examples = [
            (
                "Murad imtahana çox hazırlaşdı. Buna baxmayaraq, "
                "imtahan günü xəstələndi və nəticəsi zəif oldu.",
                "Buna baxmayaraq",
                "ziddiyyət",
            ),
            (
                "Aysel çox məşq etdi. Buna görə də yarışda yaxşı nəticə göstərdi.",
                "Buna görə də",
                "səbəb-nəticə",
            ),
            (
                "Leyla işini bitirdi, sonra evə getdi.",
                "Sonra",
                "zaman ardıcıllığı",
            ),
            (
                "Kamran yorğun idi, amma işləməyə davam etdi.",
                "Amma",
                "ziddiyyət",
            ),
            (
                "Yağış yağdığı üçün oyun təxirə salındı.",
                "üçün",
                "səbəb-nəticə",
            ),
            (
                "Nigar əvvəl məktuba baxdı, daha sonra cavab yazdı.",
                "daha sonra",
                "zaman ardıcıllığı",
            ),
            (
                "Tural çox çalışdı, buna görə nəticəsi yaxşı oldu.",
                "buna görə",
                "səbəb-nəticə",
            ),
            (
                "Ayla hazırlaşmışdı, lakin imtahan çətin oldu.",
                "lakin",
                "ziddiyyət",
            ),
            (
                "Samir yeməkdən sonra dərs oxudu.",
                "sonra",
                "zaman ardıcıllığı",
            ),
            (
                "Orxan avtobusu qaçırdı, buna görə görüşə gecikdi.",
                "buna görə",
                "səbəb-nəticə",
            ),
        ]

        sentence, marker, relation = examples[
            variant_index % len(examples)
        ]

        instruction = (
            f'{sentence} "{marker}" ifadəsi hansı əlaqəni göstərir?'
        )

        response = (
            f"Bu ifadə {relation} əlaqəsini göstərir."
        )

        return instruction, response

    raise ValueError(
        f"Unknown semantic subcategory: {subcategory}"
    )


def generate_reasoning(
    subcategory: str,
    rng: random.Random,
    variant_index: int,
) -> tuple[str, str]:
    """Reasoning örneği üretir."""

    if subcategory == "arithmetic_reasoning":
        base = 5 + variant_index

        added = 3 + (
            variant_index % 4
        )

        instruction = (
            f"Bir qutuda {base} qələm var. "
            f"Üstünə {added} qələm əlavə olunur. "
            "Qutuda neçə qələm olar?"
        )

        response = str(
            base + added
        )

        return instruction, response

    if subcategory == "logical_reasoning":
        examples = [
            (
                "Bütün lalələr çiçəkdir. "
                "Bütün çiçəklər bitkidir. "
                "Deməli, bütün lalələr bitkidirmi?",
                "Bəli.",
            ),
            (
                "Bütün pişiklər məməlidir. "
                "Bütün məməlilər heyvandır. "
                "Deməli, bütün pişiklər heyvandırmı?",
                "Bəli.",
            ),
            (
                "Bütün kvadratlar düzbucaqlıdır. "
                "Bütün düzbucaqlılar dördbucaqlıdır. "
                "Deməli, bütün kvadratlar dördbucaqlıdırmı?",
                "Bəli.",
            ),
            (
                "Bütün sərçələr quşdur. "
                "Bütün quşlar heyvandır. "
                "Deməli, bütün sərçələr heyvandırmı?",
                "Bəli.",
            ),
            (
                "Bütün tələbələr insandır. "
                "Bəzi insanlar müəllimdir. "
                "Deməli, bütün tələbələr müəllimdirmi?",
                "Xeyr.",
            ),
            (
                "Bütün romanlar kitabdır. "
                "Bəzi kitablar dərslikdir. "
                "Deməli, bütün romanlar dərslikdirmi?",
                "Xeyr.",
            ),
            (
                "Bütün almalar meyvədir. "
                "Bütün meyvələr qidadır. "
                "Deməli, bütün almalar qidadırmı?",
                "Bəli.",
            ),
            (
                "Bütün avtobuslar nəqliyyat vasitəsidir. "
                "Bəzi nəqliyyat vasitələri velosipeddir. "
                "Deməli, bütün avtobuslar velosipeddirmi?",
                "Xeyr.",
            ),
        ]

        return examples[
            variant_index % len(examples)
        ]

    if subcategory == "comparative_reasoning":
        examples = [
            (
                "Aysel Muraddan uzundur. "
                "Murad Kamrandan uzundur. "
                "Ən uzun şəxs kimdir?",
                "Aysel",
            ),
            (
                "Leyla Nigardan yaşlıdır. "
                "Nigar Ayladan yaşlıdır. "
                "Ən yaşlı şəxs kimdir?",
                "Leyla",
            ),
            (
                "Tural Samirdən sürətlidir. "
                "Samir Orxandan sürətlidir. "
                "Ən sürətli şəxs kimdir?",
                "Tural",
            ),
            (
                "Qutu A qutu B-dən ağırdır. "
                "Qutu B qutu C-dən ağırdır. "
                "Ən ağır qutu hansıdır?",
                "Qutu A",
            ),
            (
                "Kitab X kitab Y-dən bahadır. "
                "Kitab Y kitab Z-dən bahadır. "
                "Ən bahalı kitab hansıdır?",
                "Kitab X",
            ),
            (
                "A şəhəri B şəhərindən böyükdür. "
                "B şəhəri C şəhərindən böyükdür. "
                "Ən böyük şəhər hansıdır?",
                "A şəhəri",
            ),
            (
                "Model P model Q-dan dəqiqdir. "
                "Model Q model R-dən dəqiqdir. "
                "Ən dəqiq model hansıdır?",
                "Model P",
            ),
            (
                "Komanda M komanda N-dən çox xal toplayıb. "
                "Komanda N komanda K-dan çox xal toplayıb. "
                "Ən çox xal toplayan komanda hansıdır?",
                "Komanda M",
            ),
        ]

        return examples[
            variant_index % len(examples)
        ]

    if subcategory == "ordering_reasoning":
        sequences = [
            ["A", "B", "C"],
            ["D", "B", "A"],
            ["C", "E", "B"],
            ["F", "A", "D"],
            ["B", "E", "C"],
            ["G", "D", "A"],
            ["E", "C", "F"],
            ["H", "B", "G"],
        ]

        ordered = sequences[
            variant_index % len(sequences)
        ]

        instruction = (
            f"Sıra belədir: {' > '.join(ordered)}. "
            "Elementləri soldan sağa yaz."
        )

        response = ", ".join(
            ordered
        )

        return instruction, response

    if subcategory == "constraint_reasoning":
        examples = [
            (
                "Bir tədbir yalnız zal boş olduqda və təşkilatçı hazır olduqda "
                "keçirilə bilər. Zal boşdur, lakin təşkilatçı hazır deyil. "
                "Tədbir keçirilə bilərmi?",
                "Xeyr.",
            ),
            (
                "Sistem yalnız parol düzgün olduqda və hesab aktiv olduqda "
                "girişə icazə verir. Parol düzgündür, hesab isə aktiv deyil. "
                "Giriş mümkündürmü?",
                "Xeyr.",
            ),
            (
                "Tələbə yalnız layihəni təqdim etdikdə və imtahandan keçdikdə "
                "kursu tamamlayır. Hər iki şərt yerinə yetirilib. "
                "Tələbə kursu tamamlayırmı?",
                "Bəli.",
            ),
            (
                "Maşın yalnız yanacaq olduqda və mühərrik işlək olduqda "
                "hərəkət edə bilər. Yanacaq var, mühərrik işləmir. "
                "Maşın hərəkət edə bilərmi?",
                "Xeyr.",
            ),
            (
                "Sifariş yalnız ödəniş təsdiqləndikdə və məhsul anbarda olduqda "
                "göndərilir. Ödəniş təsdiqlənib və məhsul anbardadır. "
                "Sifariş göndərilə bilərmi?",
                "Bəli.",
            ),
            (
                "İstifadəçi yalnız bileti olduqda və şəxsiyyətini təsdiqlədikdə "
                "tədbirə daxil ola bilər. Bileti var, amma şəxsiyyətini "
                "təsdiqləməyib. Daxil ola bilərmi?",
                "Xeyr.",
            ),
            (
                "Fayl yalnız icazə verildikdə və şəbəkə bağlantısı olduqda "
                "yüklənə bilər. Hər iki şərt yerinə yetirilib. "
                "Fayl yüklənə bilərmi?",
                "Bəli.",
            ),
            (
                "Laboratoriyaya yalnız qoruyucu eynək və əlcək olduqda "
                "daxil olmaq olar. Eynək var, əlcək yoxdur. "
                "Daxil olmaq olarmı?",
                "Xeyr.",
            ),
        ]

        return examples[
            variant_index % len(examples)
        ]

    raise ValueError(
        f"Unknown reasoning subcategory: {subcategory}"
    )


def generate_unanswerable(
    subcategory: str,
    canonical_response: str,
    rng: random.Random,
    variant_index: int,
) -> tuple[str, str]:
    """Unanswerable-abstention örneği üretir."""

    if subcategory == "missing_information":
        examples = [
            "Aysel kitab mağazasından bir kitab aldı. Kitabın qiyməti neçə manat idi?",
            "Murad avtobusla şəhərə getdi. Avtobus bileti neçə manat idi?",
            "Leyla bir restoranda nahar etdi. Hesabın məbləği nə qədər idi?",
            "Kamran yeni telefon aldı. Telefonun qiyməti neçə manat idi?",
            "Nigar muzeyə getdi. Giriş bileti neçə manat idi?",
            "Tural bir kursa yazıldı. Kursun aylıq ödənişi nə qədər idi?",
            "Ayla mağazadan çanta aldı. Çantanın qiyməti neçə manat idi?",
            "Samir taksi ilə evə getdi. Taksi neçə manat tutdu?",
        ]

        instruction = examples[
            variant_index % len(examples)
        ]

    elif subcategory == "underspecified_constraint":
        examples = [
            "A, B və C variantlarından ən yaxşısını seç. Hansının daha yaxşı olduğunu müəyyən edən heç bir meyar verilməyib.",
            "Üç layihədən birini seç: X, Y və Z. Seçim üçün heç bir kriteriya göstərilməyib.",
            "A, B və C namizədlərindən ən uyğununu müəyyən et. Uyğunluq meyarı verilməyib.",
            "Üç marşrutdan ən yaxşısını seç. Vaxt, məsafə və qiymət barədə məlumat yoxdur.",
            "P, Q və R modellərindən birini seç. Hansı xüsusiyyətin vacib olduğu göstərilməyib.",
            "Üç kitabdan ən yaxşısını seç. Qiymətləndirmə meyarı verilməyib.",
            "A, B və C strategiyalarından optimal olanı seç. Optimallıq kriteriyası göstərilməyib.",
            "Üç restorandan ən yaxşısını seç. Qiymət, keyfiyyət və məsafə haqqında məlumat yoxdur.",
        ]

        instruction = examples[
            variant_index % len(examples)
        ]

    elif subcategory == "false_premise":
        examples = [
            "Mətndə Muradın bazar ertəsi imtahan verdiyi deyilmir. Murad bazar ertəsi neçə bal topladı?",
            "Mətndə Leylanın yarışda iştirak etdiyi deyilmir. Leyla neçəinci yeri tutdu?",
            "Ayselin kitab aldığı qeyd olunmayıb. Kitabın qiyməti nə qədər idi?",
            "Kamranın Londona səfər etdiyi barədə məlumat yoxdur. O, Londonda neçə gün qaldı?",
            "Nigarın layihəni təqdim etdiyi deyilmir. Layihədən neçə bal aldı?",
            "Turalın avtomobil aldığı barədə məlumat yoxdur. Avtomobil hansı model idi?",
            "Samirin kursa qatıldığı deyilmir. Kursu neçə ay davam etdi?",
            "Aylanın müsabiqəyə qatıldığı qeyd olunmayıb. O, neçə xal topladı?",
        ]

        instruction = examples[
            variant_index % len(examples)
        ]

    elif subcategory == "impossible_inference":
        examples = [
            "Şirkətdə uzaqdan işləyən əməkdaşların məmnuniyyəti artıb. Bu məlumatdan artımın yeganə səbəbinin uzaqdan iş olduğunu qəti şəkildə demək olarmı?",
            "Bir məktəbdə yeni proqramdan sonra qiymətlər yüksəlib. Bu məlumat proqramın artımın yeganə səbəbi olduğunu sübut edirmi?",
            "Şəhərdə velosiped yolları artdıqdan sonra trafik azalıb. Trafikin azalmasının yalnız velosiped yollarından qaynaqlandığını demək olarmı?",
            "Bir şirkətdə maaş artımından sonra işçi dövriyyəsi azalıb. Bu məlumat bunun yeganə səbəbinin maaş olduğunu göstərirmi?",
            "Universitetdə yeni kitabxana açıldıqdan sonra tələbə nəticələri yaxşılaşıb. Bu məlumat birbaşa səbəb əlaqəsini sübut edirmi?",
            "Bir tətbiq yeniləndikdən sonra istifadəçi sayı artıb. Artımın yalnız yenilənmə ilə bağlı olduğunu qəti demək olarmı?",
            "Bir şəhərdə parklar artdıqdan sonra sakinlərin məmnuniyyəti yüksəlib. Bunun yeganə səbəbinin parklar olduğunu demək olarmı?",
            "Bir zavodda yeni avadanlıq quraşdırıldıqdan sonra məhsuldarlıq artıb. Artımın yalnız avadanlıqla bağlı olduğunu sübut etmək olarmı?",
        ]

        instruction = examples[
            variant_index % len(examples)
        ]

    elif subcategory == "insufficient_context":
        examples = [
            "Kamran bir yarışda iştirak etdi. Onun yarışda neçənci yeri tutduğu barədə məlumat yoxdur. Neçənci yeri tutdu?",
            "Leyla bir imtahana girdi. Neçə bal topladığı göstərilməyib. Neçə bal aldı?",
            "Nigar bir kitab oxudu. Kitabın neçə səhifə olduğu barədə məlumat yoxdur. Kitab neçə səhifə idi?",
            "Murad bir səfərə çıxdı. Səfərin neçə gün davam etdiyi göstərilməyib. Neçə gün davam etdi?",
            "Aysel bir kursa qatıldı. Kursun qiyməti barədə məlumat yoxdur. Kurs neçə manat idi?",
            "Tural bir layihə hazırladı. Layihənin neçə gün çəkdiyi deyilməyib. Neçə gün çəkdi?",
            "Samir bir tədbirdə iştirak etdi. Tədbirin saat neçədə başladığı göstərilməyib. Saat neçədə başladı?",
            "Ayla bir müsabiqəyə qatıldı. Topladığı xal barədə məlumat yoxdur. Neçə xal topladı?",
        ]

        instruction = examples[
            variant_index % len(examples)
        ]

    else:
        raise ValueError(
            f"Unknown unanswerable subcategory: {subcategory}"
        )

    return (
        instruction,
        canonical_response.strip(),
    )


def generate_factual_knowledge(
    subcategory: str,
    rng: random.Random,
    variant_index: int,
) -> tuple[str, str]:
    """Factual-knowledge örneği üretir."""

    if subcategory == "general_knowledge":
        examples = [
            (
                "Azərbaycanın paytaxtı hansıdır?",
                "Bakı",
            ),
            (
                "Yerin təbii peyki hansıdır?",
                "Ay",
            ),
            (
                "Dünyanın ən böyük okeanı hansıdır?",
                "Sakit okean",
            ),
            (
                "Yer Günəş sistemində neçənci planetdir?",
                "Üçüncü planet",
            ),
            (
                "Bitkilərin fotosintez zamanı əsasən qəbul etdiyi qaz hansıdır?",
                "Karbon dioksid",
            ),
            (
                "İnsan bədənində qanı pompalayan orqan hansıdır?",
                "Ürək",
            ),
            (
                "Yer kürəsində ən böyük qitə hansıdır?",
                "Asiya",
            ),
        ]

        return examples[
            variant_index % len(examples)
        ]

    if subcategory == "technology_knowledge":
        examples = [
            (
                "Kompüterdə müvəqqəti işlək məlumatların "
                "saxlandığı əsas yaddaş növü hansıdır?",
                "RAM",
            ),
            (
                "Əməliyyat sisteminin aparatla proqramlar arasında "
                "əsas əlaqəni idarə edən nüvə hissəsi necə adlanır?",
                "Kernel",
            ),
            (
                "Veb səhifələrin ötürülməsində istifadə olunan "
                "əsas tətbiq səviyyəli protokol hansıdır?",
                "HTTP",
            ),
            (
                "Kompüterdə uzunmüddətli məlumat saxlamaq üçün "
                "istifadə olunan qurğulardan biri hansıdır?",
                "SSD",
            ),
            (
                "Python-da açar-dəyər cütlərini saxlayan "
                "əsas məlumat strukturu hansıdır?",
                "dictionary",
            ),
            (
                "Şəbəkədə cihazı müəyyən etmək üçün istifadə olunan "
                "ünvan növü hansıdır?",
                "IP ünvanı",
            ),
            (
                "Versiya nəzarəti üçün geniş istifadə olunan "
                "sistem hansıdır?",
                "Git",
            ),
        ]

        return examples[
            variant_index % len(examples)
        ]

    if subcategory == "quantitative_knowledge":
        number = 3 + variant_index

        instruction = (
            f"{number} ədədinin kvadratı neçədir?"
        )

        response = str(
            number**2
        )

        return instruction, response

    raise ValueError(
        f"Unknown factual subcategory: {subcategory}"
    )


def generate_example(
    category: str,
    subcategory: str,
    generation_config: dict[str, Any],
    rng: random.Random,
    variant_index: int,
) -> tuple[str, str]:
    """Kategoriye göre tek adaptation örneği üretir."""

    if category == "instruction_following":
        return generate_instruction_following(
            subcategory,
            rng,
            variant_index,
        )

    if category == "semantic_understanding":
        return generate_semantic_understanding(
            subcategory,
            rng,
            variant_index,
        )

    if category == "reasoning":
        return generate_reasoning(
            subcategory,
            rng,
            variant_index,
        )

    if category == "unanswerable_abstention":
        category_config = generation_config[
            "categories"
        ][category]

        canonical_response = category_config.get(
            "canonical_response",
            "Verilən məlumatlardan müəyyən etmək mümkün deyil.",
        )

        return generate_unanswerable(
            subcategory,
            canonical_response,
            rng,
            variant_index,
        )

    if category == "factual_knowledge":
        return generate_factual_knowledge(
            subcategory,
            rng,
            variant_index,
        )

    raise ValueError(
        f"Unknown category: {category}"
    )


def generate_records(
    candidates: list[dict[str, Any]],
    generation_config: dict[str, Any],
) -> list[AdaptationRecord]:
    """Bütün candidate kayıtlarını generated record'lara çevirir."""

    generation = generation_config.get(
        "generation",
        {},
    )

    seed = int(
        generation.get(
            "seed",
            17,
        )
    )

    rng = random.Random(
        seed
    )

    categories_config = generation_config.get(
        "categories"
    )

    if not isinstance(
        categories_config,
        dict,
    ):
        raise ValueError(
            "Generation config must contain categories."
        )

    category_indices: dict[str, int] = {}

    records: list[AdaptationRecord] = []

    for candidate in candidates:
        category = candidate["category"]

        if category not in categories_config:
            raise ValueError(
                f"Category not found in generation config: {category}"
            )

        current_index = category_indices.get(
            category,
            0,
        )

        subcategory = choose_subcategory(
            category=category,
            category_config=categories_config[category],
            index=current_index,
        )

        variant_index = (
            current_index
            // len(
            categories_config[category]["subcategories"]
            )
        )

        category_indices[category] = (
            current_index + 1
        )

        instruction, response = generate_example(
            category=category,
            subcategory=subcategory,
            generation_config=generation_config,
            rng=rng,
            variant_index=variant_index,
        )

        metadata = dict(
            candidate.get(
                "metadata",
                {},
            )
        )

        metadata["subcategory"] = subcategory
        metadata["generation_status"] = "generated"
        metadata["review_status"] = "pending"

        records.append(
            AdaptationRecord(
                item_id=candidate["item_id"],
                language=candidate["language"],
                category=category,
                instruction=instruction,
                response=response,
                source=candidate.get(
                    "source",
                    "synthetic",
                ),
                metadata=metadata,
            )
        )

    return records


def save_records(
    records: list[AdaptationRecord],
    output_path: Path,
    overwrite: bool = False,
) -> None:
    """Generated adaptation kayıtlarını JSONL'e kaydeder."""

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record.to_dict(),
                    ensure_ascii=False,
                )
                + "\n"
            )


def parse_arguments() -> argparse.Namespace:
    """CLI argumentlerini parse eder."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic Azerbaijani adaptation data."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/adaptation/generation_v1.0.yaml"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_arguments()

    candidates = load_candidates(
        args.input
    )

    config = load_yaml(
        args.config
    )

    records = generate_records(
        candidates=candidates,
        generation_config=config,
    )

    save_records(
        records=records,
        output_path=args.output,
        overwrite=args.overwrite,
    )

    print(
        f"Generated {len(records)} adaptation records."
    )


if __name__ == "__main__":
    main()