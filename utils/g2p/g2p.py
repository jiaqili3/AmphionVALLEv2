from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator
from phonemizer.utils import list2str, str2list
from typing import List, Union
import os
import json

# separator=Separator(phone=' ', word=' _ ', syllable='|'),
# # 设置用于分隔单词、音节和音素的符号
separator = Separator(word=" _ ", syllable="|", phone=" ")

# 创建支持中文的 Espeak 后端
phonemizer_zh = EspeakBackend(
    "cmn", preserve_punctuation=False, with_stress=False, language_switch="remove-flags"
)
# phonemizer_zh.separator = separator

# 创建支持英文的 Espeak 后端
phonemizer_en = EspeakBackend(
    "en-us",
    preserve_punctuation=False,
    with_stress=False,
    language_switch="remove-flags",
)
# phonemizer_en.separator = separator

# 创建支持日文的 Espeak 后端
phonemizer_ja = EspeakBackend(
    "ja", preserve_punctuation=False, with_stress=False, language_switch="remove-flags"
)
# phonemizer_ja.separator = separator

# 创建支持韩文的 Espeak 后端
phonemizer_ko = EspeakBackend(
    "ko", preserve_punctuation=False, with_stress=False, language_switch="remove-flags"
)
# phonemizer_ko.separator = separator

# 创建支持法文的 Espeak 后端
phonemizer_fr = EspeakBackend(
    "fr-fr",
    preserve_punctuation=False,
    with_stress=False,
    language_switch="remove-flags",
)
# phonemizer_fr.separator = separator

# 创建支持德文的 Espeak 后端
phonemizer_de = EspeakBackend(
    "de", preserve_punctuation=False, with_stress=False, language_switch="remove-flags"
)
# phonemizer_de.separator = separator


lang2backend = {
    "zh": phonemizer_zh,
    "ja": phonemizer_ja,
    "en": phonemizer_en,
    "fr": phonemizer_fr,
    "ko": phonemizer_ko,
    "de": phonemizer_de,
}

with open("./utils/g2p/mls_en.json", "r") as f:
    json_data = f.read()
token = json.loads(json_data)


def phonemizer_g2p(text, language):
    # 根据lang2phonemizertoke调用不同的backend
    langbackend = lang2backend[language]
    phonemes = _phonemize(
        langbackend,
        text,
        separator,
        strip=True,
        njobs=1,
        prepend_text=False,
        preserve_empty_lines=False,
    )
    token_id = []
    if isinstance(phonemes, list):
        for phone in phonemes:
            phonemes_split = phone.split(" ")
            token_id.append([token[p] for p in phonemes_split if p in token])
    else:
        phonemes_split = phonemes.split(" ")
        token_id = [token[p] for p in phonemes_split if p in token]
    return phonemes, token_id


def _phonemize(  # pylint: disable=too-many-arguments
    backend,
    text: Union[str, List[str]],
    separator: Separator,
    strip: bool,
    njobs: int,
    prepend_text: bool,
    preserve_empty_lines: bool,
):
    """Auxiliary function to phonemize()

    Does the phonemization and returns the phonemized text. Raises a
    RuntimeError on error.

    """
    # remember the text type for output (either list or string)
    text_type = type(text)

    # force the text as a list
    text = [line.strip(os.linesep) for line in str2list(text)]

    # if preserving empty lines, note the index of each empty line
    if preserve_empty_lines:
        empty_lines = [n for n, line in enumerate(text) if not line.strip()]

    # ignore empty lines
    text = [line for line in text if line.strip()]

    if text:
        # phonemize the text
        phonemized = backend.phonemize(
            text, separator=separator, strip=strip, njobs=njobs
        )
    else:
        phonemized = []

    # if preserving empty lines, reinsert them into text and phonemized lists
    if preserve_empty_lines:
        for i in empty_lines:  # noqa
            if prepend_text:
                text.insert(i, "")
            phonemized.insert(i, "")

    # at that point, the phonemized text is a list of str. Format it as
    # expected by the parameters
    if prepend_text:
        return list(zip(text, phonemized))
    if text_type == str:
        return list2str(phonemized)
    return phonemized


import sys

if __name__ == "__main__":
    text = "you are so sweet"
    lang = "en"
    phonemes, token = phonemizer_g2p(text, lang)
    print(phonemes, token)
    # phonemes_split = phonemes.split(" ")
    # phonemes_split = [p for p in phonemes_split if p]
    # print(phonemes_split)
    # print(len(phonemes_split))
    # sys.exit(0)
