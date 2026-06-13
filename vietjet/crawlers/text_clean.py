from __future__ import annotations

import re
import unicodedata


_BASE64_IMG_TAG = re.compile(r"<Base64-Image-Removed>", re.IGNORECASE)
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_HTML_TAG = re.compile(r"<[^>]+>")
_NAV_NOISE = re.compile(
    r"^[ \t]*\[?[ \t]*Skip to (?:main[ \t]+)?content[ \t]*\]?[^\n]*$",
    re.IGNORECASE | re.MULTILINE,
)
_ZERO_WIDTH = re.compile(r"[​‌‍⁠﻿]")
_MULTI_BLANK_LINE = re.compile(r"\n{3,}")
_TRAILING_WS = re.compile(r"[ \t]+$", re.MULTILINE)
_LEADING_WS = re.compile(r"^[ \t]+", re.MULTILINE)
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_PUNCT_RUN = re.compile(r"([!?.,;:])\1{2,}")
_BULLET_NOISE = re.compile(r"^\s*[\-\*•·]+\s*$", re.MULTILINE)
_EMPTY_LINK_LINE = re.compile(r"^\s*\[\s*\]\([^)]*\)\s*$", re.MULTILINE)
_TABLE_DIVIDER_NOISE = re.compile(r"^[\s\-:|]{0,3}\|[\s\-:|]+\|?\s*$", re.MULTILINE)


def _drop_md_links(text: str) -> str:
    return _MD_LINK.sub(r"\1", text)


def _normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _ZERO_WIDTH.sub("", text)
    return text


def clean_ingest_text(markdown: str) -> str:
    if not markdown:
        return ""

    text = _normalize_unicode(markdown)
    text = _BASE64_IMG_TAG.sub("", text)
    text = _MD_IMAGE.sub("", text)
    text = _drop_md_links(text)
    text = _HTML_TAG.sub("", text)
    text = _NAV_NOISE.sub("", text)
    text = _EMPTY_LINK_LINE.sub("", text)
    text = _BULLET_NOISE.sub("", text)
    text = _PUNCT_RUN.sub(r"\1", text)
    text = _TRAILING_WS.sub("", text)
    text = _LEADING_WS.sub("", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_BLANK_LINE.sub("\n\n", text)

    return text.strip()
