from __future__ import annotations

from vietjet.crawl_parallel.target_urls import all_target_urls, filter_target_urls
from vietjet.crawl_parallel.text_clean import clean_ingest_text


def test_strip_markdown_image():
    out = clean_ingest_text("![alt](http://x.com/img.png) Real text")
    assert "img.png" not in out
    assert "alt" not in out
    assert "Real text" in out


def test_strip_markdown_link_keeps_text():
    out = clean_ingest_text("[Click here](http://x.com) for info")
    assert "Click here" in out
    assert "http://x.com" not in out


def test_strip_html_tags():
    out = clean_ingest_text("<div>Some <b>bold</b> text</div>")
    assert "<" not in out
    assert ">" not in out
    assert "Some bold text" in out


def test_strip_nav_keeps_content():
    out = clean_ingest_text("Skip to main content\nReal content here")
    assert "Skip to" not in out
    assert "Real content here" in out


def test_strip_base64_tag():
    out = clean_ingest_text("<Base64-Image-Removed> after image")
    assert "Base64" not in out
    assert "after image" in out


def test_collapse_whitespace():
    out = clean_ingest_text("Multi   spaces   here")
    assert "Multi spaces here" == out


def test_collapse_punctuation_runs():
    out = clean_ingest_text("Text!!!! With???? punctuation......")
    assert "!!!!" not in out
    assert "????" not in out
    assert "..." not in out
    assert "Text!" in out


def test_collapse_blank_lines():
    out = clean_ingest_text("Line 1\n\n\n\n\nLine 2")
    assert out == "Line 1\n\nLine 2"


def test_preserve_vietnamese_diacritics():
    out = clean_ingest_text("## Phí đổi vé hạng eco")
    assert "Phí đổi vé hạng eco" in out
    assert "##" in out


def test_remove_zero_width_chars():
    raw = "Hello​world‌!"
    out = clean_ingest_text(raw)
    assert "​" not in out
    assert "‌" not in out
    assert "Helloworld" in out


def test_filter_target_urls_returns_all_when_no_doc_type():
    urls = filter_target_urls(None)
    assert urls == all_target_urls()
    assert len(urls) > 0


def test_filter_target_urls_pricing():
    urls = filter_target_urls("pricing")
    assert all("phi-va-le-phi" in u or "hoa-don-vat" in u for u in urls)
    assert len(urls) >= 1


def test_filter_target_urls_unknown_doc_type_returns_all():
    urls = filter_target_urls("nonexistent_type")
    assert urls == all_target_urls()
