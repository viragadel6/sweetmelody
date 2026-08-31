from __future__ import annotations

import regex as re
from lxml import html as lxml_html
from lxml.html import HtmlElement

BLOCK_TAGS = frozenset(
    {
        "p",
        "div",
        "article",
        "section",
        "li",
        "td",
        "th",
        "blockquote",
        "pre",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "dd",
        "dt",
        "figcaption",
    }
)
STRIP_TAGS = frozenset(
    {
        "script",
        "style",
        "noscript",
        "nav",
        "header",
        "footer",
        "aside",
        "form",
        "svg",
        "iframe",
        "button",
        "select",
        "template",
    }
)
NEGATIVE_HINTS = (
    "comment",
    "sidebar",
    "footer",
    "header",
    "menu",
    "nav",
    "share",
    "social",
    "advert",
    "promo",
    "cookie",
    "banner",
    "newsletter",
    "subscribe",
    "related",
    "breadcrumb",
    "pagination",
    "modal",
    "popup",
)
POSITIVE_HINTS = (
    "article",
    "content",
    "body",
    "main",
    "story",
    "post",
    "entry",
    "text",
    "prose",
)

_WS_RE = re.compile(r"[ \t\u00a0]+")
_NL_RE = re.compile(r"\n{3,}")
_SENT_END_RE = re.compile(r"[.!?…]")


def _class_id_signal(element: HtmlElement) -> float:
    raw = " ".join(
        str(element.get(attribute, "")) for attribute in ("class", "id", "role", "itemprop")
    ).casefold()
    if not raw:
        return 0.0
    score = 0.0
    for hint in NEGATIVE_HINTS:
        if hint in raw:
            score -= 12.0
    for hint in POSITIVE_HINTS:
        if hint in raw:
            score += 8.0
    return score


def _node_text(element: HtmlElement) -> str:
    parts: list[str] = []
    for text in element.itertext():
        cleaned = _WS_RE.sub(" ", text).strip()
        if cleaned:
            parts.append(cleaned)
    return " ".join(parts)


def _link_density(element: HtmlElement, text_length: int) -> float:
    if text_length <= 0:
        return 1.0
    anchor_chars = 0
    for anchor in element.iter("a"):
        anchor_chars += len(_node_text(anchor))
    return min(1.0, anchor_chars / float(text_length))


def _score_candidate(element: HtmlElement) -> tuple[float, str]:
    text = _node_text(element)
    length = len(text)
    if length < 120:
        return (-1.0, text)
    commas = text.count(",") + text.count(";")
    sentences = len(_SENT_END_RE.findall(text))
    density = _link_density(element, length)
    paragraph_count = sum(1 for _ in element.iter("p"))
    score = (
        (length / 100.0)
        + (commas * 1.5)
        + (sentences * 1.2)
        + (paragraph_count * 3.0)
        + _class_id_signal(element)
        - (density * 60.0)
    )
    return (score, text)


def _paragraphize(element: HtmlElement) -> str:
    blocks: list[str] = []
    for node in element.iter():
        tag = node.tag
        if not isinstance(tag, str) or tag not in BLOCK_TAGS:
            continue
        if any(child.tag in BLOCK_TAGS for child in node if isinstance(child.tag, str)):
            continue
        text = _node_text(node)
        if len(text) < 25:
            continue
        blocks.append(text)
    if not blocks:
        return _node_text(element)
    deduped: list[str] = []
    seen: set[str] = set()
    for block in blocks:
        key = block[:160]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(block)
    return "\n\n".join(deduped)


def extract_title(document: HtmlElement) -> str:
    for xpath in (
        "//meta[@property='og:title']/@content",
        "//meta[@name='twitter:title']/@content",
        "//h1//text()",
        "//title//text()",
    ):
        values = document.xpath(xpath)
        for value in values:
            title = _WS_RE.sub(" ", str(value)).strip()
            if title:
                return title[:300]
    return ""


def extract_main_content(raw_html: str, max_chars: int = 12000) -> tuple[str, str]:
    if not raw_html.strip():
        return "", ""
    try:
        document = lxml_html.fromstring(raw_html)
    except Exception:
        text = _WS_RE.sub(" ", re.sub(r"<[^>]+>", " ", raw_html)).strip()
        return text[:max_chars], ""

    title = extract_title(document)

    for tag in STRIP_TAGS:
        for node in document.xpath(f"//{tag}"):
            parent = node.getparent()
            if parent is not None:
                parent.remove(node)
    for node in document.xpath("//comment()"):
        parent = node.getparent()
        if parent is not None:
            parent.remove(node)

    best_score = -1.0
    best_element: HtmlElement | None = None
    for element in document.iter():
        tag = element.tag
        if not isinstance(tag, str):
            continue
        if tag not in {"div", "article", "section", "main", "body", "td"}:
            continue
        score, _ = _score_candidate(element)
        if score > best_score:
            best_score = score
            best_element = element

    if best_element is None:
        body = document.find("body")
        best_element = body if body is not None else document

    content = _paragraphize(best_element)
    content = _NL_RE.sub("\n\n", content).strip()
    return content[:max_chars], title
