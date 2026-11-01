import re

import markdown
import nh3
from django import template
from django.utils.safestring import mark_safe

register = template.Library()

_md = markdown.Markdown(
    extensions=["fenced_code", "tables", "toc"],
    output_format="html",
)

# Explicit allow-lists. Python-Markdown passes raw HTML through verbatim, so
# every Markdown source (documentation, tutorials and LLM answers) is
# untrusted until it has been through this sanitizer. Anything not listed here
# is removed; <script> and <style> are removed together with their content.
_ALLOWED_TAGS = {
    "p", "br", "hr",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "ul", "ol", "li",
    "strong", "em", "b", "i",
    "code", "pre", "blockquote",
    "table", "thead", "tbody", "tr", "th", "td",
    "a", "img",
}
_ALLOWED_ATTRIBUTES = {
    "a": {"href", "title"},
    "img": {"src", "alt", "title"},
    "code": {"class"},
    # `toc` generates heading ids, which the documentation anchors rely on.
    "h1": {"id"}, "h2": {"id"}, "h3": {"id"},
    "h4": {"id"}, "h5": {"id"}, "h6": {"id"},
    # Table alignment is rendered by Markdown as an inline style.
    "th": {"style"}, "td": {"style"},
}
# http(s) and mailto only. Relative URLs pass; javascript:, data:, vbscript:
# and every other scheme are dropped, including encoded or obfuscated forms,
# because the check runs on the parsed attribute value.
_ALLOWED_URL_SCHEMES = {"http", "https", "mailto"}

_CODE_CLASS = re.compile(r"language-[A-Za-z0-9_+.#-]{1,40}")
_HEADING_ID = re.compile(r"[A-Za-z0-9_-]{1,300}")
_ALIGN_STYLE = re.compile(r"text-align:\s*(left|right|center);?")


def _is_local_path(value):
    return value.startswith("/") and not value.startswith("//") and "\\" not in value


def _attribute_filter(tag, attribute, value):
    """Narrow the allowed attributes to the exact shapes Markdown produces."""
    if attribute == "class":
        return value if _CODE_CLASS.fullmatch(value) else None
    if attribute == "id":
        return value if _HEADING_ID.fullmatch(value) else None
    if attribute == "style":
        return value if _ALIGN_STYLE.fullmatch(value.strip()) else None
    if tag == "img" and attribute == "src":
        # Remote images would let untrusted content make the reader's browser
        # send requests (tracking or data exfiltration through the URL).
        return value if _is_local_path(value) else None
    return value


@register.filter(name="markdownify")
def markdownify(value):
    """Convert a Markdown string to sanitized, safe HTML."""
    if not value:
        return ""
    _md.reset()
    html = _md.convert(str(value))
    clean = nh3.clean(
        html,
        tags=_ALLOWED_TAGS,
        attributes=_ALLOWED_ATTRIBUTES,
        attribute_filter=_attribute_filter,
        url_schemes=_ALLOWED_URL_SCHEMES,
        link_rel="noopener noreferrer",
        strip_comments=True,
    )
    return mark_safe(clean)


@register.filter(name="get_item")
def get_item(dictionary, key):
    """Retrieve a value from a dict by key — Django templates can't do dict[var]."""
    if not isinstance(dictionary, dict):
        return None
    return dictionary.get(key)
