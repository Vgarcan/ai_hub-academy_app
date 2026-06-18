import markdown
from django import template
from django.utils.safestring import mark_safe

register = template.Library()

_md = markdown.Markdown(
    extensions=["fenced_code", "tables", "toc"],
    output_format="html",
)


@register.filter(name="markdownify")
def markdownify(value):
    """Convert a Markdown string to safe HTML."""
    if not value:
        return ""
    _md.reset()
    html = _md.convert(str(value))
    return mark_safe(html)


@register.filter(name="get_item")
def get_item(dictionary, key):
    """Retrieve a value from a dict by key — Django templates can't do dict[var]."""
    if not isinstance(dictionary, dict):
        return None
    return dictionary.get(key)
