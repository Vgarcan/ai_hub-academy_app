"""
Import Markdown documentation files into the database.

By default it reads two roots: the canonical reusable-platform docs
(AIHUB_DOCS_SOURCE, files 01-14) and the academy-specific docs
(ACADEMY_DOCS_SOURCE, 15+). This keeps a single source of truth — the platform
docs are not duplicated into docs_source.

Usage:
    python manage.py import_academy_docs            # reads both configured roots
    python manage.py import_academy_docs --path some/dir   # single-root override
"""
import re
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils.text import slugify

from academy.models import DocumentationChunk, DocumentationPage, DocumentationSource


def _strip_markdown(text: str) -> str:
    """Remove Markdown syntax to produce clean searchable text."""
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"`[^`]+`", " ", text)
    text = re.sub(r"!\[.*?\]\(.*?\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"#{1,6}\s+", " ", text)
    text = re.sub(r"[*_~]{1,3}", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _estimate_tokens(text: str) -> int:
    """Rough estimate: ~4 characters per token."""
    return max(1, len(text) // 4)


def _parse_heading(line: str):
    """Return (level, text) for a Markdown heading line, or None."""
    m = re.match(r"^(#{1,3})\s+(.*)", line)
    if m:
        return len(m.group(1)), m.group(2).strip()
    return None


def _split_into_chunks(markdown_text: str) -> list[dict]:
    """
    Split a Markdown document into chunks by headings (H1/H2/H3).

    Returns a list of dicts: {heading, anchor, body, order}.
    """
    lines = markdown_text.splitlines()
    chunks = []
    current_heading = "Introduction"
    current_lines = []

    def flush(heading, lines_buf, order):
        body = "\n".join(lines_buf).strip()
        if body or heading != "Introduction":
            anchor = slugify(heading)[:300]
            chunks.append({
                "heading": heading,
                "anchor": anchor,
                "body": body,
                "order": order,
            })

    order = 1
    for line in lines:
        parsed = _parse_heading(line)
        if parsed:
            flush(current_heading, current_lines, order)
            order += 1
            _, current_heading = parsed
            current_lines = []
        else:
            current_lines.append(line)

    flush(current_heading, current_lines, order)

    if not chunks:
        chunks.append({
            "heading": "Content",
            "anchor": "content",
            "body": markdown_text.strip(),
            "order": 1,
        })

    return chunks


def _page_title_from_filename(path: Path) -> str:
    """Derive a human-readable title from a filename like '04_CORE_CONCEPTS.md'."""
    stem = path.stem
    # Strip all leading numeric prefixes: '01_02_FOO' → 'FOO'
    stem = re.sub(r"^(\d+_)+", "", stem)
    return stem.replace("_", " ").title()


def collect_doc_files(explicit_path=None):
    """Return an ordered list of ``(md_file, root)`` pairs to import.

    With ``explicit_path`` it behaves as a single-root import (legacy ``--path``).
    Otherwise it reads the canonical reusable-platform docs first
    (``AIHUB_DOCS_SOURCE``, files 01-14) and then the academy-specific docs
    (``ACADEMY_DOCS_SOURCE``, 15+). The platform ``README`` is skipped (it is a
    developer index, not a published page) and duplicate page slugs are dropped so
    the first (platform) root wins. This gives one source of truth with no
    duplicated, drifting copies.
    """
    if explicit_path:
        roots = [(Path(explicit_path), False)]
    else:
        roots = []
        platform = getattr(settings, "AIHUB_DOCS_SOURCE", None)
        academy = getattr(settings, "ACADEMY_DOCS_SOURCE", None)
        if platform:
            roots.append((Path(platform), True))
        if academy:
            roots.append((Path(academy), False))

    seen_slugs = set()
    files = []
    for root, is_platform in roots:
        if not root.exists():
            continue
        for md_file in sorted(root.glob("**/*.md")):
            if is_platform and md_file.stem.upper() == "README":
                continue
            slug = slugify(md_file.stem)[:50]
            if slug in seen_slugs:
                continue
            seen_slugs.add(slug)
            files.append((md_file, root))
    return files


class Command(BaseCommand):
    help = "Import Markdown files from a directory into the documentation database."

    def add_arguments(self, parser):
        parser.add_argument(
            "--path",
            type=str,
            default=None,
            help="Directory containing .md files (default: ACADEMY_DOCS_SOURCE in settings).",
        )
        parser.add_argument(
            "--source-name",
            type=str,
            default="AI Hub Official Docs",
            help="Name for the DocumentationSource record.",
        )
        parser.add_argument(
            "--embed",
            action="store_true",
            help="Run embed_docs after importing (requires Ollama + bge-m3:latest).",
        )
        parser.add_argument(
            "--embed-model",
            type=str,
            default="bge-m3:latest",
            help="Embedding model to use when --embed is set (default: bge-m3:latest).",
        )

    def handle(self, *args, **options):
        doc_files = collect_doc_files(options["path"])
        if not doc_files:
            self.stderr.write("No .md files found to import.")
            return

        source_name = options["source_name"]
        source_slug = slugify(source_name)
        source, created = DocumentationSource.objects.get_or_create(
            slug=source_slug,
            defaults={"name": source_name, "is_active": True},
        )
        if created:
            self.stdout.write(f"Created source: {source_name}")

        pages_created = 0
        chunks_created = 0

        for order, (md_file, root) in enumerate(doc_files, start=1):
            title = _page_title_from_filename(md_file)
            page_slug = slugify(md_file.stem)[:50]
            body = md_file.read_text(encoding="utf-8")

            page, page_created = DocumentationPage.objects.update_or_create(
                slug=page_slug,
                defaults={
                    "source": source,
                    "title": title,
                    "source_path": str(md_file.relative_to(root.parent)),
                    "body_markdown": body,
                    "order": order,
                    "is_active": True,
                },
            )
            if page_created:
                pages_created += 1
            else:
                # Refresh chunks for updated pages
                page.chunks.all().delete()

            raw_chunks = _split_into_chunks(body)
            # Re-number chunks sequentially from 1 regardless of heading nesting
            for i, c in enumerate(raw_chunks, 1):
                c["order"] = i

            with transaction.atomic():
                if not page_created:
                    page.chunks.all().delete()
                bulk = [
                    DocumentationChunk(
                        page=page,
                        heading=c["heading"],
                        anchor=c["anchor"],
                        body_markdown=c["body"],
                        order=c["order"],
                        search_text=_strip_markdown(c["body"]),
                        token_estimate=_estimate_tokens(c["body"]),
                        is_active=True,
                    )
                    for c in raw_chunks
                ]
                DocumentationChunk.objects.bulk_create(bulk)
            chunks_created += len(bulk)

            self.stdout.write(f"  Imported: {md_file.name} ({len(raw_chunks)} chunks)")

        self.stdout.write(
            self.style.SUCCESS(
                f"Done. {pages_created} pages created, {chunks_created} chunks created."
            )
        )

        if options.get("embed"):
            self.stdout.write("\nGenerating embeddings...")
            from django.core.management import call_command
            call_command("embed_docs", model=options["embed_model"])
