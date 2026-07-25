"""
Import Markdown documentation files into the database.

By default it reads two roots: the canonical reusable-platform docs
(AIHUB_DOCS_SOURCE) and the academy-specific docs (ACADEMY_DOCS_SOURCE). This
keeps a single source of truth — the platform docs are not duplicated into
docs_source.

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
    (``AIHUB_DOCS_SOURCE``) and then the academy-specific docs
    (``ACADEMY_DOCS_SOURCE``). The platform ``README`` is skipped (it is a
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


def sync_documentation_files(
    doc_files,
    *,
    source_name: str,
    deactivate_missing: bool = False,
):
    """Synchronize an ordered documentation inventory without needless churn.

    Unchanged pages keep their chunks and embeddings. Content changes replace
    the page chunks atomically, while metadata-only changes preserve them.
    ``deactivate_missing`` is safe only for a complete inventory of the source.
    """
    source_slug = slugify(source_name)
    source, source_created = DocumentationSource.objects.get_or_create(
        slug=source_slug,
        defaults={"name": source_name, "is_active": True},
    )
    source_updates = []
    if source.name != source_name:
        source.name = source_name
        source_updates.append("name")
    if not source.is_active:
        source.is_active = True
        source_updates.append("is_active")
    if source_updates:
        source.save(update_fields=source_updates)

    result = {
        "source_created": source_created,
        "checked": 0,
        "created": 0,
        "updated": 0,
        "unchanged": 0,
        "deactivated": 0,
        "chunks_created": 0,
        "details": [],
    }
    current_slugs = []

    for order, (md_file, root) in enumerate(doc_files, start=1):
        result["checked"] += 1
        title = _page_title_from_filename(md_file)
        page_slug = slugify(md_file.stem)[:50]
        current_slugs.append(page_slug)
        body = md_file.read_text(encoding="utf-8")
        source_path = str(md_file.relative_to(root.parent))
        existing = DocumentationPage.objects.filter(slug=page_slug).first()

        metadata = {
            "source": source,
            "title": title,
            "source_path": source_path,
            "order": order,
            "is_active": True,
        }
        content_changed = existing is None or existing.body_markdown != body
        metadata_changed = existing is not None and any(
            getattr(existing, field) != value
            for field, value in metadata.items()
        )

        if existing is not None and not content_changed and not metadata_changed:
            result["unchanged"] += 1
            result["details"].append(
                {
                    "file": md_file.name,
                    "status": "unchanged",
                    "chunks": existing.chunks.count(),
                }
            )
            continue

        with transaction.atomic():
            page, page_created = DocumentationPage.objects.update_or_create(
                slug=page_slug,
                defaults={**metadata, "body_markdown": body},
            )
            chunks_created = 0
            if content_changed:
                page.chunks.all().delete()
                raw_chunks = _split_into_chunks(body)
                bulk = [
                    DocumentationChunk(
                        page=page,
                        heading=chunk["heading"],
                        anchor=chunk["anchor"],
                        body_markdown=chunk["body"],
                        order=index,
                        search_text=_strip_markdown(chunk["body"]),
                        token_estimate=_estimate_tokens(chunk["body"]),
                        is_active=True,
                    )
                    for index, chunk in enumerate(raw_chunks, start=1)
                ]
                DocumentationChunk.objects.bulk_create(bulk)
                chunks_created = len(bulk)

        status = "created" if page_created else "updated"
        result[status] += 1
        result["chunks_created"] += chunks_created
        result["details"].append(
            {
                "file": md_file.name,
                "status": status,
                "chunks": page.chunks.count(),
            }
        )

    if deactivate_missing:
        stale_pages = DocumentationPage.objects.filter(source=source, is_active=True)
        if current_slugs:
            stale_pages = stale_pages.exclude(slug__in=current_slugs)
        result["deactivated"] = stale_pages.update(is_active=False)

    return result


class Command(BaseCommand):
    help = "Import Markdown files from a directory into the documentation database."

    def add_arguments(self, parser):
        parser.add_argument(
            "--path",
            type=str,
            default=None,
            help=(
                "Directory containing .md files. By default both "
                "AIHUB_DOCS_SOURCE and ACADEMY_DOCS_SOURCE are imported."
            ),
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

        result = sync_documentation_files(
            doc_files,
            source_name=options["source_name"],
            deactivate_missing=options["path"] is None,
        )
        if result["source_created"]:
            self.stdout.write(f"Created source: {options['source_name']}")
        for detail in result["details"]:
            self.stdout.write(
                f"  {detail['status'].title()}: {detail['file']} "
                f"({detail['chunks']} chunks)"
            )

        self.stdout.write(
            self.style.SUCCESS(
                "Done. "
                f"{result['created']} pages created, "
                f"{result['updated']} updated, "
                f"{result['unchanged']} unchanged, "
                f"{result['deactivated']} deactivated, "
                f"{result['chunks_created']} chunks created."
            )
        )

        if options.get("embed"):
            self.stdout.write("\nGenerating embeddings...")
            from django.core.management import call_command
            call_command("embed_docs", model=options["embed_model"])
