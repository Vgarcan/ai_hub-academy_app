"""
Python Callable tool: sync_all_docs

Checks every Markdown file in ACADEMY_DOCS_SOURCE against the database
and syncs any file whose content has changed.  Returns a summary dict
that the GAME agent receives as `tool_results.sync_all_docs`.
"""
from pathlib import Path

from django.conf import settings
from django.utils.text import slugify

from academy.models import DocumentationChunk, DocumentationPage, DocumentationSource
from academy.management.commands.import_academy_docs import (
    _estimate_tokens,
    _page_title_from_filename,
    _split_into_chunks,
    _strip_markdown,
)


def sync_all_docs(payload: dict, config: dict) -> dict:
    base_path = getattr(settings, "ACADEMY_DOCS_SOURCE", None)
    if not base_path:
        return {
            "error": "ACADEMY_DOCS_SOURCE is not configured in settings.",
            "checked": 0, "synced": 0, "unchanged": 0, "details": [],
        }

    base_path = Path(base_path)
    if not base_path.exists():
        return {
            "error": f"docs_source path does not exist: {base_path}",
            "checked": 0, "synced": 0, "unchanged": 0, "details": [],
        }

    md_files = sorted(base_path.glob("**/*.md"))
    if not md_files:
        return {
            "checked": 0, "synced": 0, "unchanged": 0, "details": [],
            "message": "No Markdown files found in docs_source.",
        }

    source_name = config.get("source_name", "AI Hub Official Docs")
    source_slug = slugify(source_name)
    source, _ = DocumentationSource.objects.get_or_create(
        slug=source_slug,
        defaults={"name": source_name, "is_active": True},
    )

    checked = synced = unchanged = 0
    details = []

    for order, md_file in enumerate(md_files, start=1):
        checked += 1
        page_slug = slugify(md_file.stem)[:49]
        file_content = md_file.read_text(encoding="utf-8")

        existing = DocumentationPage.objects.filter(slug=page_slug).first()
        if existing and existing.body_markdown == file_content:
            unchanged += 1
            details.append({"file": md_file.name, "status": "unchanged"})
            continue

        title = _page_title_from_filename(md_file)
        page, _ = DocumentationPage.objects.update_or_create(
            slug=page_slug,
            defaults={
                "source": source,
                "title": title,
                "source_path": str(md_file.relative_to(base_path.parent)),
                "body_markdown": file_content,
                "order": order,
                "is_active": True,
            },
        )
        page.chunks.all().delete()
        raw_chunks = _split_into_chunks(file_content)
        for chunk_data in raw_chunks:
            DocumentationChunk.objects.create(
                page=page,
                heading=chunk_data["heading"],
                anchor=chunk_data["anchor"],
                body_markdown=chunk_data["body"],
                order=chunk_data["order"],
                search_text=_strip_markdown(chunk_data["body"]),
                token_estimate=_estimate_tokens(chunk_data["body"]),
                is_active=True,
            )

        synced += 1
        details.append({
            "file": md_file.name,
            "status": "created" if not existing else "updated",
            "chunks": len(raw_chunks),
        })

    return {
        "checked": checked,
        "synced": synced,
        "unchanged": unchanged,
        "details": details,
        "message": f"Sync complete. {synced} file(s) synced, {unchanged} unchanged.",
    }
