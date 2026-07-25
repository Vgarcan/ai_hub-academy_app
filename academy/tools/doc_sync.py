"""
Python Callable tool: sync_all_docs

Checks every Markdown file in the configured AI Hub and Academy documentation
roots against the database and syncs any file whose content has changed.
Returns a summary dict that the GAME agent receives as
`tool_results.sync_all_docs`.
"""
from academy.management.commands.import_academy_docs import (
    collect_doc_files,
    sync_documentation_files,
)


def sync_all_docs(payload: dict, config: dict) -> dict:
    doc_files = collect_doc_files()
    if not doc_files:
        return {
            "error": (
                "No Markdown files found in AIHUB_DOCS_SOURCE or "
                "ACADEMY_DOCS_SOURCE."
            ),
            "checked": 0, "synced": 0, "unchanged": 0, "details": [],
        }

    source_name = config.get("source_name", "AI Hub Official Docs")
    result = sync_documentation_files(
        doc_files,
        source_name=source_name,
        deactivate_missing=True,
    )
    synced = result["created"] + result["updated"]

    return {
        "checked": result["checked"],
        "synced": synced,
        "unchanged": result["unchanged"],
        "deactivated": result["deactivated"],
        "details": result["details"],
        "message": (
            f"Sync complete. {synced} file(s) synced, "
            f"{result['unchanged']} unchanged, "
            f"{result['deactivated']} deactivated."
        ),
    }
