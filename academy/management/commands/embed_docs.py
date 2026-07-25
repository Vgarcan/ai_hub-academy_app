"""
Generate semantic embeddings for all active DocumentationChunks.

Usage:
    python manage.py embed_docs
    python manage.py embed_docs --model bge-m3:latest
    python manage.py embed_docs --force        # re-embed already-embedded chunks
"""
from django.core.management.base import BaseCommand

from academy.models import DocumentationChunk
from academy.services.embeddings import get_embedding


class Command(BaseCommand):
    help = "Generate or refresh semantic embeddings for all active DocumentationChunks."

    def add_arguments(self, parser):
        parser.add_argument(
            "--model",
            default="bge-m3:latest",
            help="Ollama embedding model (default: bge-m3:latest).",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Re-embed chunks that already have embeddings.",
        )

    def handle(self, *args, **options):
        model = options["model"]
        force = options["force"]

        qs = DocumentationChunk.objects.filter(
            is_active=True,
            page__is_active=True,
            page__source__is_active=True,
        ).select_related("page")
        if not force:
            qs = qs.filter(embedding__isnull=True)

        total = qs.count()
        if not total:
            if force:
                self.stdout.write("No active documentation chunks found.")
            else:
                self.stdout.write(
                    "All active documentation chunks already embedded. "
                    "Use --force to refresh."
                )
            return

        self.stdout.write(f"Embedding {total} chunk(s) with '{model}'...")

        ok = failed = 0
        for chunk in qs:
            # Combine heading + body for richer semantic context
            text = f"{chunk.heading}\n\n{chunk.search_text or chunk.body_markdown[:3000]}"
            emb = get_embedding(text, model=model)
            if emb:
                chunk.embedding = emb
                chunk.save(update_fields=["embedding"])
                ok += 1
                self.stdout.write(f"  [ok] {chunk.page.title} / {chunk.heading}")
            else:
                failed += 1
                self.stderr.write(
                    self.style.WARNING(f"  [skip] {chunk.page.title} / {chunk.heading} — no embedding returned")
                )

        if failed:
            msg = f"Done. {ok} embedded, {failed} skipped (check Ollama is running with '{model}')."
            self.stdout.write(self.style.WARNING(msg))
        else:
            self.stdout.write(self.style.SUCCESS(f"Done. {ok} chunks embedded."))
