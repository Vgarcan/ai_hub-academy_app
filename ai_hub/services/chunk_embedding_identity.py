"""`k1` — the identity of the exact text a chunk vector is an embedding of.

`k1` answers exactly one question:

    What canonical TEXT INPUT is this vector an embedding of?

It is deliberately NOT document provenance, chunk-set identity, database
identity, collection identity, authorization, provider identity or model
identity. Model and vector-space identity is `e1` (S-18).

Future vector validity is the conjunction:

    current vector  =  current k1  AND  matching e1

**Why this lives in its own module.** `i1`, `c1`, `k1` and `e1` are four
separate contracts that must be versioned independently. Putting `k1` beside
`c1` in `knowledge_lifecycle.py` would invite a future edit to a shared helper
that silently moves two contracts at once - and moving `k1` invalidates every
vector ever produced, while moving `c1` changes what "chunks were modified"
means. They are not allowed to share a fate.

**The renderer here is load-bearing.** S-20's embedding execution must send
exactly the string `canonical_chunk_embedding_text()` returns. No other caller
may rebuild embedding text independently: two renderings that differ by one
newline are two different vector spaces wearing one fingerprint.
"""

import hashlib
import json
import unicodedata


K1_PREFIX = "k1:sha256:"


def _normalize_representation(value) -> str:
    """CRLF/CR -> LF, then Unicode NFC. Representation only.

    Deliberately does NOT strip, collapse whitespace, case-fold, trim or rewrite
    punctuation. Persisted whitespace is part of the semantic input to an
    embedding model: an indented code block and a flattened one genuinely embed
    differently, and pretending otherwise would make `k1` claim two different
    inputs are the same.

    A Windows/Unix round-trip is not an edit, and NFC/NFD are the same character
    sequence - those two are representation, which is why they are normalized
    and nothing else is. Mirrors `knowledge_lifecycle._normalize_representation`
    by intent, but is deliberately a separate implementation so the two
    contracts cannot be moved together by accident.
    """
    text = "" if value is None else str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", text)


def canonical_chunk_embedding_text(chunk) -> str:
    """The EXACT string a future embedding call must send for this chunk.

    Reads exactly two fields - `section_title` and `content` - and nothing else.
    The rendering is:

        section_title non-empty:   f"{section_title}\\n\\n{content}"
        section_title empty:       content

    No leading blank line is invented when there is no title. A whitespace-only
    section title is NOT empty and therefore stays in the text, because
    normalization is representation-only and never trims.

    **`document.title` is deliberately excluded.** `KnowledgeDocumentChunk` is
    the canonical retrieval unit; for the derived single-chunk generator the
    document title already BECOMES `section_title`, and for explicitly authored
    chunks the chunk's own semantic title is `section_title`. Reaching outside
    the canonical chunk to fold in mutable document context would make a vector
    depend on text nobody edited.

    Also excluded: `chunk_index`, chunk/document ids, collection name,
    ApplicationScope, tags, metadata, `token_estimate`, timestamps, authority
    mode, `i1`, `c1`, `e1`. If product requirements ever decide a title or tags
    should influence embeddings, that is a NEW k-contract version - never a
    silent addition to this one.
    """
    section_title = _normalize_representation(getattr(chunk, "section_title", ""))
    content = _normalize_representation(getattr(chunk, "content", ""))
    if section_title:
        return f"{section_title}\n\n{content}"
    return content


def chunk_embedding_input_fingerprint(embedding_text: str) -> str:
    """`k1:sha256:<64 hex>` over the canonical embedding text.

    Canonical JSON rather than hashing the raw string directly, matching the
    `i1`/`c1`/`e1` house convention: the envelope names the contract, so a bare
    digest can never be mistaken for a different contract's digest, and the
    encoding is pinned rather than implied.
    """
    payload = {
        "contract": "k1",
        "embedding_text": embedding_text,
    }
    serialized = json.dumps(
        payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return f"{K1_PREFIX}{digest}"


def chunk_embedding_fingerprint(chunk) -> str:
    """`k1` for one persisted chunk. The only way callers should compute it."""
    return chunk_embedding_input_fingerprint(canonical_chunk_embedding_text(chunk))
