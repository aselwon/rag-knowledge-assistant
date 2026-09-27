import hashlib
import re
from dataclasses import dataclass

CHUNK_WORDS = 180
OVERLAP_WORDS = 30
CHUNKER_VERSION = "section-window-v1"


@dataclass
class Chunk:
    id: str
    source: str
    section: str
    text: str
    ordinal: int


def chunk_document(source: str, text: str) -> list[Chunk]:
    """Split at Markdown headings, then use overlapping word windows per section."""
    sections: list[tuple[str, str]] = []
    heading, lines = "Introduction", []
    for line in text.splitlines():
        match = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", line)
        if match:
            sections.append((heading, "\n".join(lines)))
            heading, lines = match.group(1), []
        else:
            lines.append(line)
    sections.append((heading, "\n".join(lines)))
    chunks = []
    for section, content in sections:
        words = content.split()
        for start in range(0, len(words), CHUNK_WORDS - OVERLAP_WORDS):
            body = " ".join(words[start : start + CHUNK_WORDS])
            ordinal = len(chunks)
            identity = f"{source}\0{section}\0{ordinal}\0{body}"
            chunk_id = hashlib.sha256(identity.encode()).hexdigest()[:24]
            chunks.append(Chunk(chunk_id, source, section, body, ordinal))
            if start + CHUNK_WORDS >= len(words):
                break
    return chunks
