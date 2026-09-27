from app.chunking import CHUNK_WORDS, chunk_document


def test_sections_stable_ids_and_metadata():
    text = "# Handbook\n\n## Leave\nTwenty five days.\n\n## Security\nUse MFA."
    first = chunk_document("handbook.md", text)
    assert first == chunk_document("handbook.md", text)
    assert [c.section for c in first] == ["Leave", "Security"]
    assert all(c.source == "handbook.md" for c in first)
    assert first[0].id != chunk_document("other.md", text)[0].id


def test_long_sections_overlap_without_losing_tail():
    words = [f"word{i}" for i in range(420)]
    chunks = chunk_document("long.txt", " ".join(words))
    assert all(len(c.text.split()) <= CHUNK_WORDS for c in chunks)
    assert set(" ".join(c.text for c in chunks).split()) == set(words)
    assert chunks[0].text.split()[-30:] == chunks[1].text.split()[:30]
    assert chunks[-1].text.endswith("word419")


def test_empty_and_heading_only_documents_have_no_chunks():
    assert chunk_document("empty.md", " \n# Empty\n") == []
