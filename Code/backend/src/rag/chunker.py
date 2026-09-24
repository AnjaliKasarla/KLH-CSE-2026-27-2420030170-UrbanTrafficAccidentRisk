import re
from typing import Dict, List


def _split_into_sections(text: str) -> List[str]:
    """Split Markdown content at level-2 headings."""

    sections = re.split(r"(?m)(?=^##\s+)", text)

    return [
        section.strip()
        for section in sections
        if section.strip()
    ]


def _split_large_section(
    section: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    """
    Split a large section using paragraph boundaries first.

    Paragraphs are kept intact whenever possible. If a single
    paragraph is too large, it is split using sentence boundaries.
    """

    paragraphs = re.split(r"\n\s*\n", section)

    chunks = []
    current = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        candidate = (
            f"{current}\n\n{paragraph}".strip()
            if current
            else paragraph
        )

        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)

        # If one paragraph itself exceeds the limit,
        # split it using sentence boundaries.
        if len(paragraph) > chunk_size:
            sentences = re.split(
                r"(?<=[.!?])\s+",
                paragraph,
            )

            current = ""

            for sentence in sentences:
                sentence = sentence.strip()

                if not sentence:
                    continue

                candidate = (
                    f"{current} {sentence}".strip()
                    if current
                    else sentence
                )

                if len(candidate) <= chunk_size:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)

                    # Extremely long sentence fallback.
                    if len(sentence) > chunk_size:
                        for start in range(
                            0,
                            len(sentence),
                            chunk_size - chunk_overlap,
                        ):
                            piece = sentence[
                                start:start + chunk_size
                            ].strip()

                            if piece:
                                chunks.append(piece)

                        current = ""
                    else:
                        current = sentence

        else:
            current = paragraph

    if current:
        chunks.append(current)

    return chunks


def chunk_documents(
    documents: List[Dict[str, str]],
    chunk_size: int = 1200,
    chunk_overlap: int = 100,
) -> List[Dict[str, str]]:
    """
    Create clean, section-aware RAG chunks.

    Markdown sections are preserved.
    Paragraphs are preserved whenever possible.
    """

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0."
        )

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be >= 0 and smaller "
            "than chunk_size."
        )

    chunks = []

    for document in documents:
        source = document["source"]
        content = document["content"]

        sections = _split_into_sections(content)

        chunk_index = 0

        for section in sections:

            if len(section) <= chunk_size:
                section_chunks = [section]
            else:
                section_chunks = _split_large_section(
                    section,
                    chunk_size,
                    chunk_overlap,
                )

            for section_chunk in section_chunks:
                chunks.append(
                    {
                        "chunk_id": (
                            f"{source}::chunk_{chunk_index}"
                        ),
                        "source": source,
                        "content": section_chunk,
                    }
                )

                chunk_index += 1

    return chunks


if __name__ == "__main__":
    from document_loader import load_markdown_documents

    knowledge_base_dir = (
        "data/external/knowledge_base"
    )

    documents = load_markdown_documents(
        knowledge_base_dir
    )

    chunks = chunk_documents(documents)

    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")

    for chunk in chunks:
        print("\n" + "=" * 70)
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Source: {chunk['source']}")
        print(f"Characters: {len(chunk['content'])}")
        print(f"Preview:\n{chunk['content'][:500]}")