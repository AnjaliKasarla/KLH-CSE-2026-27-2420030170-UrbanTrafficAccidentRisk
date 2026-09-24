import re
from typing import Dict, List


def _split_into_sections(text: str) -> List[str]:
    """Split Markdown content using level-2 headings."""

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
    """Split an oversized section while preserving overlap."""

    chunks = []

    start = 0

    while start < len(section):
        end = min(start + chunk_size, len(section))

        chunk = section[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(section):
            break

        start = end - chunk_overlap

    return chunks


def chunk_documents(
    documents: List[Dict[str, str]],
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
) -> List[Dict[str, str]]:
    """
    Create section-aware chunks from Markdown documents.

    Markdown sections are kept together whenever possible.
    Large sections are split using overlapping character windows.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be >= 0 and smaller than chunk_size."
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

    knowledge_base_dir = "data/external/knowledge_base"

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
        print(chunk["content"][:300])