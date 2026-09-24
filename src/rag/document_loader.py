from pathlib import Path
from typing import Dict, List


def load_markdown_documents(knowledge_base_dir: str) -> List[Dict[str, str]]:
    """
    Load Markdown knowledge-base documents.

    Each document contains:
    - source: source filename
    - content: document text
    """

    base_path = Path(knowledge_base_dir)

    if not base_path.exists():
        raise FileNotFoundError(
            f"Knowledge-base directory not found: {base_path}"
        )

    documents = []

    for file_path in sorted(base_path.glob("*.md")):
        content = file_path.read_text(encoding="utf-8").strip()

        if not content:
            continue

        documents.append(
            {
                "source": file_path.name,
                "content": content,
            }
        )

    return documents


if __name__ == "__main__":
    knowledge_base_dir = "data/external/knowledge_base"

    documents = load_markdown_documents(knowledge_base_dir)

    print(f"Loaded documents: {len(documents)}")

    for document in documents:
        print(f"\nSource: {document['source']}")
        print(f"Characters: {len(document['content'])}")
        print(f"Preview:\n{document['content'][:300]}")