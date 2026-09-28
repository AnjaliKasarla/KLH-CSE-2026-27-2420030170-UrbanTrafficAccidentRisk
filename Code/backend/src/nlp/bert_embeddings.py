"""
DistilBERT embedding generation for the Urban Traffic Accident project.

Embeddings are generated from the derived Accident_Context_Text
representation.

Features:
- CPU/GPU device detection
- Batch inference
- 768-dimensional embeddings
- float32 storage
- NumPy memory-mapped storage
- Checkpoint/resume support
- Safe interruption handling
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


DEFAULT_MODEL_NAME = "distilbert-base-uncased"
DEFAULT_MAX_LENGTH = 128
DEFAULT_BATCH_SIZE = 8

EMBEDDING_DIMENSION = 768


class DistilBERTEmbedder:
    """Generate contextual embeddings using DistilBERT."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        max_length: int = DEFAULT_MAX_LENGTH,
        device: str | None = None,
    ) -> None:

        self.model_name = model_name
        self.max_length = max_length

        if device is None:
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = torch.device(device)

        print(
            f"Loading model: {self.model_name}"
        )

        print(
            f"Using device: {self.device}"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )

        self.model = AutoModel.from_pretrained(
            self.model_name
        )

        self.model.to(self.device)
        self.model.eval()

    def encode_batch(
        self,
        texts: List[str],
    ) -> np.ndarray:
        """
        Generate DistilBERT embeddings for one batch.

        Returns:
            float32 NumPy array with shape:
            (batch_size, 768)
        """

        if not texts:
            return np.empty(
                (0, EMBEDDING_DIMENSION),
                dtype=np.float32,
            )

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = self.model(
                **inputs
            )

            # First-token representation.
            embeddings = (
                outputs.last_hidden_state[:, 0, :]
            )

        return embeddings.cpu().numpy().astype(
            np.float32
        )

    def encode_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "Accident_Context_Text",
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> np.ndarray:
        """
        Generate embeddings for a dataframe.

        Intended for small datasets/tests.

        For large datasets use:
            save_dataframe_embeddings()
        """

        if text_column not in df.columns:
            raise ValueError(
                f"Text column '{text_column}' not found."
            )

        texts = (
            df[text_column]
            .fillna("")
            .astype(str)
            .tolist()
        )

        all_embeddings = []

        total = len(texts)

        for start in range(
            0,
            total,
            batch_size,
        ):

            end = min(
                start + batch_size,
                total,
            )

            batch_embeddings = (
                self.encode_batch(
                    texts[start:end]
                )
            )

            all_embeddings.append(
                batch_embeddings
            )

            print(
                f"Processed {end:,}/{total:,} texts"
            )

        if not all_embeddings:
            return np.empty(
                (0, EMBEDDING_DIMENSION),
                dtype=np.float32,
            )

        return np.vstack(
            all_embeddings
        )


def save_embeddings(
    embeddings: np.ndarray,
    output_path: str | Path,
) -> None:
    """Save an embedding matrix as a NumPy file."""

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        output_path,
        embeddings,
    )


def save_dataframe_embeddings(
    df: pd.DataFrame,
    output_path: str | Path,
    text_column: str = "Accident_Context_Text",
    batch_size: int = DEFAULT_BATCH_SIZE,
    checkpoint_path: str | Path | None = None,
) -> None:
    """
    Generate and incrementally save embeddings.

    Uses NumPy memmap so the complete embedding matrix does not
    need to remain in RAM.

    A checkpoint stores the next row that needs to be processed.
    If the process is interrupted, running the function again
    resumes from that position.
    """

    if text_column not in df.columns:
        raise ValueError(
            f"Text column '{text_column}' not found."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if checkpoint_path is None:
        checkpoint_path = output_path.with_suffix(
            ".checkpoint"
        )

    checkpoint_path = Path(
        checkpoint_path
    )

    total = len(df)

    # --------------------------------------------------
    # Determine resume position
    # --------------------------------------------------

    start_index = 0

    if checkpoint_path.exists():

        try:

            start_index = int(
                checkpoint_path.read_text(
                    encoding="utf-8"
                ).strip()
            )

            if start_index < 0:
                start_index = 0

            if start_index > total:
                start_index = total

            print(
                f"Checkpoint found."
            )

            print(
                f"Resuming from "
                f"{start_index:,}/{total:,}"
            )

        except ValueError:

            print(
                "Invalid checkpoint found."
            )

            print(
                "Starting from row 0."
            )

            start_index = 0

    else:

        print(
            "No checkpoint found."
        )

        print(
            "Starting from row 0."
        )

    # --------------------------------------------------
    # Open/create memory-mapped file
    # --------------------------------------------------

    if output_path.exists():

        print(
            f"Opening existing embedding file:"
        )

        print(
            output_path
        )

        mode = "r+"

    else:

        print(
            "Creating new embedding file:"
        )

        print(
            output_path
        )

        mode = "w+"

    embeddings = np.memmap(
        output_path,
        dtype=np.float32,
        mode=mode,
        shape=(
            total,
            EMBEDDING_DIMENSION,
        ),
    )

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------

    embedder = DistilBERTEmbedder()

    texts = (
        df[text_column]
        .fillna("")
        .astype(str)
        .tolist()
    )

    # --------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------

    try:

        for start in range(
            start_index,
            total,
            batch_size,
        ):

            end = min(
                start + batch_size,
                total,
            )

            batch_texts = texts[
                start:end
            ]

            batch_embeddings = (
                embedder.encode_batch(
                    batch_texts
                )
            )

            embeddings[
                start:end
            ] = batch_embeddings

            embeddings.flush()

            # Save the NEXT row to process.
            checkpoint_path.write_text(
                str(end),
                encoding="utf-8",
            )

            print(
                f"Saved "
                f"{end:,}/{total:,} embeddings"
            )

    except KeyboardInterrupt:

        embeddings.flush()

        print(
            "\nEmbedding generation interrupted."
        )

        if checkpoint_path.exists():

            resume_position = (
                checkpoint_path.read_text(
                    encoding="utf-8"
                ).strip()
            )

            print(
                f"Resume position: "
                f"{resume_position}"
            )

        print(
            "Run the same command again "
            "to resume."
        )

        raise

    finally:

        embeddings.flush()

        del embeddings

    # --------------------------------------------------
    # Successful completion
    # --------------------------------------------------

    if checkpoint_path.exists():

        checkpoint_path.unlink()

    print(
        "\nEmbedding generation completed "
        "successfully."
    )

    print(
        f"Total embeddings: {total:,}"
    )

    print(
        f"Embedding dimension: "
        f"{EMBEDDING_DIMENSION}"
    )