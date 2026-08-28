"""
Shingle extraction, S(D, w), for the five data modalities used in the paper.

    modality        shingle type          datasets
    -------------------------------------------------------------------
    bytes           byte sequence         Linux memory dumps, PDF files
    words           word sequence         fake news
    characters      character sequence    DNA (3-nucleotide "words")
    image_2d        2-D pixel window      chest X-ray (partition MinHash)

Every extractor takes a shingle length and an overlap and returns an iterable
of ``bytes`` objects, which is the input expected by the Rabin fingerprint
functions. The stride is ``length - overlap``; a larger overlap therefore
produces more, more strongly correlated shingles. The two overlap settings per
dataset used in Experiment 2 are listed in ``configs/datasets.yaml``.
"""

from __future__ import annotations

import re
from typing import Iterable, Iterator, Sequence

import numpy as np

__all__ = [
    "byte_shingles",
    "word_shingles",
    "character_shingles",
    "image_shingles",
    "extract_shingles",
]


def _stride(length: int, overlap: int) -> int:
    if overlap >= length:
        raise ValueError("overlap must be strictly smaller than the shingle length")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    return length - overlap


# --------------------------------------------------------------------------- #
def byte_shingles(data: bytes, length: int, overlap: int) -> Iterator[bytes]:
    """Contiguous byte windows; used for memory dumps and PDF files."""
    step = _stride(length, overlap)
    for start in range(0, max(len(data) - length + 1, 0), step):
        yield data[start:start + length]


def word_shingles(text: str, length: int, overlap: int,
                  lowercase: bool = True, strip_punctuation: bool = True) -> Iterator[bytes]:
    """Word n-grams; used for the fake news dataset (length 4, overlap 3 or 2).

    The pre-processing matches Table 1: lowercasing and punctuation removal.
    """
    if lowercase:
        text = text.lower()
    if strip_punctuation:
        text = re.sub(r"[^\w\s]", " ", text)
    words = text.split()
    step = _stride(length, overlap)
    for start in range(0, max(len(words) - length + 1, 0), step):
        yield " ".join(words[start:start + length]).encode("utf-8")


def character_shingles(text: str, length: int, overlap: int) -> Iterator[bytes]:
    """Character windows; used for nucleotide sequences.

    For the DNA dataset the manuscript treats every triplet of nucleotides as a
    "word" (an amino-acid codon), so ``length`` and ``overlap`` are multiples of
    three (252 and 81 / 27 nucleotides respectively).
    """
    step = _stride(length, overlap)
    for start in range(0, max(len(text) - length + 1, 0), step):
        yield text[start:start + length].encode("utf-8")


def image_shingles(image: np.ndarray, size: int, overlap: int) -> Iterator[bytes]:
    """Square pixel windows of a grayscale image, scanned row by row.

    The chest X-ray sketches use windows of 128x128 pixels with an overlap of
    32 (Experiment 1) or 64 (Experiment 2) pixels.
    """
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("image must be a 2-D grayscale array")
    step = _stride(size, overlap)
    height, width = image.shape
    for row in range(0, max(height - size + 1, 0), step):
        for col in range(0, max(width - size + 1, 0), step):
            yield np.ascontiguousarray(image[row:row + size, col:col + size]).tobytes()


def image_partitions(image: np.ndarray, n_rows: int = 2, n_cols: int = 2,
                     overlap_fraction: float = 0.25) -> Iterator[np.ndarray]:
    """Overlapping partitions of an image, following partition MinHash [4].

    Each partition is hashed independently, so that duplicated *regions* are
    detected rather than whole-image duplicates only.
    """
    image = np.asarray(image)
    height, width = image.shape
    ph, pw = height // n_rows, width // n_cols
    oh, ow = int(ph * overlap_fraction), int(pw * overlap_fraction)
    for r in range(n_rows):
        for c in range(n_cols):
            top = max(r * ph - oh, 0)
            left = max(c * pw - ow, 0)
            yield image[top:min(top + ph + 2 * oh, height),
                        left:min(left + pw + 2 * ow, width)]


# --------------------------------------------------------------------------- #
def extract_shingles(item, modality: str, length: int, overlap: int) -> Iterable[bytes]:
    """Dispatch to the extractor matching ``modality``.

    Parameters
    ----------
    modality : {"bytes", "words", "characters", "image_2d"}
    """
    modality = modality.lower()
    if modality == "bytes":
        return byte_shingles(item, length, overlap)
    if modality == "words":
        return word_shingles(item, length, overlap)
    if modality == "characters":
        return character_shingles(item, length, overlap)
    if modality == "image_2d":
        return image_shingles(item, length, overlap)
    raise ValueError(f"Unknown modality '{modality}'")
