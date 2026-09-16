"""Encoding-tolerant loading for the competition CSVs.

``train_stage1.csv`` shipped as a cp1252 export that replaced every Cyrillic
character with '?', and was later re-downloaded as clean UTF-8. Both versions
exist in the wild -- the damaged one is archived under ``data_archive/`` -- so
every read site tries UTF-8 first and falls back to cp1251, rather than
hardcoding an encoding that is right for only one of them.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ENCODING_CANDIDATES = ("utf-8", "cp1251")


def read_competition_csv(path, **kwargs) -> pd.DataFrame:
    """``pd.read_csv`` that works on both the damaged and the repaired files."""
    last = None
    for encoding in ENCODING_CANDIDATES:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            last = exc
    raise UnicodeDecodeError(
        "none of %s decoded %s (%s)" % (ENCODING_CANDIDATES, path, last), b"", 0, 1, ""
    )


def is_corrupted(path) -> bool:
    """True when the file is the damaged cp1252 export."""
    raw = Path(path).read_bytes()
    return raw.count(0x3F) / max(len(raw), 1) > 0.5
