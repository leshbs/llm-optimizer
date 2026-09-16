"""Repair the setup / data-loading / encoding cells of baseline.ipynb.

Three things are wrong or missing in the original cells:

1. The loader has no integrity check. The entire Stage 1 architecture (sections
   129 and 130) is built on the assumption that ``train_stage1.csv`` is
   corrupted. If that file were ever replaced with a clean one the notebook
   would silently keep using corruption-invariant features and quietly throw
   away the recovered text, so the load step now fingerprints the file and says
   which regime it is in.

2. Section 2.1 attributes the damage to "a round-trip through ASCII with
   errors='replace'". The byte evidence rules ASCII out: an ASCII round trip
   would also have destroyed the four surviving punctuation characters.

3. Section 2.1's conclusion states the text "cannot be recovered", which
   section 130 has since disproved for about half the subsection lines.

Run:  python src/fix_notebook_loading.py   (idempotent)
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parent.parent / "baseline.ipynb"

LOADER_ID = "54b5d085"
ENCODING_MD_ID = "9a5a5a86"
CONCLUSION_MD_ID = "c0ecb956"
FORENSICS_ID = "enc_forensics"
QMARK_ID = "c0603d99"

LOADER = '''# ---- 1. Data loading ----
# train_stage1.csv is the one damaged file. It is read as cp1251 for historical
# reasons; cp1252 gives a byte-identical result here (see 2.1), because the file
# contains only four bytes above 0x7F and the two code pages agree on all four.
# train_stage1.csv shipped damaged in the original download (a cp1252 export
# that replaced every Cyrillic character with '?'; see 2.1). A clean UTF-8 copy
# was later re-downloaded from the platform. The loader accepts either, so the
# notebook runs against both the historical and the repaired file.
ENCODING_CANDIDATES = ("utf-8", "cp1251")
EXPECTED_SHAPE = {
    "train_stage1.csv": (1767, 3),
    "test_stage1.csv": (442, 2),
    "train_stage2.csv": (690, 4),
    "test_stage2.csv": (173, 3),
}


def corruption_fingerprint(path):
    """Byte-level description of how badly a CSV has been mangled.

    ``replacement_ratio`` is the share of bytes that are a literal ASCII '?'.
    A healthy Russian-language CSV sits near 0; the damaged file sits near 0.78.
    """
    raw = Path(path).read_bytes()
    high = [b for b in raw if b > 0x7F]
    return {
        "bytes": len(raw),
        "replacement_ratio": raw.count(0x3F) / max(len(raw), 1),
        "high_byte_ratio": len(high) / max(len(raw), 1),
        "distinct_high_bytes": sorted(set(high)),
    }


def load_competition_data(data_dir=DATA_DIR, strict=True):
    """Load all four competition files and verify they are what we expect.

    Raises rather than warns on a shape mismatch: every downstream section
    hardcodes these row counts, so a silent change would invalidate the
    cached fold assignments and every reported CV number.
    """
    frames = {}
    for name in EXPECTED_SHAPE:
        for encoding in ENCODING_CANDIDATES:
            try:
                frames[name] = pd.read_csv(data_dir / name, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise UnicodeDecodeError(
                "none of %s decoded %s" % (ENCODING_CANDIDATES, name), b"", 0, 1, ""
            )
        if strict:
            expected = EXPECTED_SHAPE[name]
            got = frames[name].shape
            assert got == expected, f"{name}: expected {expected}, got {got}"
            assert not frames[name].isna().any().any(), f"{name}: unexpected nulls"
    for name in ("train_stage1.csv", "train_stage2.csv"):
        assert set(frames[name]["label"].unique()) <= {0, 1}, f"{name}: non-binary labels"
    return frames


_loaded_frames = load_competition_data()
train_s1 = _loaded_frames["train_stage1.csv"]
test_s1 = _loaded_frames["test_stage1.csv"]
train_s2 = _loaded_frames["train_stage2.csv"]
test_s2 = _loaded_frames["test_stage2.csv"]

print("Stage 1:", train_s1.shape, test_s1.shape)
print("Stage 2:", train_s2.shape, test_s2.shape)

# Guard rail: sections 129 and 130 only make sense while train_stage1 is
# damaged. If a clean file is ever dropped in, say so loudly rather than
# silently training a corruption-invariant model on readable text.
_enc_fp = corruption_fingerprint(DATA_DIR / "train_stage1.csv")
CORRUPTED_TRAIN_S1 = _enc_fp["replacement_ratio"] > 0.5
print(
    "\\ntrain_stage1.csv: %.1f%% of bytes are literal '?', %d distinct bytes > 0x7F"
    % (100 * _enc_fp["replacement_ratio"], len(_enc_fp["distinct_high_bytes"]))
)
print(
    "  -> %s"
    % (
        "DAMAGED, as expected by sections 2.1 / 129 / 130"
        if CORRUPTED_TRAIN_S1
        else "READABLE (repaired copy) -- section 131 is the Stage 1 path"
    )
)

# The Stage 2 protocols came out of the same export, which enforced a
# 32,767-character cell limit; the longest ones are cut off mid-field. It does
# not change any modelling decision (section 132 measures that), but it should
# be visible at load time rather than rediscovered later.
CELL_LIMIT_TRUNCATION = 32765
_trunc = {
    name: int((frames_s2["protocol_text"].astype(str).str.len() == CELL_LIMIT_TRUNCATION).sum())
    for name, frames_s2 in (("train_stage2.csv", train_s2), ("test_stage2.csv", test_s2))
}
print("
protocol_text truncated at the %d-character export limit:" % CELL_LIMIT_TRUNCATION)
for _name, _n in _trunc.items():
    _total = len(train_s2) if _name.startswith("train") else len(test_s2)
    print("  %-17s %3d / %3d rows (%.1f%%)" % (_name, _n, _total, 100 * _n / _total))'''

QMARK = '''# ---- 2.1: how much text was lost, and is it back? ----
def qmark_ratio(text: str) -> float:
    return text.count("?") / max(len(text), 1)


_archived = Path("data_archive") / "train_stage1_corrupted_original.csv"
if _archived.exists():
    _damaged = pd.read_csv(_archived, encoding="cp1251")
    _r = _damaged["title_text"].map(qmark_ratio)
    print("original download      : '?' share of title_text  mean %.1f%%, min %.1f%%, max %.1f%%"
          % (100 * _r.mean(), 100 * _r.min(), 100 * _r.max()))
    print("  example:", _damaged["title_text"].iloc[0].splitlines()[0])
    print()

_r_new = train_s1["title_text"].map(qmark_ratio)
_r_test = test_s1["title_text"].map(qmark_ratio)
print("repaired train_stage1  : '?' share mean %.2f%%  -> text is readable" % (100 * _r_new.mean()))
print("  example:", train_s1["title_text"].iloc[0].splitlines()[0])
print()
print("test_stage1 (never damaged): '?' share mean %.2f%%" % (100 * _r_test.mean()))
print("  example:", test_s1["title_text"].iloc[0].splitlines()[0])'''

FORENSICS = '''# ---- 2.1a: what destroyed the original file? ----
# The live train_stage1.csv has since been re-downloaded and is clean, so the
# forensics run against the archived copy of the original damaged download.
ARCHIVED_CORRUPT = Path("data_archive") / "train_stage1_corrupted_original.csv"

if not ARCHIVED_CORRUPT.exists():
    print("archived corrupted original not present; skipping forensics")
else:
    _enc_raw = ARCHIVED_CORRUPT.read_bytes()
    _enc_high = sorted(set(b for b in _enc_raw if b > 0x7F))
    print("archived original : %d bytes" % len(_enc_raw))
    print("literal '?' bytes : %.1f%% of the file" % (100 * _enc_raw.count(0x3F) / len(_enc_raw)))
    print("bytes above 0x7F  : %d occurrences, %d distinct -> %s"
          % (sum(1 for b in _enc_raw if b > 0x7F), len(_enc_high), [hex(b) for b in _enc_high]))
    print("they decode to    : %r" % "".join(bytes([b]).decode("cp1251") for b in _enc_high))
    print()

    _enc_probe = "".join(chr(c) for c in (0x2013, 0x2014, 0x00AB, 0x00BB))  # the survivors
    _enc_cyr = "".join(chr(c) for c in (0x41F, 0x440, 0x438, 0x432, 0x435, 0x442))
    print("%-10s %-14s %s" % ("codec", "survivors", "Cyrillic"))
    for _enc_name in ("cp1252", "latin-1", "ascii", "cp1251", "koi8-r"):
        _enc_p = _enc_probe.encode(_enc_name, errors="replace").hex(" ")
        _enc_c = _enc_cyr.encode(_enc_name, errors="replace")
        print("%-10s %-14s %s"
              % (_enc_name, _enc_p, "all '?'" if _enc_c == b"?" * len(_enc_cyr) else "preserved"))
    print()
    print("-> cp1252 is the only one that keeps the punctuation AND destroys Cyrillic.")

    # The repaired file proves the substitution was character-for-character:
    # masking its Cyrillic reproduces the archived corrupted text almost exactly.
    _old = pd.read_csv(ARCHIVED_CORRUPT, encoding="cp1251").set_index("id")
    _new = train_s1.set_index("id")
    _cyr_re = re.compile("[" + chr(0x410) + "-" + chr(0x44F) + chr(0x401) + chr(0x451) + "]")
    _same = sum(_cyr_re.sub("?", _new.title_text[i]) == _old.title_text[i] for i in _new.index)
    print()
    print("ids identical            :", list(_new.index) == list(_old.index))
    print("labels identical         :", (_new.label == _old.label).all())
    print("mask(repaired) == damaged: %d / %d rows" % (_same, len(_new)))'''

ENCODING_MD = """### 2.1 Encoding: a damaged download, diagnosed and replaced

The original `train_stage1.csv` would not load as UTF-8 and, once read as
`cp1251`, turned out to have lost its Cyrillic entirely: **77.7% of every byte
in the file was a literal ASCII `?` (0x3F)**.

**The damage was a cp1252 export with `errors="replace"`** -- not the "ASCII
round trip" earlier versions of this section claimed. The cell above reproduces
the evidence from the archived copy:

* only **44 bytes** in the whole file exceeded 0x7F, taking exactly four
  values, the en dash, em dash and the two guillemets;
* `cp1252` is the only common codec that *keeps* those four while replacing
  every Cyrillic character -- `ascii` and `latin-1` would have replaced the
  dashes too, which rules an ASCII round trip out;
* the substitution was character-for-character, with no truncation.

Nothing in this repository caused it: no cell and no module writes
`train_stage1.csv`, and the first commit already contained the damaged file. It
arrived with the download.

**The file has since been re-downloaded from the competition platform and is now
clean UTF-8.** It is the same dataset -- identical 1,767 ids in identical order,
identical labels, identical 0.2168 positive rate -- and masking the repaired
text reproduces the archived damaged text for 1,744 of 1,767 rows. The 23
exceptions differ only in characters cp1252 also could not encode (a zero-width
space, a Greek beta, a combining breve, a minus sign), which is further
confirmation of the diagnosis.

The damaged original is kept at
`data_archive/train_stage1_corrupted_original.csv` so this section stays
reproducible, and the loader accepts either file.

**Consequence for everything that follows.** Sections 129 and 130 exist purely
to work around this corruption -- word-shape n-grams, masked feature spaces,
corpus-matching decoders. With real text available they are obsolete, and
section 131 replaces them with a straightforward lexical model that scores
substantially better. They are kept as an experimental record, not as the
shipping path."""

CONCLUSION_MD = """**Conclusion.** With the repaired file in place, `train_stage1.title_text` is
ordinary readable Russian and Stage 1 becomes a normal text-classification
problem: lemmatised word n-grams over the subsection line and its hierarchy,
char n-grams for morphological robustness, and structural plus
patient-specificity features. Section 131 does that and reaches a nested
cross-validated M1 of **0.8434**, against 0.7527 for the decoder-based model and
0.7111 for the shape-based one.

The corruption-era analysis is kept below because it explains why the earlier
Stage 1 models look the way they do, and because the transfer failure it
uncovered is a real finding: a lexical TF-IDF fit on the damaged text put 0.9468
of its weight on `?`-bearing n-grams that could never fire at inference, against
0.0000 of the test mass (section 129). That asymmetry, rather than the
corruption in itself, is what made the original Stage 1 model underperform."""


def replace_source(cell: dict, text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = nb["cells"]
    by_id = {str(c.get("id", "")): c for c in cells}

    missing = [i for i in (LOADER_ID, ENCODING_MD_ID, CONCLUSION_MD_ID, QMARK_ID) if i not in by_id]
    if missing:
        raise SystemExit("cells not found, notebook structure changed: %s" % missing)

    replace_source(by_id[LOADER_ID], LOADER)
    replace_source(by_id[QMARK_ID], QMARK)
    replace_source(by_id[ENCODING_MD_ID], ENCODING_MD)
    replace_source(by_id[CONCLUSION_MD_ID], CONCLUSION_MD)

    # Insert the forensics cell right after the '?'-ratio cell, once.
    cells[:] = [c for c in cells if str(c.get("id", "")) != FORENSICS_ID]
    idx = next(i for i, c in enumerate(cells) if str(c.get("id", "")) == QMARK_ID)
    cells.insert(
        idx + 1,
        {
            "cell_type": "code",
            "id": FORENSICS_ID,
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": FORENSICS.splitlines(keepends=True),
        },
    )

    NOTEBOOK.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("patched loader, encoding section, conclusion; inserted forensics cell")
    print("total cells now %d" % len(cells))


if __name__ == "__main__":
    main()
