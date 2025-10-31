#!/usr/bin/env python3
"""Sanitize release title and notes by fixing common mojibake sequences.

Usage:
  sanitize_release_text.py --infile IN --outfile OUT
  sanitize_release_text.py --text "Some title"

This script replaces common mis-encoded sequences (like â€” -> —) and
performs Unicode normalization.
"""
import argparse
import unicodedata

REPLACEMENTS = [
    ("â€”", "—"),
    ("â€“", "–"),
    ("â€˜", "‘"),
    ("â€™", "’"),
    ("â€œ", "“"),
    ("â€", "”"),
        ("Â", "")
    , ("ÔÇô", "–")
    , ("ÔÇö", "—")
    , ("Ôé¼", "…")
]


def sanitize_text(s: str) -> str:
    # Heuristic: try to fix common double-encoding by attempting to reinterpret
    # the string as latin1/cp1252 bytes decoded as utf-8. Choose the candidate
    # that reduces the occurrence of typical mojibake markers (Ã, Â, Ô, �).
    def score(text: str) -> int:
        return sum(text.count(c) for c in ("Ã", "Â", "Ô", "�"))

    candidates = [s]
    try:
        candidates.append(s.encode("latin1").decode("utf-8"))
    except Exception:
        pass
    try:
        candidates.append(s.encode("cp1252").decode("utf-8"))
    except Exception:
        pass

    best = min(candidates, key=score)

    for a, b in REPLACEMENTS:
        best = best.replace(a, b)

    best = unicodedata.normalize("NFC", best)
    return best


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--infile")
    p.add_argument("--outfile")
    p.add_argument("--text")
    args = p.parse_args(argv)

    if args.text:
        print(sanitize_text(args.text))
        return 0

    if args.infile and args.outfile:
        with open(args.infile, "r", encoding="utf-8", errors="surrogateescape") as f:
            data = f.read()
        data2 = sanitize_text(data)
        # write as utf-8
        with open(args.outfile, "w", encoding="utf-8") as f:
            f.write(data2)
        print(args.outfile)
        return 0

    p.print_usage()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
