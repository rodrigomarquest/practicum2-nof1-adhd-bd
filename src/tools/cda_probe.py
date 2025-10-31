import os
import re
import sys

path = "data/etl/P000001/snapshots/2025-09-29/extracted/apple/apple_health_export/export_cda.xml"
if not os.path.exists(path):
    print("ERROR: file not found:", path)
    sys.exit(1)
size = os.path.getsize(path)
print("File:", path)
print("Size bytes:", size)

TOKENS = [
    "mood",
    "state of mind",
    "pleasant",
    "unpleasant",
    "neutral",
    "anxious",
    "stressed",
    "happy",
    "sad",
    "angry",
    "valence",
    "affect",
]
pat = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in TOKENS) + r")\b", re.IGNORECASE
)
HEAD_BYTES = 2_000_000
TAIL_BYTES = 2_000_000


def scan_region(f, start, length):
    f.seek(start)
    data = f.read(length)
    try:
        s = data.decode("utf-8", errors="replace")
    except Exception:
        s = data.decode("latin1", errors="replace")
    matches = list(pat.finditer(s))
    return s, matches


head_len = min(HEAD_BYTES, size)
with open(path, "rb") as f:
    head_s, head_matches = scan_region(f, 0, head_len)
    if size > TAIL_BYTES:
        tail_start = max(0, size - TAIL_BYTES)
        tail_s, tail_matches = scan_region(f, tail_start, TAIL_BYTES)
    else:
        tail_s = ""
        tail_matches = []

print()
print("Head matches:", len(head_matches))
print("Tail matches:", len(tail_matches))
print("Total matches (head+tail):", len(head_matches) + len(tail_matches))

SHOW = 10
count = 0
print("\n--- First matches (head) ---")
for m in head_matches[:SHOW]:
    start = max(0, m.start() - 40)
    end = min(len(head_s), m.end() + 40)
    snippet = head_s[start:end].replace("\n", "\\n")
    print("...{}...".format(snippet))
    count += 1
    if count >= SHOW:
        break

if count < SHOW:
    print("\n--- First matches (tail) ---")
    for m in tail_matches[: SHOW - count]:
        start = max(0, m.start() - 40)
        end = min(len(tail_s), m.end() + 40)
        snippet = tail_s[start:end].replace("\n", "\\n")
        print("...{}...".format(snippet))
        count += 1
        if count >= SHOW:
            break

all_text = head_s + "\n" + tail_s
found = {t: 0 for t in TOKENS}
for mm in pat.finditer(all_text):
    tok = mm.group(1).lower()
    if tok in found:
        found[tok] += 1

print("\nToken counts (head+tail):")
for t in TOKENS:
    print("{}: {}".format(t, found[t]))
