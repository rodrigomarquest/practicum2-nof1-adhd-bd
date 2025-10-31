from pathlib import Path
import re, sys

def patch_atomic_writer_in_pipeline(p: Path):
    s = p.read_text(encoding="utf-8")
    # Corrigir definição da função _write_atomic_csv para chamar df.to_csv
    s = re.sub(
        r"def _write_atomic_csv\([^\)]*\):.*?finally:\s*[\s\S]*?pass\s*\n",
        (
            "def _write_atomic_csv(df, out_path: str):\n"
            "    import os, tempfile\n"
            "    d = os.path.dirname(out_path) or \".\"\n"
            "    fd, tmp = tempfile.mkstemp(prefix=\".tmp_\", dir=d, suffix=\".csv\")\n"
            "    os.close(fd)\n"
            "    try:\n"
            "        df.to_csv(tmp, index=False)\n"
            "        os.replace(tmp, out_path)\n"
            "    finally:\n"
            "        if os.path.exists(tmp):\n"
            "            try: os.remove(tmp)\n"
            "            except: pass\n"
            "\n"
        ),
        flags=re.DOTALL
    )
    # Remover qualquer ocorrência acidental de _write_atomic_csv(df, tmp) dentro da função
    s = s.replace("_write_atomic_csv(df, tmp)", "df.to_csv(tmp, index=False)")
    # Limpar possíveis encoding/future duplicados após o fim
    lines = s.splitlines()
    seen_future = False
    seen_encoding = False
    out = []
    for i,ln in enumerate(lines):
        if ln.strip().startswith("# -*- coding:"):
            if seen_encoding and i != 0:
                continue
            seen_encoding = True
        if ln.strip().startswith("from __future__ import annotations"):
            if seen_future and i != 1:
                continue
            seen_future = True
        out.append(ln)
    p.write_text("\n".join(out) + "\n", encoding="utf-8")

def patch_cardio_features(p: Path):
    s = p.read_text(encoding="utf-8")
    # Remover placeholders e escapes
    s = s.replace("__CSV__(", "to_csv(").replace("\\)", ")")
    # Se houver função _write_atomic_csv nossa anterior, opcionalmente usar to_csv simples aqui
    p.write_text(s, encoding="utf-8")

def main():
    pl = Path("etl_pipeline.py")
    cf = Path("etl_modules/cardiovascular/cardio_features.py")
    if pl.exists():
        patch_atomic_writer_in_pipeline(pl)
        print("Patched:", pl)
    else:
        print("WARN: etl_pipeline.py not found")
    if cf.exists():
        patch_cardio_features(cf)
        print("Patched:", cf)
    else:
        print("WARN: cardio_features.py not found")

if __name__ == "__main__":
    main()
