#!/usr/bin/env python3
import os, re, csv, plistlib, json
from datetime import datetime

PLIST_DIR = r"C:\dev\practicum2-nof1-adhd-bd\decrypted_output\screentime_plists"
OUT_CSV   = os.path.join(PLIST_DIR, "usage_daily_from_plists.csv")

DATE_PATTERNS = [
    re.compile(r"^\d{4}-\d{2}-\d{2}$"),         # 2025-10-17
    re.compile(r"^\d{4}/\d{2}/\d{2}$"),         # 2025/10/17
    re.compile(r"^\d{4}\.\d{2}\.\d{2}$"),       # 2025.10.17
]

def norm_date(s: str) -> str | None:
    s = str(s)
    for pat in DATE_PATTERNS:
        if pat.match(s):
            s2 = s.replace("/", "-").replace(".", "-")
            try:
                return datetime.strptime(s2, "%Y-%m-%d").date().isoformat()
            except Exception:
                return None
    return None

def is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)

def walk(obj, path, acc, source_plist):
    """
    Procura por estruturas de uso diário. Heurísticas:
      1) dict { <date_string>: <number or dict with 'total'>, ... }
      2) list de dicts com chaves 'date' e 'total'/'seconds'
      3) dict com subchaves que parecem dias e valores numéricos
    """
    # Caso 1: dicionário com chaves-data
    if isinstance(obj, dict):
        # 1a) map direto: { '2025-10-15': 7340, ... }
        date_keys = [k for k in obj.keys() if isinstance(k, str) and norm_date(k)]
        if date_keys:
            for k in date_keys:
                d = norm_date(k)
                v = obj[k]
                if is_number(v):
                    acc.append((d, ".".join(path) or "/", int(v), source_plist))
                elif isinstance(v, dict):
                    # tenta achar 'total', 'duration', 'seconds'
                    for cand in ("total", "duration", "seconds", "value", "usage", "screenTime"):
                        if cand in v and is_number(v[cand]):
                            acc.append((d, ".".join(path+[k,cand]), int(v[cand]), source_plist))
        else:
            # 1b) procurar subestruturas
            for k, v in obj.items():
                walk(v, path+[str(k)], acc, source_plist)

    # Caso 2: lista de dicts com {date:..., total/seconds:...}
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            if isinstance(item, dict):
                # tenta ler 'date'
                dk = None
                for cand in ("date", "day", "Date", "Day"):
                    if cand in item and isinstance(item[cand], str):
                        dk = norm_date(item[cand])
                        if dk: break
                # se não tem campo 'date', mas a lista é de pares, segue o walk normal
                if dk:
                    # tenta achar valor numérico por campos comuns
                    val = None
                    for cand in ("total", "seconds", "duration", "value", "usage", "screenTime"):
                        if cand in item and is_number(item[cand]):
                            val = int(item[cand]); break
                    if val is not None:
                        acc.append((dk, ".".join(path+[f"[{idx}]"]), val, source_plist))
                        continue
                # fallback: seguir descendo
            walk(item, path+[f"[{idx}]"], acc, source_plist)

    # outros tipos: ignora
    else:
        return

def parse_plist(plist_path, source_name):
    with open(plist_path, "rb") as f:
        data = plistlib.load(f)

    # log inicial: imprime chaves top-level (útil para debugging)
    # print(source_name, "top-level:", list(data.keys()))

    acc = []
    walk(data, [], acc, source_name)
    return acc

def main():
    files = []
    for fname in ("DeviceActivity.plist", "ScreenTimeAgent.plist"):
        p = os.path.join(PLIST_DIR, fname)
        if os.path.isfile(p):
            files.append((p, fname.split(".")[0]))
    if not files:
        print("❌ Nenhum plist encontrado em", PLIST_DIR)
        return

    all_rows = []
    for path, src in files:
        rows = parse_plist(path, src)
        all_rows.extend(rows)

    # Deduplicar e agregar por (date, metric_path, source)
    agg = {}
    for d, metric_path, val, src in all_rows:
        if d is None: 
            continue
        key = (d, metric_path, src)
        agg[key] = agg.get(key, 0) + max(0, int(val))

    if not agg:
        print("⚠️ Não encontrei contadores diários nos plists. Pode ser um snapshot só com configurações (sem agregados de uso).")
        print("   Mesmo assim, o arquivo será criado vazio para registro.")
    
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "metric_path", "value_seconds", "source_plist"])
        for (d, metric_path, src), val in sorted(agg.items()):
            w.writerow([d, metric_path, int(val), src])

    # Relato
    print(f"✅ Escrevi: {OUT_CSV}")
    print(f"   Linhas: {len(agg)} (cada uma = um agregado diário encontrado)")
    if not agg:
        print("   Se ficar vazio, próximo passo é tentar KnowledgeC no próximo backup completo;")
        print("   por ora seguimos com labels via State of Mind e/ou Zepp Emotion conforme teu plano.")

if __name__ == "__main__":
    main()
