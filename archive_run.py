"""
Arquiva os artefatos de uma execução (pré-processamento, treino e teste) em runs/<nome>/
e deixa o projeto limpo para a próxima execução.
Rode com: python archive_run.py <nome> [--dry-run]

- move:   divisão (all_index), modelo, estado do treino, dados processados e saídas (inclusive do run_claude.py)
- copia:  RAW_DATA/raw_data e raw_data_ids.json (entrada da execução; os índices do all_index apontam para eles)
- apaga:  PROCESSED/ e GIT_MERGE_FILES/ (intermediários, regenerados pelo pré-processamento)
A saída original dos autores (OUTPUT/test_output) não é tocada.
"""
import sys, json, shutil
from datetime import datetime
from pathlib import Path

MOVE = [
    "all_index", "best_acc", "train_state", "best_model.pt",
    "processed_train.pkl", "processed_valid.pkl", "processed_test.pkl",
    "OUTPUT/DEV_OUTPUT", "OUTPUT/test_OUTPUT",
    "OUTPUT/train_process", "OUTPUT/dev_process", "OUTPUT/test_process",
    "OUTPUT/total_gen_output_dev.json", "OUTPUT/total_gen_output_test_beam.json",
    "OUTPUT/claude", "RESULTS",
]
COPY = ["RAW_DATA/raw_data", "RAW_DATA/raw_data_ids.json"]
DELETE = ["PROCESSED", "GIT_MERGE_FILES"]

args = [a for a in sys.argv[1:] if not a.startswith("--")]
dry = "--dry-run" in sys.argv
if len(args) != 1:
    sys.exit("uso: python archive_run.py <nome> [--dry-run]")

dest = Path("runs") / args[0]
if dest.exists():
    sys.exit(f"{dest} já existe; escolha outro nome.")

# resumo da execução, lido antes de mover qualquer coisa
info = {"name": args[0], "archived_at": datetime.now().isoformat(timespec="seconds")}
if Path("RAW_DATA/raw_data").exists():
    info["num_examples"] = len(json.load(open("RAW_DATA/raw_data", encoding="utf-8"))[0])
if Path("all_index").exists():
    info["split"] = {k: len(v) for k, v in json.load(open("all_index")).items()}
for line in Path("OUTPUT/test_process").read_text().splitlines() if Path("OUTPUT/test_process").exists() else []:
    k, _, v = line.partition(":")
    info.setdefault("test_process", {})[k.strip()] = float(v)

print(f"{'[dry-run] ' if dry else ''}arquivando em {dest}")
if not dry:
    dest.mkdir(parents=True)

plan = [("move", p) for p in MOVE] + [("copia", p) for p in COPY] + [("apaga", p) for p in DELETE]
for action, p in plan:
    p = Path(p)
    if not p.exists():
        continue
    print(f"  {action:5} {p}")
    if dry:
        continue
    if action == "apaga":
        shutil.rmtree(p) if p.is_dir() else p.unlink()
        continue
    (dest / p).parent.mkdir(parents=True, exist_ok=True)
    if action == "copia":
        shutil.copy2(p, dest / p)
    else:
        shutil.move(str(p), str(dest / p))

if not dry:
    # run_mergegen.py grava em RESULTS/ sem criar a pasta
    Path("RESULTS").mkdir(exist_ok=True)
    json.dump(info, open(dest / "run_info.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(json.dumps(info, indent=2, ensure_ascii=False))
