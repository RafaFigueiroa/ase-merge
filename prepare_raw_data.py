"""
Cria RAW_DATA/raw_data com UM EXEMPLO POR CONFLITO (não por arquivo), como no paper.
Rode com: python prepare_raw_data.py [linguagem]   (padrão: Java; use "all" para todas)

Os conflitos vêm do campo conflicting_chunks dos <id>_metadata.json do dataset do MergeBERT.
Para cada conflito:
  base/a/b = contexto antes + trecho da versão + contexto depois
  res      = res_region (a resolução do desenvolvedor para aquele trecho)
O paper não diz quanto contexto usa; aqui são CONTEXT_LINES linhas de cada lado,
tiradas de lookback/lookahead.

Descarta conflitos sem resolução encontrada (found_res_region), cujo arquivo resolvido
ficou vazio (RES_FILE_EMPTY), com metadados corrompidos ou de arquivos minificados.

Também grava RAW_DATA/raw_data_ids.json com a origem de cada exemplo
([linguagem, id do arquivo, índice do conflito]), na mesma ordem do raw_data.
"""
import sys, json
from pathlib import Path

ROOT = Path("RAW_DATA/fse2022/automated-analysis-data")
LANGS = {"Java": ".java", "CSharp": ".cs", "JavaScript": ".js", "TypeScript": ".ts"}
# Medido em 400 conflitos de Java: com 2 linhas, 94,8% das entradas cabem nos 500 tokens
# (o paper diz 95%). Com 0 linhas cai para 58%: sem contexto, o merge por token não tem
# onde ancorar e junta as três versões inteiras num único conflito.
CONTEXT_LINES = 2


def conflicts(lang):
    """Gera (id, índice, base, a, b, res) para cada conflito válido da linguagem."""
    d = ROOT / lang
    ids = sorted(f.name[:-len("_metadata.json")] for f in d.glob("*_metadata.json"))
    for cid in ids:
        m = json.loads((d / f"{cid}_metadata.json").read_text(encoding="utf-8", errors="replace"))
        if m.get("metadata_corrupted") or m.get("minified"):
            continue
        for k, c in enumerate(m.get("conflicting_chunks", [])):
            if not c.get("found_res_region") or c.get("label") == "RES_FILE_EMPTY":
                continue
            before = c["lookback"].replace("<START\\>", "").splitlines(keepends=True)[-CONTEXT_LINES:]
            after = c["lookahead"].replace("<END\\>", "").splitlines(keepends=True)[:CONTEXT_LINES]
            before, after = "".join(before), "".join(after)
            yield (cid, k,
                   before + c["base_contents"] + after,
                   before + c["a_contents"] + after,
                   before + c["b_contents"] + after,
                   c["res_region"])


def write_raw_data(examples):
    """examples: lista de (linguagem, id, índice, base, a, b, res)."""
    base_list, a_list, b_list, res_list = ([e[i] for e in examples] for i in (3, 4, 5, 6))
    with open("RAW_DATA/raw_data", "w", encoding="utf-8") as f:
        json.dump([base_list, a_list, b_list, res_list], f)
    with open("RAW_DATA/raw_data_ids.json", "w", encoding="utf-8") as f:
        json.dump([list(e[:3]) for e in examples], f)
    print(f"Total: {len(examples)} conflitos -> RAW_DATA/raw_data", flush=True)


def parse_langs(arg):
    if arg == "all":
        return list(LANGS)
    if arg not in LANGS:
        sys.exit(f"linguagem desconhecida: {arg} (use {', '.join(LANGS)} ou all)")
    return [arg]


if __name__ == "__main__":
    langs = parse_langs(sys.argv[1] if len(sys.argv) > 1 else "Java")
    examples = []
    for lang in langs:
        cur = [(lang,) + c for c in conflicts(lang)]
        print(f"{lang}: {len(cur)} conflitos", flush=True)
        examples += cur
    write_raw_data(examples)
