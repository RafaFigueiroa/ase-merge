"""
Resolve os conflitos do conjunto de teste com a API do Claude e mede a acurácia (exact match),
para comparar com o MergeGen nos mesmos exemplos.
Rode com: python run_claude.py [--run DIR] [--split test|valid] [--limit N]
                               [--model claude-opus-5-5] [--effort medium] [--workers 4]

Lê <DIR>/RAW_DATA/raw_data, <DIR>/RAW_DATA/raw_data_ids.json e <DIR>/all_index (DIR padrão: ".",
ou uma execução arquivada, ex.: runs/toy400-java). Não depende de torch/transformers: basta
o ambiente .venv-claude com requirements-claude.txt (requer Python >= 3.10, por isso não roda
no ambiente ase-merge, de 3.8; veja o README). A chave vem de ANTHROPIC_API_KEY, no ambiente ou num .env na raiz do projeto.

Saídas em <DIR>/OUTPUT/claude/<modelo>/<split>/:
  results.jsonl   uma linha por exemplo (resolução gerada, acerto, tokens); permite retomar
  SUCCEED/, FAIL/ um arquivo por exemplo, no mesmo formato do test_OUTPUT do MergeGen
  summary.json    acurácia, posições acertadas (como RESULTS/test_gen_exactly_match_ids) e tokens

Exact match = resolução e referência iguais após normalizar espaços (' '.join(s.split())),
a mesma normalização do dataset_parallel.py. Diferença: o MergeGen trunca a referência
em 200 tokens; aqui a comparação é com a referência inteira.
"""
import argparse, json, re, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from prepare_raw_data import CONTEXT_LINES

# ANTHROPIC_API_KEY pode vir de um .env na raiz do projeto; variáveis já definidas têm prioridade
load_dotenv(Path(__file__).parent / ".env")

SYSTEM = (
    "You resolve merge conflicts. You receive a three-way conflict (ours, base, theirs) with a few "
    "lines of surrounding context. Produce the code that should replace the conflicting region, "
    "as the developer who merged the branches would have written it."
)

PROMPT = """Language: {lang}

Context before the conflict:
{before}
Conflict:
<<<<<<< ours
{a}||||||| base
{base}=======
{b}>>>>>>> theirs

Context after the conflict:
{after}
Reply with the resolved code for the conflicting region only, inside <resolution></resolution> tags.
Do not include the context lines, conflict markers or markdown fences. If the region should be empty, \
reply with <resolution></resolution>."""

RESOLUTION_RE = re.compile(r"<resolution>(.*?)</resolution>", re.S)


def normalize(s):
    return " ".join(s.split())


def split_context(base, a, b):
    """Separa o contexto (linhas iguais nas três versões, no máximo CONTEXT_LINES de cada lado)
    dos trechos em conflito. Se o arquivo tinha menos linhas de contexto, uma linha do trecho
    igual nas três versões pode ir para o contexto; é raro e não muda a resolução esperada."""
    versions = [v.splitlines(keepends=True) for v in (base, a, b)]
    shortest = min(len(v) for v in versions)
    n_before = 0
    while n_before < min(CONTEXT_LINES, shortest) and versions[0][n_before] == versions[1][n_before] == versions[2][n_before]:
        n_before += 1
    n_after = 0
    while (n_after < min(CONTEXT_LINES, shortest - n_before)
           and versions[0][-1 - n_after] == versions[1][-1 - n_after] == versions[2][-1 - n_after]):
        n_after += 1
    before = "".join(versions[0][:n_before])
    after = "".join(versions[0][len(versions[0]) - n_after:])
    chunks = ["".join(v[n_before:len(v) - n_after]) for v in versions]
    # garante que cada trecho termine em \n, para os marcadores ficarem em linhas próprias
    chunks = [c if not c or c.endswith("\n") else c + "\n" for c in chunks]
    return before, after, chunks


def resolve(client, model, effort, lang, base, a, b):
    before, after, (c_base, c_a, c_b) = split_context(base, a, b)
    prompt = PROMPT.format(lang=lang, before=before or "(none)\n", after=after or "(none)\n",
                           base=c_base, a=c_a, b=c_b)
    response = client.beta.messages.create(
        model=model,
        max_tokens=16000,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
        output_config={"effort": effort},
        # se os classificadores de segurança recusarem (falso positivo), a API refaz o pedido
        # em outro modelo, em vez de devolver a recusa
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    text = "".join(blk.text for blk in response.content if blk.type == "text")
    found = RESOLUTION_RE.findall(text)
    return {
        "output": found[-1].strip("\n") if found else "",
        "parse_error": not found,
        "stop_reason": response.stop_reason,
        "model": response.model,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--run", default=".", help="pasta com RAW_DATA/ e all_index (padrão: projeto atual)")
    p.add_argument("--split", default="test", choices=["test", "valid"])
    p.add_argument("--limit", type=int, help="usa só os N primeiros exemplos do split")
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--workers", type=int, default=4, help="requisições simultâneas")
    args = p.parse_args()

    run = Path(args.run)
    all_base, all_a, all_b, all_res = json.load(open(run / "RAW_DATA/raw_data", encoding="utf-8"))
    ids_path = run / "RAW_DATA/raw_data_ids.json"
    langs = [i[0] for i in json.load(open(ids_path, encoding="utf-8"))] if ids_path.exists() else None
    split_index = json.load(open(run / "all_index"))[args.split]
    if args.limit:
        split_index = split_index[:args.limit]

    out = run / "OUTPUT/claude" / args.model / args.split
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.jsonl"
    done = {}
    if results_path.exists():
        for line in open(results_path, encoding="utf-8"):
            r = json.loads(line)
            done[r["index"]] = r
    todo = [i for i in split_index if i not in done]
    print(f"{args.split}: {len(split_index)} exemplos, {len(done)} já resolvidos, {len(todo)} a resolver "
          f"com {args.model} (effort {args.effort})", flush=True)

    client = anthropic.Anthropic()
    lock = threading.Lock()

    def work(i):
        r = resolve(client, args.model, args.effort, langs[i] if langs else "unknown",
                    all_base[i], all_a[i], all_b[i])
        r["index"] = i
        r["match"] = normalize(r["output"]) == normalize(all_res[i])
        return r

    errors = 0
    with ThreadPoolExecutor(args.workers) as pool, open(results_path, "a", encoding="utf-8") as f:
        futures = {pool.submit(work, i): i for i in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            i = futures[fut]
            try:
                r = fut.result()
            except anthropic.APIError as e:
                # não grava: a próxima execução tenta de novo
                errors += 1
                print(f"  [{n}/{len(todo)}] {i}: erro {type(e).__name__}: {e}", flush=True)
                continue
            with lock:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                done[i] = r
            flag = "ok" if r["match"] else ("recusa" if r["stop_reason"] == "refusal" else "falha")
            print(f"  [{n}/{len(todo)}] {i}: {flag}", flush=True)

    # SUCCEED/FAIL e resumo, na ordem do split (como o test_beam do run_mergegen.py)
    for sub in ("SUCCEED", "FAIL"):
        (out / sub).mkdir(exist_ok=True)
        for old in (out / sub).iterdir():
            old.unlink()
    resolved = [i for i in split_index if i in done]
    for i in resolved:
        r = done[i]
        with open(out / ("SUCCEED" if r["match"] else "FAIL") / str(i), "w", encoding="utf-8") as p_:
            p_.write("ref:\n%s\n" % normalize(all_res[i]))
            p_.write("---------------------------------\n")
            p_.write("output:\n%s\n" % normalize(r["output"]))
            p_.write("---------------------------------\n")
    positions = [k for k, i in enumerate(resolved) if done[i]["match"]]
    summary = {
        "model": args.model,
        "effort": args.effort,
        "split": args.split,
        "num_examples": len(split_index),
        "num_resolved": len(resolved),
        "exactly_match": len(positions) / len(resolved) if resolved else 0.0,
        "exactly_match_positions": positions,
        "refusals": sum(done[i]["stop_reason"] == "refusal" for i in resolved),
        "parse_errors": sum(done[i]["parse_error"] for i in resolved),
        "input_tokens": sum(done[i]["input_tokens"] for i in resolved),
        "output_tokens": sum(done[i]["output_tokens"] for i in resolved),
    }
    json.dump(summary, open(out / "summary.json", "w", encoding="utf-8"), indent=2)
    print(f"exactly match: {summary['exactly_match']:.4f} ({len(positions)}/{len(resolved)})"
          + (f"; {errors} com erro, rode de novo para tentar outra vez" if errors else ""))
    print(f"saídas em {out}")


if __name__ == "__main__":
    main()
