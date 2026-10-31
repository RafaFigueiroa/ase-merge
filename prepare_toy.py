"""
Cria RAW_DATA/raw_data com os N primeiros conflitos de uma linguagem (para testes rápidos).
Rode com: python prepare_toy.py [N] [linguagem]   (padrão: 200 Java; use "all" para N de cada)

Mesmo formato do prepare_raw_data.py: um exemplo por conflito, com contexto.
"""
import sys
from itertools import islice
from prepare_raw_data import conflicts, write_raw_data, parse_langs

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
langs = parse_langs(sys.argv[2] if len(sys.argv) > 2 else "Java")

examples = []
for lang in langs:
    cur = [(lang,) + c for c in islice(conflicts(lang), N)]
    print(f"{lang}: {len(cur)} conflitos")
    examples += cur
write_raw_data(examples)
