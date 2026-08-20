import json, re, sys

SRC = "src/notebooks/_nb_src.py"
OUT = "src/notebooks/imputacao_granularidade_comportamental.ipynb"

lines = open(SRC, encoding="utf-8").read().splitlines()

cells = []
cur_type = None
cur_lines = []

def flush():
    global cur_type, cur_lines
    if cur_type is not None:
        src = "\n".join(cur_lines).strip("\n") + "\n"
        cells.append({"cell_type": cur_type,
                      "metadata": {},
                      "source": [l + "\n" for l in src.splitlines()]})
        cur_lines = []

for line in lines:
    if line.startswith("# %%"):
        flush()
        cur_type = "markdown" if "[markdown]" in line else "code"
    else:
        cur_lines.append(line)
flush()

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.14"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

# also dump concatenated code for a compile check
code = "\n\n".join(
    "".join(c["source"]) for c in cells if c["cell_type"] == "code"
)
open("src/notebooks/_nb_code.py", "w", encoding="utf-8").write(code)

print(f"cells={len(cells)} code_cells={sum(1 for c in cells if c['cell_type']=='code')}")
