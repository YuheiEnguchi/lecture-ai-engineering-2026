"""第2回 宿題（発展）: 同じ評価セットで「用語集を検索する仕組み」を採点し、LoRA 学習と比べる。

使い方:
    uv run python scripts/eval_glossary_lookup.py

search_glossary（第3回のツール）の先頭の結果を答えとみなし、ノートブックと同じ基準で採点する。
- 見つかった: answer = 先頭の定義、needs_review = false
- 見つからない: answer = 「確認が必要」、needs_review = true
検索語は 2 通り試す。query = 質問から用語だけを取り出せた場合（エージェントの LLM が取り出す想定）、
question = 質問文をそのまま検索語にした場合。
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_sft_dataset import GLOSSARY, eval_set  # noqa: E402

from app.tools import search_glossary  # noqa: E402


def lookup(query: str) -> dict:
    raw = search_glossary(query)
    if raw.startswith("["):
        top = json.loads(raw)[0]
        return {"answer": top["definition"], "needs_review": False, "hit": top["term"]}
    return {"answer": "用語集に記載がありません。確認が必要です。", "needs_review": True, "hit": None}


def main() -> None:
    terms = json.loads(GLOSSARY.read_text(encoding="utf-8"))["terms"]
    evaluation = eval_set(terms)
    for mode in ("query", "question"):
        rates: dict[str, list[bool]] = defaultdict(list)
        print(f"## 検索語 = {mode}")
        for row in evaluation["test"]:
            g = evaluation["gold"][row["question"]]
            out = lookup(row[mode])
            ok = out["needs_review"] == g["needs_review"] and all(w in out["answer"] for w in g["must_include"])
            rates[row["category"]].append(ok)
            rates["all"].append(ok)
            print(f"{'○' if ok else '×'} [{row['category']}] {row[mode]} -> {out['hit']}")
        print({k: f"{sum(v)}/{len(v)}" for k, v in rates.items()})
        print()


if __name__ == "__main__":
    main()
