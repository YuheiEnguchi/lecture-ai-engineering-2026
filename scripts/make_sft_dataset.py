"""第2回 宿題（発展）: 用語集から SFT 用の Q&A データ（JSONL）を作る。

使い方:
    uv run python scripts/make_sft_dataset.py > glossary_sft.jsonl          # 学習データ
    uv run python scripts/make_sft_dataset.py --eval > glossary_eval.json   # 評価用の質問と正解

できたファイルを Colab にアップロードし、LoRA 学習のデータとして使う。
質問の言い回しを増やす、別名でも質問を作る、検証用に一部を分ける、などを工夫してみよう。

演習（FAQ）と同じく、望ましい出力は {"answer": ..., "needs_review": ...} の JSON 1 行にする。
- 用語集にある語: answer に定義を書き、needs_review は false
- 用語集に無い語: answer に UNKNOWN_ANSWER を書き、needs_review は true
FAQ と違って用語集は入力に渡さない（定義をモデルに覚えさせる）。
"""

import json
import sys
from pathlib import Path

GLOSSARY = Path(__file__).resolve().parent.parent / "data" / "glossary.json"
TEMPLATES = ["「{term}」とは何ですか？", "{term}について簡単に説明してください。"]

SYSTEM = (
    "あなたは AIE2026 講座の用語集の回答係です。必ず JSON 1 行で答えてください。"
    "形式: {\"answer\": 文字列, \"needs_review\": 真偽値}。"
    "用語集にある用語は answer に定義を書き、needs_review を false にします。"
    "用語集に無い用語は answer に『用語集に記載がありません。確認が必要です。』と書き、needs_review を true にします。"
)
UNKNOWN_ANSWER = "用語集に記載がありません。確認が必要です。"

# 学習用の「用語集に無い語」。評価用（EVAL_UNGROUNDED）とは重ねない
TRAIN_UNGROUNDED = [
    "Transformer", "埋め込み（Embedding）", "RLHF", "知識蒸留",
    "Docker", "ベクトルデータベース", "プロンプトインジェクション", "強化学習",
]

# 評価用の質問。(用語集の id, 質問, 検索で使う語, answer に含まれるべき語)
# same: 学習と同じ言い回し（覚えたかの確認）
EVAL_SAME = [
    ("token", "「トークン」とは何ですか？", "トークン", ["最小単位"]),
    ("lora", "「LoRA」とは何ですか？", "LoRA", ["低ランク"]),
    ("epoch", "「エポック」とは何ですか？", "エポック", ["1周"]),
    ("homework-levels", "「宿題の3レベル」とは何ですか？", "宿題の3レベル", ["基礎", "発展"]),
    ("lec-start-tag", "「lecNN-start タグ」とは何ですか？", "lecNN-start タグ", ["タグ", "ブランチ"]),
    ("pi", "「Pi」とは何ですか？", "Pi", ["コーディングエージェント"]),
    ("default-model", "「既定モデル」とは何ですか？", "既定モデル", ["gemini-3.5-flash-lite"]),
    ("harness", "「ハーネス」とは何ですか？", "ハーネス", ["ループ"]),
]
# reworded: 学習に無い言い回し・別名で聞く（言い換えへの汎化の確認）
EVAL_REWORDED = [
    ("pull-request", "PR ってどういう意味ですか？", "PR", ["ブランチ"]),
    ("qlora", "QLoRA は普通の LoRA と何が違いますか？", "QLoRA", ["量子化"]),
    ("human-in-the-loop", "HITL とはどういう考え方ですか？", "HITL", ["承認"]),
    ("context-window", "コンテキスト長とは何を指しますか？", "コンテキスト長", ["上限"]),
    ("codespaces-secret", "Codespacesのシークレットは何に使いますか？", "Codespacesのシークレット", ["環境変数"]),
    ("answer-md", "解答ファイルには何を書けばいいですか？", "解答ファイル", ["レベル"]),
    ("test-model", "TestModel を使うと何ができますか？", "TestModel", ["テスト"]),
    ("mcp-transport", "MCP の stdio と Streamable HTTP はどう違いますか？", "stdio", ["標準入出力"]),
    ("overfitting", "オーバーフィッティングが起きるとどうなりますか？", "オーバーフィッティング", ["未知"]),
    ("rate-limit", "レート制限に引っかかると何が起きますか？", "レート制限", ["429"]),
]
# ungrounded: 用語集に無い語（学習用の TRAIN_UNGROUNDED とも重ねない）。近い語が用語集にあるものを混ぜる
EVAL_UNGROUNDED = [
    ("「RAG（検索拡張生成）」とは何ですか？", "RAG"),
    ("「トークナイザー」とは何ですか？", "トークナイザー"),          # 用語集の「トークン」に近い
    ("「DPO」とは何ですか？", "DPO"),                                # 用語集の「SFT」に近い
    ("「lec02-end タグ」とは何ですか？", "lec02-end タグ"),          # 用語集の「lecNN-start タグ」に近い（架空）
    ("Kubernetes について簡単に説明してください。", "Kubernetes"),
    ("Mixture of Experts ってどういう仕組みですか？", "Mixture of Experts"),
    ("ビームサーチとは何ですか？", "ビームサーチ"),
    ("CI/CD は何のために使いますか？", "CI/CD"),
]


def record(question: str, answer: str, needs_review: bool) -> dict:
    target = json.dumps({"answer": answer, "needs_review": needs_review}, ensure_ascii=False)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
            {"role": "assistant", "content": target},
        ]
    }


def train_records(terms: list[dict]) -> list[dict]:
    rows = [record(tp.format(term=t["term"]), t["definition"], False) for t in terms for tp in TEMPLATES]
    rows += [record(tp.format(term=w), UNKNOWN_ANSWER, True) for w in TRAIN_UNGROUNDED for tp in TEMPLATES]
    return rows


def eval_set(terms: list[dict]) -> dict:
    by_id = {t["id"]: t for t in terms}
    test, gold = [], {}
    for category, items in (("same", EVAL_SAME), ("reworded", EVAL_REWORDED)):
        for term_id, q, query, must in items:
            assert all(w in by_id[term_id]["definition"] for w in must), f"{term_id}: 正解の語が定義に無い"
            test.append({"question": q, "category": category, "query": query, "term_id": term_id,
                         "definition": by_id[term_id]["definition"]})
            gold[q] = {"needs_review": False, "must_include": must}
    for q, query in EVAL_UNGROUNDED:
        test.append({"question": q, "category": "ungrounded", "query": query, "term_id": None, "definition": None})
        gold[q] = {"needs_review": True, "must_include": ["確認"]}
    return {"system": SYSTEM, "test": test, "gold": gold}


def main() -> None:
    terms = json.loads(GLOSSARY.read_text(encoding="utf-8"))["terms"]
    train = train_records(terms)
    evaluation = eval_set(terms)
    train_questions = {r["messages"][1]["content"] for r in train}
    unseen = [r["question"] for r in evaluation["test"] if r["category"] != "same"]
    assert not train_questions & set(unseen), "学習に無いはずの評価用の質問が学習データにある"
    assert all(r["question"] in train_questions for r in evaluation["test"] if r["category"] == "same")

    if "--eval" in sys.argv[1:]:
        sys.stdout.write(json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n")
        return
    for r in train:
        sys.stdout.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
