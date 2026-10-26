"""Day1 発展課題: 応答時間とトークン数をモデル・質問ごとに計測する。

Codespaces で `uv run python app/bench.py` として実行する。
API キーは環境変数（Codespaces の Secret）から読み、表示・保存しない。
"""

import csv
import os
import statistics
import time

from google import genai

MODELS = {
    # モデル名: 呼び出し間隔（秒）＝無料枠の毎分上限を超えないように空ける
    "gemini-3.5-flash-lite": 5,  # 毎分15
    "gemini-3.8-flash": 13,  # 毎分5
}
QUESTIONS = {
    "short": "Pythonのリストとタプルの違いを3行で説明してください。",
    "long": "RAG（検索拡張生成）の仕組みを、初心者向けに400字程度で説明してください。",
    "course": "この講座（AIエンジニアリング実践講座2026）のDay1の宿題は何ですか。",
}
REPEATS = 3
OUT = "lectures/01/bench_results.csv"


def main() -> None:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    client = genai.Client(api_key=key)
    rows = []
    for model, wait in MODELS.items():
        for qid, q in QUESTIONS.items():
            for i in range(1, REPEATS + 1):
                t0 = time.perf_counter()
                row = {"model": model, "question": qid, "run": i}
                try:
                    res = client.models.generate_content(model=model, contents=q)
                    u = res.usage_metadata
                    row.update(
                        status="ok",
                        seconds=round(time.perf_counter() - t0, 2),
                        input_tokens=u.prompt_token_count,
                        output_tokens=u.candidates_token_count,
                        thinking_tokens=u.thoughts_token_count or 0,
                        total_tokens=u.total_token_count,
                        answer_head=(res.text or "")[:60].replace("\n", " "),
                    )
                except Exception as e:  # 429・503 も結果として残す
                    row.update(status=type(e).__name__, seconds=round(time.perf_counter() - t0, 2), answer_head=str(e)[:80])
                rows.append(row)
                print(row)
                time.sleep(wait)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fields = ["model", "question", "run", "status", "seconds", "input_tokens", "output_tokens", "thinking_tokens", "total_tokens", "answer_head"]
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print("\n| モデル | 質問 | 成功/試行 | 平均秒 | 平均出力tok | 平均思考tok |")
    print("|---|---|---|---|---|---|")
    for model in MODELS:
        for qid in QUESTIONS:
            rs = [r for r in rows if r["model"] == model and r["question"] == qid]
            ok = [r for r in rs if r["status"] == "ok"]
            avg = lambda k: round(statistics.mean(r[k] for r in ok), 1) if ok else "-"  # noqa: E731
            print(f"| {model} | {qid} | {len(ok)}/{len(rs)} | {avg('seconds')} | {avg('output_tokens')} | {avg('thinking_tokens')} |")


if __name__ == "__main__":
    main()
