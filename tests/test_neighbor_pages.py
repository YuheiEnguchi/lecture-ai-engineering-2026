"""宿題（実装）: 検索結果に前後ページを付ける処理を、APIを呼ばずに確かめる。"""

import json

import pytest

from app import retrieval


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "lectures.json"
    path.write_text(json.dumps([
        {"id": f"lec{n:02d}-main", "title": f"講義{n}", "content": [
            {"section": "章", "content": [
                {"id": f"lec{n:02d}-main-p{p:03d}", "title": f"ページ{p}", "page": p, "content": f"本文{n}-{p}"}
                for p in range(1, 6)]}]}
        for n in (1, 2)
    ], ensure_ascii=False), encoding="utf-8")
    return path


def hit(page_id, document_id, distance=0.1):
    return {"page_id": page_id, "document_id": document_id, "distance": distance}


def ids(results):
    return [r["page_id"] for r in results]


def test_window_zero_keeps_hits(source):
    hits = [hit("lec01-main-p003", "lec01-main")]
    assert retrieval.add_neighbor_pages(hits, 0, source) == hits


def test_adds_previous_and_next_page(source):
    results = retrieval.add_neighbor_pages([hit("lec01-main-p003", "lec01-main")], 1, source)
    assert ids(results) == ["lec01-main-p002", "lec01-main-p003", "lec01-main-p004"]
    assert results[0]["distance"] is None and results[0]["neighbor_of"] == "lec01-main-p003"
    assert results[0]["content"] == "本文1-2"


def test_does_not_cross_documents(source):
    # lec01 の最終ページの「次」は lec02 の1ページ目なので付けない。
    results = retrieval.add_neighbor_pages([hit("lec01-main-p005", "lec01-main")], 1, source)
    assert ids(results) == ["lec01-main-p004", "lec01-main-p005"]


def test_removes_duplicates(source):
    # p002 と p004 が当たると、p003 は両方の隣だが1回だけ付ける。ヒット同士も重複させない。
    hits = [hit("lec01-main-p002", "lec01-main"), hit("lec01-main-p004", "lec01-main"),
            hit("lec01-main-p003", "lec01-main")]
    results = retrieval.add_neighbor_pages(hits, 1, source)
    assert ids(results) == ["lec01-main-p001", "lec01-main-p002", "lec01-main-p004", "lec01-main-p005",
                            "lec01-main-p003"]
    assert len(ids(results)) == len(set(ids(results)))
    # p003 はヒットなので、distance は検索の値のまま残る。
    assert results[-1]["distance"] == 0.1
