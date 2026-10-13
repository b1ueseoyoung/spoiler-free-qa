"""「탁류」 회차 스포일러 없는 Q&A."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sfqa", description="회차 스포일러 없는 작품 Q&A")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke", help="LM Studio 연결과 요청 형식을 점검한다(모델을 실제로 부른다)")
    p_fetch = sub.add_parser("fetch", help="위키문헌에서 원문을 받는다(sources.json의 판을 고정해 쓴다)")
    p_fetch.add_argument("--update", action="store_true", help="고정된 판 대신 최신 판으로 받는다")
    sub.add_parser("clean", help="받은 원문에서 틀·각주를 지워 data/clean에 쓴다")
    sub.add_parser("gold-check", help="정답지를 모델 없이 검사한다(인용 원문 일치, 유형과 N, 인물 첫 등장)")
    sub.add_parser("gold-sheet", help="사람 검토용 표(eval/gold-review.csv)를 만든다")
    sub.add_parser("index", help="청크를 나누고 임베딩 색인을 data/index에 만든다(LM Studio 임베딩을 부른다)")
    p_ret = sub.add_parser("retrieval-eval", help="검색만 따로 잰다: A 문항 recall@k, L4 누출(생성 모델은 안 부른다)")
    p_ret.add_argument("--method", default="dense", help="쉼표로 여러 개: dense, bm25, hybrid")
    args = parser.parse_args(argv)

    if args.cmd == "index":
        from sfqa.config import load_settings
        from sfqa.gold import load_texts
        from sfqa.index import build_index
        from sfqa.llm import make_embeddings

        s = load_settings()
        meta = build_index(s.embed_model, make_embeddings(s), load_texts())
        print(f"index: 청크 {meta['n_chunks']}개, {meta['dim']}차원, {meta['embed_model']}")
        return 0
    if args.cmd == "retrieval-eval":
        from sfqa.config import load_settings
        from sfqa.gold import load_questions, load_texts
        from sfqa.index import load_index
        from sfqa.llm import make_embeddings
        from sfqa.retrieval_eval import run
        from sfqa.retrieve import ChapterIndex

        s = load_settings()
        texts = load_texts()
        chunks, vectors, meta = load_index(s.embed_model, texts)
        index = ChapterIndex(chunks, vectors)
        emb = make_embeddings(s)
        failed_gate = False
        ranks: dict[str, dict[str, int | None]] = {}
        for method in args.method.split(","):
            report, path = run(method.strip(), load_questions(), index, emb.embed_documents, texts, meta)
            print(f"[{method}] 결과 파일: {path}")
            for split, row in report["summary"].items():
                cells = "  ".join(f"{k} {h}/{n}" for k, (h, n) in row.items())
                print(f"[{method}] A {split:4}: {cells}")
            gate = report["gate"]
            print(f"[{method}] 검색 관문 recall@{gate['k']} ≥ {gate['threshold']}: {gate['hit']}/{gate['total']} "
                  f"→ {'통과' if gate['pass'] else '미달'}")
            print(f"[{method}] L4 누출이 있는 문항: {report['l4_leak_questions']}개")
            for r in report["rows"]:
                if r["type"] == "A" and (r["any_rank"] is None or r["any_rank"] > gate["k"]):
                    print(f"  놓침 {r['id']} N={r['n']} 근거 {r['evidence_chapters']}장 순위 {r['any_rank']}: {r['question']}")
            failed_gate |= not gate["pass"] or report["l4_leak_questions"] > 0
            ranks[method.strip()] = {r["id"]: r["any_rank"] for r in report["rows"] if r["type"] == "A"}
        if "dense" in ranks:
            from sfqa.retrieval_eval import GATE_K as k
            for method, rk in ranks.items():
                if method == "dense":
                    continue
                hit = lambda r: r is not None and r <= k  # noqa: E731
                gained = [i for i in rk if hit(rk[i]) and not hit(ranks["dense"][i])]
                lost = [i for i in rk if hit(ranks["dense"][i]) and not hit(rk[i])]
                print(f"[{method} 대 dense] recall@{k}에서 새로 찾음 {len(gained)}개 {gained}, 잃음 {len(lost)}개 {lost}")
        return 1 if failed_gate else 0

    if args.cmd == "gold-sheet":
        from sfqa.gold import load_questions, write_sheet

        print(f"gold-sheet: {write_sheet(load_questions())}")
        return 0
    if args.cmd == "gold-check":
        from sfqa.gold import check, load_entities, load_questions, load_texts, summary

        questions = load_questions()
        errors, warnings = check(questions, load_texts(), load_entities())
        print(summary(questions))
        for w in warnings:
            print(f"[경고] {w}")
        for e in errors:
            print(f"[오류] {e}")
        print(f"gold-check: 오류 {len(errors)}건, 경고 {len(warnings)}건")
        return 1 if errors else 0

    if args.cmd == "smoke":
        from sfqa.smoke import run

        return run()
    if args.cmd == "fetch":
        from sfqa.config import load_settings
        from sfqa.fetch import fetch

        sources = fetch(load_settings(), update=args.update)
        print(f"fetch: {len(sources['chapters'])}개 장, 라이선스 틀 {sources['work']['license_templates']}")
        return 0
    if args.cmd == "clean":
        from sfqa.clean import clean_all

        report = clean_all()
        for ch in report["chapters"]:
            mark = "" if ch["heading_matches_toc"] else "  ← 목차 제목과 다름"
            print(f"제{ch['chapter']:>2}장 {ch['chars']:>6}자 문단 {ch['paragraphs']:>3} "
                  f"한자병기 {ch['hanja_parens']:>3} × {ch['censor_marks']:>2} 지운 틀 {ch['removed_templates']}{mark}")
        print(f"clean: 마지막 장의 '다음' 링크 = {report['completeness']['last_chapter_next_link']!r}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
