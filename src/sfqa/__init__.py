"""「탁류」 회차 스포일러 없는 Q&A."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


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
    p_run = sub.add_parser("eval-run", help="정답지 문항을 조건별로 돌린다(카세트 기록, 호출 상한)")
    p_run.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    p_run.add_argument("--conditions", default="C0,C1,C2,C3,C4")
    p_run.add_argument("--max-calls", type=int, required=True, help="실제 모델 호출 상한(카세트에 있으면 세지 않는다)")
    p_run.add_argument("--limit", type=int, help="앞에서부터 이 개수만")
    p_run.add_argument("--k", type=int, default=18)
    p_run.add_argument("--replay", action="store_true", help="카세트만 쓴다(네트워크 없음)")
    p_run.add_argument("--cassette", help="카세트 이름(기본: split 이름). eval/cassettes/<이름>.jsonl")
    p_score = sub.add_parser("eval-score", help="실행 기록을 모델 없이 채점한다")
    p_score.add_argument("run_dir")
    p_score.add_argument("--with-review", action="store_true", help="review.csv의 사람 판정을 따로 반영해 센다")
    p_web = sub.add_parser("web", help="로컬 웹 데모를 연다(127.0.0.1만)")
    p_web.add_argument("--port", type=int, default=8765)
    p_ask = sub.add_parser("ask", help="질문 하나에 답한다(LM Studio를 부른다)")
    p_ask.add_argument("--upto", type=int, required=True, help="독자가 읽은 마지막 장(1~19)")
    p_ask.add_argument("--condition", default="C4", help="C0~C4. 기본 C4(최종 시스템)")
    p_ask.add_argument("--k", type=int, default=18, help="가져올 청크 수")
    p_ask.add_argument("question")
    p_cmp = sub.add_parser("compare-run", help="모델 비교: 모델을 하나씩 올려 KLUE 240 + test 44를 돌린다(docs/model-eval-spec.md)")
    p_cmp.add_argument("--models", default="all", help="쉼표로 여러 개, 또는 all / local")
    p_cmp.add_argument("--max-calls", type=int, required=True, help="모델마다 실제 호출 상한(카세트에 있으면 세지 않는다)")
    p_cmp.add_argument("--check-only", action="store_true", help="올리기·점검·dev 12문항만 한다")
    p_cmp.add_argument("--out", help="이어서 돌릴 runs/compare-* 폴더")
    p_rep = sub.add_parser("compare-report", help="비교 결과 폴더에서 compare.md를 다시 만든다(모델 없음)")
    p_rep.add_argument("out")
    args = parser.parse_args(argv)

    if args.cmd == "compare-run":
        from sfqa.compare import PROFILES, run_compare
        from sfqa.config import load_settings

        if args.models == "all":
            slugs = list(PROFILES)
        elif args.models == "local":
            slugs = [k for k, p in PROFILES.items() if not p.baseline]
        else:
            slugs = [m.strip() for m in args.models.split(",")]
        out = run_compare(load_settings(), slugs, args.max_calls, Path(args.out) if args.out else None, args.check_only)
        print((out / "compare.md").read_text(encoding="utf-8"))
        print(f"compare-run: {out}")
        return 0
    if args.cmd == "compare-report":
        from sfqa.compare import write_report

        print(write_report(Path(args.out)).read_text(encoding="utf-8"))
        return 0

    if args.cmd == "eval-run":
        from sfqa.config import load_settings
        from sfqa.evaluate import run_eval

        out, info = run_eval(load_settings(), args.split, [c.strip() for c in args.conditions.split(",")],
                             args.max_calls, k=args.k, limit=args.limit, replay=args.replay, cassette_name=args.cassette)
        print(f"eval-run: {out}")
        print(f"  기록 {info['records']}건, 모델 호출 {info['model_calls']}회, "
              + ("완료" if info["complete"] else f"중간에 멈춤({info['stopped']})"))
        print(f"  채점: uv run sfqa eval-score {out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out}")
        return 0 if info["complete"] else 3
    if args.cmd == "eval-score":
        from sfqa.score import score_run

        out = score_run(Path(args.run_dir), with_review=args.with_review)
        print((Path(args.run_dir) / "summary.md").read_text(encoding="utf-8"))
        print(f"eval-score: 사람 검토표 {out['review_rows']}행 → {Path(args.run_dir) / 'review.csv'}")
        return 0
    if args.cmd == "web":
        from sfqa.web import serve

        serve(port=args.port)
        return 0
    if args.cmd == "ask":
        from sfqa.answer import build_chain, make_env
        from sfqa.config import load_settings

        if not 1 <= args.upto <= 19:
            parser.error("--upto는 1~19다")
        r = build_chain(make_env(load_settings(), k=args.k), args.condition).invoke(
            {"question": args.question, "n": args.upto})
        resp = r.response
        print(f"[{r.condition}, {r.n}장까지] {r.status}: {r.answer}")
        for q in r.quotes:
            mark = {"verified": "", "chapter_fixed": f" (장 번호 고침 → {q['found_chapter']}장)"}.get(q["status"], f" [버림: {q['status']}]")
            ctx = "" if q["in_context"] or q["status"] not in ("verified", "chapter_fixed") else " [발췌 밖, 기억에서 옮김]"
            print(f"  - 제{q['chapter']}장: {q['text']}{mark}{ctx}")
        if r.reasons:
            print(f"  검증 메모: {'; '.join(r.reasons)}")
        if r.status != r.model_status:
            print(f"  모델 원래 답({r.model_status}): {r.model_answer}")
        chapters = sorted({d['chapter'] for d in r.retrieved})
        print(f"  검색: 청크 {len(r.retrieved)}개, 장 {chapters}")
        print(f"  모델: 입력 {resp['input_tokens']} 출력 {resp['output_tokens']} 토큰, {resp['latency_s']}초"
              + (", 잘림" if r.truncated else "") + (", 형식 실패" if r.invalid else ""))
        return 0

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
