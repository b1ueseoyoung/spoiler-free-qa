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
    args = parser.parse_args(argv)

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
