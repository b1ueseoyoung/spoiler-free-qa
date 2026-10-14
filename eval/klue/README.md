# KLUE-MRC 평가 표본

`klue-mrc-val-240.jsonl`은 KLUE-MRC validation에서 유형별로 80문항씩 뽑은 240문항이다(`scripts/export_klue.py`, seed 20261014).

- 출처: KLUE: Korean Language Understanding Evaluation (Park et al., 2021). https://klue-benchmark.com , Hugging Face `klue/klue`의 `mrc` 설정
- 라이선스: CC BY-SA 4.0. 이 표본도 같은 라이선스(CC BY-SA 4.0)를 따른다.
- 원본에서 바꾼 것: 문항을 골라냈고, 제목과 정답 표기의 HTML 엔티티(`&amp;` 등)를 문자로 풀었다. `answer_start`, `source`, `news_category` 등 일부 열은 뺐다.
- 유형 3(`is_impossible: true`)의 `answers`는 정답이 아니라 지문 속 가짜 답(함정)이다.
