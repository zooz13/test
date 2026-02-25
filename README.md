# Multi-LLM Perspective Comparator (Prototype)

다중 LLM 응답을 claim 단위로 비교해 `Topic × Stance` 구조로 `Common Positive / Common Negative / Common Conditional / Conflicts / Unique`를 보여주는 프로토타입입니다.

## 현재 프로토타입 범위
- Claim 분해: 규칙 기반 (영어/한국어 커넥터 일부 지원)
- Topic 분류: 규칙 기반 멀티 라벨(`비용/채용/문화/속도/생산성/전략/META`, claim당 최대 2개)
- Stance 분류: `POSITIVE/NEGATIVE/CONDITIONAL/META`
- 임베딩: 경량 TF-IDF
- 클러스터링: Topic 내부에서 코사인 유사도 threshold 기반
- 상충 탐지: 휴리스틱(부정 cue + 반대 단어 + 유사도)
  - 기본: NLI 모델 기반 contradiction score 결합(로드 실패 시 휴리스틱 폴백)
- 결과 출력:
  - `Topic별 Common Positive`
  - `Topic별 Common Negative`
  - `Topic별 Common Conditional`
  - `Conflicts` 상충 claim 쌍
  - `Weak Conflicts` (경계값 구간)
  - `Unique` 모델별 고유 주장
  - `Perspective Map` 시각화:
    - Topic × Stance 매트릭스(클릭 상세)
    - 2D 임베딩 스캐터(토픽 색/스탠스 모양/충돌선)
- 웹 UI:
  - 채팅형 질문 입력
  - `GPT 3모델 자동 수집` 또는 `수동 3모델 입력`
  - `수동 배치 입력`: 질문 여러 개 + 각 질문당 3모델 답변 직접 입력
  - claim 벡터 2D 산점도 + claim별 top terms

## 모델 정책(요청 반영)
- 기본 3모델: `gpt-4o`, `gpt-4.1`, `gpt-4.1-mini`
- 자동 fallback: 개별 모델 호출 실패 시 `gpt-4o-mini`로 재시도
- 변경 가능: `GPT_MODELS` 환경변수 (콤마 구분, 앞 3개 사용)

## CLI 실행
```bash
python3 main.py --input data/sample_input.json
python3 main.py --input data/sample_ko_input.json
python3 main.py --input data/sample_conflict_input.json
```

## 웹 실행
```bash
python3 webapp.py
# then open http://127.0.0.1:8787
```

### GPT 자동 수집 모드 사용 시
```bash
export OPENAI_API_KEY="<your_key>"
# optional
export GPT_MODELS="gpt-4o,gpt-4.1,gpt-4.1-mini"
python3 webapp.py
```

### NLI 상충 탐지(기본 ON)
`transformers`가 설치되어 있으면 NLI를 사용할 수 있습니다.

CLI:
```bash
python3 main.py --input data/sample_ko_input.json --use-nli
```

웹:
```bash
export ENABLE_NLI=1
python3 webapp.py
```

선택 모델 지정:
```bash
export NLI_MODEL="joeddav/xlm-roberta-large-xnli"
```

미설치/로딩 실패 시 자동으로 기존 휴리스틱으로 폴백합니다.

API key가 없으면 웹에서 `수동 입력(3모델)` 또는 `수동 입력(여러 질문 배치)` 모드로 데모할 수 있습니다.
배치 입력 템플릿: `data/manual_batch_template_10.json`

## 입력 형식 (CLI)
```json
{
  "question": "...",
  "responses": [
    {"model": "gpt-4o", "text": "..."},
    {"model": "gpt-4.1", "text": "..."},
    {"model": "gpt-4.1-mini", "text": "..."}
  ]
}
```

## 구조
- `main.py`: CLI 진입점
- `webapp.py`: 웹 서버 + `/api/analyze`
- `web/index.html`: 채팅형 UI + 시각화
- `comparator/segmenter.py`: claim 분해
- `comparator/embedder.py`: 토큰화/정규화/TF-IDF
- `comparator/polarity.py`: claim polarity(PRO/CON/NEUTRAL) 분류
- `comparator/clusterer.py`: 클러스터링
- `comparator/conflict.py`: 상충 점수
- `comparator/projection.py`: 벡터 2D 투영
- `comparator/pipeline.py`: 전체 비교 파이프라인
- `comparator/collector.py`: GPT 3모델 수집

## 한계 (프로토타입)
- 한국어 형태소 수준 분석이 아님 (규칙 기반 토큰화)
- 유사어 처리 범위가 제한적
- 상충 탐지는 NLI 미사용 휴리스틱이라 정밀도 제한
- 2D 시각화는 해시 기반 투영으로 해석용 참고 수준

## 다음 단계(짧게)
1. 임베딩을 sentence-transformers/OpenAI embedding으로 교체
2. 상충 탐지를 NLI 모델로 교체
3. 클러스터 라벨링(선택적 LLM 포장) 추가
## 시각화 JSON 키
- `topics_axis`: 매트릭스 row 토픽 목록
- `stances`: 매트릭스 column 스탠스 목록
- `buckets`: `(topic, stance)` 버킷 메타(모델 다양성/대표 주장/conflict_count)
- `claims`: claim 레벨 시각화 레코드(`topic_labels`, `stance`, `emb2d`)
- `visual_conflicts`: 스캐터 edge 렌더링용 충돌 쌍
