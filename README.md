# GoldiRAGs (Self-Retrieval Augmented Generation)

GoldiRAGs는 모델이 검색한 문서의 관련성과 지원성을 자체적으로 평가하고, 최종 응답 생성에 이 문서들을 사용할지 여부를 결정하는 시스템입니다.

## 개선된 GoldiRAGs 특징

이 프로젝트는 기존 Self-RAG 접근 방식을 다음과 같이 개선했습니다:

1. **다차원 관련성 평가**:
   - 단순 이분법적 평가(관련/비관련) 대신 5가지 평가 기준 도입
   - 주제 관련성, 정보 가치, 사실적 품질, 완전성, 시간적 관련성
   - 가중치 기반 종합 점수 계산

2. **문서 관련성 구간 처리**:
   - 0.66 이상: 관련성 높음 → 즉시 사용
   - 0.33~0.66: 중간 관련성 → 증강 시도
   - 0.33 미만: 관련성 낮음 → 폐기

3. **타겟팅된 질문 재작성 전략**:
   - 문서 약점 분석 기반 후속 질문 생성
   - 관련성 평가 결과를 활용한 고급 질문 생성
   - 여러 문서 통합 고려한 보완적 질문

4. **적응형 증강 종료 조건**:
   - 최대 증강 시도 횟수 제한
   - 최소 점수 향상 임계값 적용
   - 충분한 관련성 달성 시 조기 종료

5. **문서 통합 최적화**:
   - 중복 정보 감지 및 제거
   - 관련성 기반 정보 재정렬
   - 문서 구조화 및 재구성

6. **병렬 처리 및 효율성**:
   - 비동기 평가 및 처리
   - 문서 증강 병렬화
   - 처리 시간 최적화

## 프로젝트 구조

```
goldirags/
├── __init__.py
├── config.py          # 설정 관리
├── main.py            # 메인 실행 파일
├── pipeline.py        # GoldiRAGs 파이프라인
├── evaluator/         # 문서 평가 모듈
│   ├── __init__.py
│   ├── base_evaluator.py
│   ├── relevance_evaluator.py  # ISREL-Token (다차원 평가)
│   ├── support_evaluator.py    # ISSUP-Token
│   └── use_evaluator.py        # ISUSE-Token
├── generator/         # 응답 생성 모듈
│   ├── __init__.py
│   ├── base_generator.py
│   └── llm_generator.py
├── retriever/         # 문서 검색 모듈
│   ├── __init__.py
│   ├── base_retriever.py
│   ├── vector_retriever.py     # Retrieve-Token
│   └── query_rewriter.py       # 질문 재작성 모듈
└── utils/             # 유틸리티 함수
    ├── __init__.py
    ├── common.py
    └── document_processor.py   # 문서 처리 유틸리티
```

## 설치 방법

1. 저장소 클론:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

3. `.env` 파일 생성 및 설정:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## 사용 방법

### 커맨드 라인에서 사용

특정 쿼리에 대해 실행:
```bash
python run_goldirags.py --query "질문을 입력하세요."
```

문서 파일과 함께 실행:
```bash
python run_goldirags.py --files data/sample1.txt data/sample2.txt --query "질문을 입력하세요."
```

대화형 모드로 실행:
```bash
python run_goldirags.py --files data/sample1.txt data/sample2.txt
```

### 코드로 사용

```python
import asyncio
from goldirags.pipeline import GoldiRAGsPipeline
from goldirags.retriever.vector_retriever import VectorRetriever
from goldirags.evaluator.relevance_evaluator import RelevanceEvaluator
from goldirags.evaluator.support_evaluator import SupportEvaluator
from goldirags.evaluator.use_evaluator import UseEvaluator
from goldirags.generator.llm_generator import LLMGenerator

async def main():
    # 파이프라인 컴포넌트 초기화
    retriever = VectorRetriever()
    relevance_evaluator = RelevanceEvaluator()
    support_evaluator = SupportEvaluator()
    use_evaluator = UseEvaluator()
    generator = LLMGenerator()
    
    # 문서 추가
    text_files = ["data/sample1.txt", "data/sample2.txt"]
    retriever = VectorRetriever.from_text_files(text_files)
    
    # 파이프라인 생성
    pipeline = GoldiRAGsPipeline(
        retriever=retriever,
        relevance_evaluator=relevance_evaluator,
        support_evaluator=support_evaluator,
        use_evaluator=use_evaluator,
        generator=generator
    )
    
    # 쿼리 실행
    result = await pipeline.run("질문을 입력하세요.")
    print(result["response"])

if __name__ == "__main__":
    asyncio.run(main())
```

## 평가 결과

다양한 데이터셋에서의 GoldiRAGs 성능 평가 결과입니다:

![평가 결과](images/evaluation_results.png)

*표 1: PopQA, Bio(FactScore), Pub, ARC 데이터셋 성능 평가 결과. 굵은 숫자는 모든 방법 및 LLM 중 최고 성능, 회색 굵은 숫자는 특정 LLM 사용 시 최고 성능을 나타냅니다.*

## 참고 자료

- [Self-RAG 논문](https://arxiv.org/abs/2310.11511)
- [LangChain 문서](https://www.langchain.com/) 