import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# OpenAI API 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-ada-002"
LLM_MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0.0

# 검색기 설정
TOP_K = 3
RETRIEVER_THRESHOLD = 0.7  # 벡터 유사도 임계값

# 평가 프롬프트 설정
RELEVANCE_THRESHOLD = 0.5  # ISREL 결정 임계값
SUPPORT_THRESHOLD = 0.5    # ISSUP 결정 임계값
USE_THRESHOLD = 0.5        # ISUSE 결정 임계값

# 관련성 점수 구간 설정
DISCARD_THRESHOLD = 0.33   # 이 점수 미만은 폐기
KEEP_THRESHOLD = 0.66      # 이 점수 이상은 유지

# 증강 설정
MAX_ENHANCEMENT_ATTEMPTS = 3  # 최대 증강 시도 횟수
MIN_SCORE_IMPROVEMENT = 0.05  # 최소 점수 향상 임계값
FOLLOW_UP_QUESTIONS_PER_DOC = 3  # 문서당 생성할 후속 질문 수

# Tokenization 관련 설정
MAX_NEW_TOKENS = 512
MAX_INPUT_TOKENS = 3000 