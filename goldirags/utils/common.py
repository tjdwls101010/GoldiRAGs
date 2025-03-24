import logging
import tiktoken
from typing import List, Dict, Any, Optional

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("self_rag")

def get_token_count(text: str, model: str = "gpt-3.5-turbo") -> int:
    """입력 텍스트의 토큰 수를 계산합니다."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def truncate_text_tokens(text: str, max_tokens: int, model: str = "gpt-3.5-turbo") -> str:
    """토큰 수에 맞게 텍스트를 잘라냅니다."""
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])

def format_retrieved_documents(docs: List[Dict[str, Any]]) -> str:
    """검색된 문서들을 포맷팅합니다."""
    if not docs:
        return "검색된 문서가 없습니다."
    
    result = []
    for i, doc in enumerate(docs, 1):
        content = doc.get("page_content", "")
        source = doc.get("metadata", {}).get("source", "알 수 없는 출처")
        result.append(f"--- 문서 {i} ---\n출처: {source}\n내용: {content}\n")
    
    return "\n".join(result)

def extract_boolean_from_response(response: str) -> bool:
    """LLM 응답에서 불리언 값을 추출합니다."""
    response = response.lower().strip()
    
    # "true"/"false" 형식 검사
    if response == "true":
        return True
    elif response == "false":
        return False
    
    # "yes"/"no" 형식 검사
    if "yes" in response:
        return True
    if "no" in response:
        return False
    
    # 0~1 사이 값으로 표현된 경우
    try:
        value = float(response)
        return value >= 0.5
    except:
        pass
    
    # 기본적으로 False 반환
    return False 