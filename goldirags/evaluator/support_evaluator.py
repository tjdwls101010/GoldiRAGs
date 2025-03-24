from typing import Dict, Any, Union, List
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from self_rag.evaluator.base_evaluator import BaseEvaluator
from self_rag.config import OPENAI_API_KEY, LLM_MODEL, TEMPERATURE, SUPPORT_THRESHOLD
from self_rag.utils.common import logger, extract_boolean_from_response

class SupportEvaluator(BaseEvaluator):
    """문서의 지원성을 평가하는 ISSUP-Token 구현"""
    
    def __init__(
        self,
        model_name: str = LLM_MODEL,
        temperature: float = TEMPERATURE,
        threshold: float = SUPPORT_THRESHOLD
    ):
        """
        Args:
            model_name: 사용할 LLM 모델 이름
            temperature: 모델 temperature
            threshold: 지원성 판단 임계값
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=OPENAI_API_KEY
        )
        self.threshold = threshold
        
        # 지원성 평가를 위한 프롬프트 템플릿
        self.prompt_template = ChatPromptTemplate.from_template("""
당신은 검색된 문서가 후보 응답을 지원하는지 평가하는 AI 모델입니다.
문서의 내용이 제시된 후보 응답을 사실적으로 지원하는지 평가해야 합니다.

사용자 쿼리: {query}

후보 응답: {candidate_answer}

검색된 문서:
{document}

검색된 문서가 후보 응답의 내용을 사실적으로 지원하는지 평가하세요.
문서가 후보 응답을 지원하면 "True", 그렇지 않으면 "False"로만 응답하세요.
""")
        
        logger.info("SupportEvaluator 초기화 완료")
    
    def evaluate(
        self, 
        query: str, 
        document: Dict[str, Any], 
        candidate_answer: str
    ) -> bool:
        """문서가 후보 응답을 지원하는지 평가합니다.
        
        Args:
            query: 사용자 쿼리
            document: 평가할 문서
            candidate_answer: 평가할 후보 응답
            
        Returns:
            지원성 여부 (True/False)
        """
        # 문서 내용 추출
        document_content = document.get("page_content", "")
        if not document_content:
            logger.warning("문서 내용이 비어 있습니다.")
            return False
        
        if not candidate_answer:
            logger.warning("후보 응답이 비어 있습니다.")
            return False
        
        # 입력 생성
        messages = self.prompt_template.format_messages(
            query=query,
            document=document_content,
            candidate_answer=candidate_answer
        )
        
        try:
            # LLM에 질의
            response = self.llm.invoke(messages)
            result_text = response.content
            
            # 응답에서 불리언 값 추출
            is_supporting = extract_boolean_from_response(result_text)
            
            logger.info(f"후보 응답에 대한 문서 지원성 평가: {is_supporting}")
            return is_supporting
            
        except Exception as e:
            logger.error(f"지원성 평가 중 오류 발생: {str(e)}")
            return False 