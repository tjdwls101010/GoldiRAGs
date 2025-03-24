from typing import Dict, Any, Union, List
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from self_rag.evaluator.base_evaluator import BaseEvaluator
from self_rag.config import OPENAI_API_KEY, LLM_MODEL, TEMPERATURE, USE_THRESHOLD
from self_rag.utils.common import logger, extract_boolean_from_response

class UseEvaluator(BaseEvaluator):
    """문서의 사용 여부를 결정하는 ISUSE-Token 구현"""
    
    def __init__(
        self,
        model_name: str = LLM_MODEL,
        temperature: float = TEMPERATURE,
        threshold: float = USE_THRESHOLD
    ):
        """
        Args:
            model_name: 사용할 LLM 모델 이름
            temperature: 모델 temperature
            threshold: 사용 결정 임계값
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=OPENAI_API_KEY
        )
        self.threshold = threshold
        
        # 사용 결정을 위한 프롬프트 템플릿
        self.prompt_template = ChatPromptTemplate.from_template("""
당신은 검색된 문서를 최종 응답 생성에 사용할지 결정하는 AI 모델입니다.
문서의 관련성과 지원성을 고려하여 최종 응답 생성에 이 문서를 사용할지 여부를 결정해야 합니다.

사용자 쿼리: {query}

검색된 문서:
{document}

문서 관련성 평가: {is_relevant}
문서 지원성 평가: {is_supporting}

위 정보를 바탕으로 이 문서를 최종 응답 생성에 사용해야 하는지 결정하세요.
사용해야 한다면 "True", 사용하지 말아야 한다면 "False"로만 응답하세요.
""")
        
        logger.info("UseEvaluator 초기화 완료")
    
    def evaluate(
        self, 
        query: str, 
        document: Dict[str, Any], 
        is_relevant: bool = False,
        is_supporting: bool = False
    ) -> bool:
        """문서를 최종 응답 생성에 사용할지 결정합니다.
        
        Args:
            query: 사용자 쿼리
            document: 평가할 문서
            is_relevant: 문서 관련성 평가 결과
            is_supporting: 문서 지원성 평가 결과
            
        Returns:
            사용 결정 (True/False)
        """
        # 문서 내용 추출
        document_content = document.get("page_content", "")
        if not document_content:
            logger.warning("문서 내용이 비어 있습니다.")
            return False
        
        # 입력 생성
        messages = self.prompt_template.format_messages(
            query=query,
            document=document_content,
            is_relevant=str(is_relevant),
            is_supporting=str(is_supporting)
        )
        
        try:
            # LLM에 질의
            response = self.llm.invoke(messages)
            result_text = response.content
            
            # 응답에서 불리언 값 추출
            should_use = extract_boolean_from_response(result_text)
            
            logger.info(f"문서 사용 결정: {should_use}")
            return should_use
            
        except Exception as e:
            logger.error(f"사용 결정 평가 중 오류 발생: {str(e)}")
            return False 