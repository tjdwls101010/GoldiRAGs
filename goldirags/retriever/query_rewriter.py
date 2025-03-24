from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from self_rag.config import OPENAI_API_KEY, LLM_MODEL, TEMPERATURE, FOLLOW_UP_QUESTIONS_PER_DOC
from self_rag.utils.common import logger

class FollowUpQuestion(BaseModel):
    """후속 질문 모델"""
    question: str = Field(description="생성된 후속 질문")
    reasoning: str = Field(description="질문이 어떻게 원본 문서의 약점을 보완할 수 있는지 설명")
    focus_area: str = Field(description="질문이 초점을 맞추는 누락된 정보 영역")

class QueryRewriter:
    """질문 재작성 및 후속 질문 생성을 담당하는 클래스"""
    
    def __init__(
        self,
        model_name: str = LLM_MODEL,
        temperature: float = TEMPERATURE,
        max_questions: int = FOLLOW_UP_QUESTIONS_PER_DOC
    ):
        """
        Args:
            model_name: 사용할 LLM 모델 이름
            temperature: 모델 temperature
            max_questions: 생성할 최대 질문 수
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=OPENAI_API_KEY
        )
        self.max_questions = max_questions
        
        # 문서의 약점 분석 및 후속 질문 생성 프롬프트
        self.standard_rewrite_prompt = ChatPromptTemplate.from_template("""
당신은 주어진 질문과 문서를 분석하여 관련성을 높이기 위한 추가 질문을 생성하는 전문가입니다.

원본 질문: {query}
검색된 문서: {document}

위 문서는 원본 질문에 대해 부분적으로만 관련성이 있거나 일부 정보가 부족합니다.
원본 질문에 더 관련된 정보를 검색하기 위한 후속 질문 {max_questions}개를 생성해주세요.

후속 질문들은:
1. 원본 질문의 맥락을 유지해야 합니다.
2. 문서에 누락된 정보나 부족한 측면을 보완할 수 있어야 합니다.
3. 구체적이고 명확해야 합니다.
4. 검색 시스템이 더 관련성 높은 문서를 찾는 데 도움이 되어야 합니다.
5. 서로 다른 측면을 다루어 다양성을 확보해야 합니다.

각 질문에 대해 다음 정보를 제공해주세요:
1. 질문 내용
2. 이 질문이 어떻게 원본 문서의 약점을 보완하는지 설명
3. 질문이 초점을 맞추는 누락된 정보 영역
""")

        # 문서 관련성 평가 정보가 있는 경우의 고급 재작성 프롬프트
        self.advanced_rewrite_prompt = ChatPromptTemplate.from_template("""
당신은 주어진 질문과 문서를 분석하여 관련성을 높이기 위한 추가 질문을 생성하는 전문가입니다.

원본 질문: {query}
검색된 문서: {document}

문서 관련성 평가:
{relevance_evaluation}

위 문서는 원본 질문에 대해 부분적으로만 관련성이 있습니다. 특히 위 관련성 평가에서 
낮은 점수를 받은 분야에 중점을 두고 개선이 필요합니다.

원본 질문에 더 관련된 정보를 검색하기 위한 후속 질문 {max_questions}개를 생성해주세요.

후속 질문들은:
1. 원본 질문의 맥락을 유지해야 합니다.
2. 문서에 누락된 정보나 부족한 측면을 보완할 수 있어야 합니다.
3. 구체적이고 명확해야 합니다.
4. 검색 시스템이 더 관련성 높은 문서를 찾는 데 도움이 되어야 합니다.
5. 서로 다른 측면을 다루어 다양성을 확보해야 합니다.
6. 특히 관련성 평가에서 낮은 점수(0점)를 받은 영역을 중점적으로 보완해야 합니다.

각 질문에 대해 다음 정보를 제공해주세요:
1. 질문 내용
2. 이 질문이 어떻게 원본 문서의 약점을 보완하는지 설명
3. 질문이 초점을 맞추는 누락된 정보 영역
""")
        
        # 여러 문서를 통합 고려하는 통합 재작성 프롬프트
        self.integration_rewrite_prompt = ChatPromptTemplate.from_template("""
당신은 주어진 질문과 여러 문서를 분석하여 관련성을 높이기 위한 추가 질문을 생성하는 전문가입니다.

원본 질문: {query}

이미 검색된 문서들:
{documents}

위 문서들은 원본 질문에 대해 부분적으로만 관련성이 있거나 정보가 충분하지 않습니다.
현재 문서들의 정보를 보완하면서, 중복되지 않고 누락된 중요 정보를 찾을 수 있는 
후속 질문 {max_questions}개를 생성해주세요.

후속 질문들은:
1. 원본 질문의 맥락을 유지해야 합니다.
2. 기존 문서들에 없는 새로운 정보를 찾을 수 있어야 합니다.
3. 구체적이고 명확해야 합니다.
4. 서로 다른 측면을 다루어 다양성을 확보해야 합니다.
5. 현재 문서들의 정보를 통합할 때 발생하는 논리적 간극을 채울 수 있어야 합니다.

각 질문에 대해 다음 정보를 제공해주세요:
1. 질문 내용
2. 이 질문이 어떻게 기존 문서들의 한계를 보완하는지 설명
3. 질문이 초점을 맞추는 누락된 정보 영역
""")
        
        logger.info("QueryRewriter 초기화 완료")
    
    def generate_follow_up_questions(
        self, 
        query: str, 
        document: Dict[str, Any],
        relevance_evaluation: Optional[Dict[str, Any]] = None
    ) -> List[FollowUpQuestion]:
        """문서의 관련성을 높이기 위한 후속 질문을 생성합니다.
        
        Args:
            query: 원본 쿼리
            document: 평가할 문서
            relevance_evaluation: 관련성 평가 결과 (있는 경우)
            
        Returns:
            생성된 후속 질문 목록
        """
        # 문서 내용 추출
        document_content = document.get("page_content", "")
        if not document_content:
            logger.warning("문서 내용이 비어 있습니다.")
            return []
        
        # 관련성 평가 정보가 있는 경우 고급 프롬프트 사용
        if relevance_evaluation and isinstance(relevance_evaluation, dict):
            # 관련성 평가 정보 포맷팅
            relevance_info = ""
            if "criteria_scores" in relevance_evaluation and "criteria_reasoning" in relevance_evaluation:
                for criterion, score in relevance_evaluation["criteria_scores"].items():
                    reasoning = relevance_evaluation["criteria_reasoning"].get(criterion, "")
                    relevance_info += f"- {criterion}: 점수 {score}, 근거: {reasoning}\n"
            
            # 고급 프롬프트 사용
            prompt = self.advanced_rewrite_prompt
            messages = prompt.format_messages(
                query=query,
                document=document_content,
                relevance_evaluation=relevance_info,
                max_questions=self.max_questions
            )
        else:
            # 기본 프롬프트 사용
            prompt = self.standard_rewrite_prompt
            messages = prompt.format_messages(
                query=query,
                document=document_content,
                max_questions=self.max_questions
            )
        
        try:
            # 구조화된 출력을 위한 LLM 설정
            structured_llm = self.llm.with_structured_output(List[FollowUpQuestion])
            
            # LLM에 질의
            response = structured_llm.invoke(messages)
            
            # 결과 로깅
            logger.info(f"쿼리 '{query}'에 대해 {len(response)}개 후속 질문 생성됨")
            for i, q in enumerate(response, 1):
                logger.info(f"후속 질문 {i}: {q.question}")
            
            return response
            
        except Exception as e:
            logger.error(f"후속 질문 생성 중 오류 발생: {str(e)}")
            return []
    
    def generate_integration_questions(
        self, 
        query: str, 
        documents: List[Dict[str, Any]]
    ) -> List[FollowUpQuestion]:
        """여러 문서의 통합 고려하여 후속 질문을 생성합니다.
        
        Args:
            query: 원본 쿼리
            documents: 문서 목록
            
        Returns:
            생성된 후속 질문 목록
        """
        if not documents:
            logger.warning("문서가 제공되지 않았습니다.")
            return []
        
        # 문서 내용 추출 및 포맷팅
        formatted_docs = ""
        for i, doc in enumerate(documents, 1):
            content = doc.get("page_content", "")
            if content:
                formatted_docs += f"\n--- 문서 {i} ---\n{content}\n"
        
        # 통합 프롬프트 사용
        messages = self.integration_rewrite_prompt.format_messages(
            query=query,
            documents=formatted_docs,
            max_questions=self.max_questions
        )
        
        try:
            # 구조화된 출력을 위한 LLM 설정
            structured_llm = self.llm.with_structured_output(List[FollowUpQuestion])
            
            # LLM에 질의
            response = structured_llm.invoke(messages)
            
            # 결과 로깅
            logger.info(f"쿼리 '{query}'와 {len(documents)}개 문서에 대해 {len(response)}개 통합 후속 질문 생성됨")
            
            return response
            
        except Exception as e:
            logger.error(f"통합 후속 질문 생성 중 오류 발생: {str(e)}")
            return [] 