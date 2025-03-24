from typing import Dict, Any, Union, List
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from self_rag.evaluator.base_evaluator import BaseEvaluator
from self_rag.config import OPENAI_API_KEY, LLM_MODEL, TEMPERATURE, RELEVANCE_THRESHOLD
from self_rag.utils.common import logger, extract_boolean_from_response

# 관련성 평가 기준 및 가중치 정의
RELEVANCE_CRITERIA_WEIGHTS = {
    "topic_relevance": 1.2,     # 주제 관련성
    "information_value": 1.5,   # 정보 가치
    "factual_quality": 1.3,     # 사실적 품질
    "completeness": 1.0,        # 완전성
    "temporal_relevance": 0.8   # 시간적 관련성
}

# 가중치의 합 계산
TOTAL_WEIGHTS = sum(RELEVANCE_CRITERIA_WEIGHTS.values())

# 각 기준별 개별 평가 모델
class CriterionScore(BaseModel):
    score: int = Field(description="관련성 점수 (0 또는 1)")
    reasoning: str = Field(description="평가 근거")

# 최종 통합 관련성 점수 모델
class DetailedRelevanceScore(BaseModel):
    criteria_scores: Dict[str, int] = Field(description="각 관련성 기준에 대한 점수 (0 또는 1)")
    criteria_reasoning: Dict[str, str] = Field(description="각 기준 점수에 대한 판단 근거")
    total_score: float = Field(description="전체 관련성 점수 (0~1)")

class RelevanceEvaluator(BaseEvaluator):
    """다차원 관련성 평가를 수행하는 ISREL-Token 구현"""
    
    def __init__(
        self,
        model_name: str = LLM_MODEL,
        temperature: float = TEMPERATURE,
        threshold: float = RELEVANCE_THRESHOLD
    ):
        """
        Args:
            model_name: 사용할 LLM 모델 이름
            temperature: 모델 temperature
            threshold: 관련성 판단 임계값
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=OPENAI_API_KEY
        )
        self.threshold = threshold
        
        # 각 기준별 프롬프트 템플릿
        self.topic_relevance_template = """
당신은 문서가 질문의 주제와 얼마나 관련이 있는지 평가하는 전문가입니다.

질문: {query}
문서: {document}

주제 관련성(topic_relevance)을 평가해주세요:
- 문서가 질문의 주제 및 핵심 키워드와 관련되어 있는지 평가하세요
- 질문의 핵심 개념이나 키워드가 문서에 포함되어 있는지 확인하세요
- 문서의 전반적인 내용이 질문의 주제와 일치하는지 판단하세요

0(전혀 관련 없음) 또는 1(관련 있음)으로 점수를 매기고, 그 이유를 설명하세요.
"""

        self.information_value_template = """
당신은 문서가 질문에 얼마나 유용한 정보를 제공하는지 평가하는 전문가입니다.

질문: {query}
문서: {document}

정보 가치(information_value)를 평가해주세요:
- 문서가 질문에 직접적으로 답하는 내용을 포함하는지 평가하세요
- 문서의 정보가 질문 해결에 실질적으로 도움이 되는지 판단하세요
- 제공된 정보의 구체성과 명확성을 고려하세요

0(유용한 정보 없음) 또는 1(유용한 정보 있음)으로 점수를 매기고, 그 이유를 설명하세요.
"""

        self.factual_quality_template = """
당신은 문서 내용의 사실적 품질을 평가하는 전문가입니다.

질문: {query}
문서: {document}

사실적 품질(factual_quality)을 평가해주세요:
- 문서의 정보가 사실에 기반하고 있는지 평가하세요
- 정보의 출처나 근거가 명확하게 제시되어 있는지 확인하세요
- 내용이 논리적이고 일관성이 있는지 판단하세요

0(낮은 품질) 또는 1(높은 품질)으로 점수를 매기고, 그 이유를 설명하세요.
"""

        self.completeness_template = """
당신은 문서가 질문에 대해 얼마나 완전한 답변을 제공하는지 평가하는 전문가입니다.

질문: {query}
문서: {document}

완전성(completeness)을 평가해주세요:
- 문서가 질문의 모든 측면을 다루고 있는지 평가하세요
- 추가 정보 없이도 질문에 답할 수 있을 만큼 충분한지 판단하세요
- 중요한 세부 사항이 누락되었는지 확인하세요

0(불완전함) 또는 1(완전함)으로 점수를 매기고, 그 이유를 설명하세요.
"""

        self.temporal_relevance_template = """
당신은 문서의 정보가 시간적으로 적절한지 평가하는 전문가입니다.

질문: {query}
문서: {document}

시간적 관련성(temporal_relevance)을 평가해주세요:
- 질문이 특정 시간대나 최신 정보를 요구한다면, 문서가 그에 부합하는지 평가하세요
- 정보의 시간적 맥락이 질문에 적합한지 판단하세요
- 시간적 요소가 질문에 중요하지 않은 경우 기본적으로 관련성이 있다고 판단하세요

0(시간적으로 부적절함) 또는 1(시간적으로 적절함)으로 점수를 매기고, 그 이유를 설명하세요.
"""
        
        logger.info("다차원 RelevanceEvaluator 초기화 완료")
    
    def evaluate_criterion(self, query: str, document: str, criterion: str) -> CriterionScore:
        """개별 기준에 대한 평가를 수행합니다."""
        # 기준별 프롬프트 선택
        template = getattr(self, f"{criterion}_template", self.topic_relevance_template)
        
        # 프롬프트 설정
        prompt = ChatPromptTemplate.from_template(template)
        
        # LLM 설정
        criterion_llm = self.llm.with_structured_output(CriterionScore)
        
        # 입력 생성
        messages = prompt.format_messages(
            query=query,
            document=document
        )
        
        try:
            # LLM에 질의
            response = criterion_llm.invoke(messages)
            logger.info(f"기준 '{criterion}' 평가 점수: {response.score} - {response.reasoning[:30]}...")
            return response
            
        except Exception as e:
            logger.error(f"기준 '{criterion}' 평가 중 오류 발생: {str(e)}")
            # 오류 발생 시 기본값 반환
            return CriterionScore(score=0, reasoning=f"평가 중 오류 발생: {str(e)}")
    
    def evaluate(self, query: str, document: Dict[str, Any]) -> Union[bool, float, DetailedRelevanceScore]:
        """문서의 관련성을 다차원으로 평가합니다.
        
        Args:
            query: 사용자 쿼리
            document: 평가할 문서
            
        Returns:
            통합된 관련성 점수와 세부 평가 결과
        """
        # 문서 내용 추출
        document_content = document.get("page_content", "")
        if not document_content:
            logger.warning("문서 내용이 비어 있습니다.")
            return 0.0
        
        # 각 기준별 평가 수행
        criteria_scores = {}
        criteria_reasoning = {}
        
        for criterion in RELEVANCE_CRITERIA_WEIGHTS.keys():
            result = self.evaluate_criterion(query, document_content, criterion)
            criteria_scores[criterion] = result.score
            criteria_reasoning[criterion] = result.reasoning
        
        # 가중치를 적용한 총점 계산
        weighted_score = 0
        for criterion, score in criteria_scores.items():
            weight = RELEVANCE_CRITERIA_WEIGHTS.get(criterion, 1.0)
            weighted_score += score * weight
        
        # 정규화된 가중 점수 계산
        normalized_weighted_score = weighted_score / TOTAL_WEIGHTS
        
        logger.info(f"쿼리 '{query}'에 대한 문서 다차원 관련성 평가: {normalized_weighted_score:.2f}")
        
        # 결과 반환
        detailed_score = DetailedRelevanceScore(
            criteria_scores=criteria_scores,
            criteria_reasoning=criteria_reasoning,
            total_score=normalized_weighted_score
        )
        
        # 상세 평가 결과와 불리언 값 모두 반환
        return detailed_score 