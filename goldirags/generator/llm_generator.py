from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from self_rag.generator.base_generator import BaseGenerator
from self_rag.config import OPENAI_API_KEY, LLM_MODEL, TEMPERATURE, MAX_NEW_TOKENS
from self_rag.utils.common import logger, format_retrieved_documents

class LLMGenerator(BaseGenerator):
    """LLM을 사용한 응답 생성기 구현"""
    
    def __init__(
        self,
        model_name: str = LLM_MODEL,
        temperature: float = TEMPERATURE,
        max_new_tokens: int = MAX_NEW_TOKENS
    ):
        """
        Args:
            model_name: 사용할 LLM 모델 이름
            temperature: 모델 temperature
            max_new_tokens: 최대 생성 토큰 수
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=OPENAI_API_KEY,
            max_tokens=max_new_tokens
        )
        
        # 문서가 있는 경우와 없는 경우의 프롬프트 템플릿
        self.with_docs_prompt = ChatPromptTemplate.from_template("""
당신은 유용하고 정확한 정보를 제공하는 도우미 AI입니다.
제공된 문서를 참조하여 사용자의 질문에 정확하게 답변해야 합니다.

사용자 질문: {query}

참조 문서:
{formatted_documents}

위 문서의 정보를 바탕으로 사용자 질문에 답변해주세요.
문서에서 확인할 수 없는 내용은 추측하지 말고, 알고 있는 내용만 정확하게 답변해주세요.
답변은 사실적이고 도움이 되어야 합니다.
""")
        
        self.no_docs_prompt = ChatPromptTemplate.from_template("""
당신은 유용하고 정확한 정보를 제공하는 도우미 AI입니다.
사용자의 질문에 최선을 다해 답변해주세요.

사용자 질문: {query}

위 질문에 답변해주세요.
확실하지 않은 정보는 추측하지 말고, 알고 있는 내용만 정확하게 답변해주세요.
답변은 사실적이고 도움이 되어야 합니다.
""")
        
        logger.info("LLMGenerator 초기화 완료")
    
    def generate(self, query: str, documents: Optional[List[Dict[str, Any]]] = None) -> str:
        """쿼리와 문서를 바탕으로 응답을 생성합니다.
        
        Args:
            query: 사용자 쿼리
            documents: 사용할 문서 목록 (선택 사항)
            
        Returns:
            생성된 응답
        """
        if documents and len(documents) > 0:
            # 문서가 있는 경우
            formatted_docs = format_retrieved_documents(documents)
            messages = self.with_docs_prompt.format_messages(
                query=query,
                formatted_documents=formatted_docs
            )
            logger.info(f"문서 {len(documents)}개를 사용하여 응답 생성")
        else:
            # 문서가 없는 경우
            messages = self.no_docs_prompt.format_messages(query=query)
            logger.info("문서 없이 응답 생성")
        
        try:
            # LLM에 질의
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"응답 생성 중 오류 발생: {str(e)}")
            return f"응답 생성 중 오류가 발생했습니다: {str(e)}" 