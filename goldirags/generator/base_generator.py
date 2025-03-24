from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseGenerator(ABC):
    """응답 생성기의 기본 인터페이스"""
    
    @abstractmethod
    def generate(
        self, 
        query: str, 
        documents: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """쿼리와 문서를 바탕으로 응답을 생성합니다.
        
        Args:
            query: 사용자 쿼리
            documents: 사용할 문서 목록 (선택 사항)
            
        Returns:
            생성된 응답
        """
        pass 