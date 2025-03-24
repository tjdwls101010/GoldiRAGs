from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseRetriever(ABC):
    """검색기(Retriever)의 기본 인터페이스"""
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """쿼리에 관련된 문서를 검색합니다.
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 최대 문서 수
            
        Returns:
            검색된 문서 목록
        """
        pass
    
    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """문서를 검색 인덱스에 추가합니다.
        
        Args:
            documents: 추가할 문서 목록
        """
        pass 