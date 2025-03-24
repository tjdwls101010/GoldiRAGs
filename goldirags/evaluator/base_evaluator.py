from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple

class BaseEvaluator(ABC):
    """문서 평가자의 기본 인터페이스"""
    
    @abstractmethod
    def evaluate(
        self, 
        query: str, 
        document: Dict[str, Any]
    ) -> Union[bool, float, Dict[str, Any]]:
        """문서를 평가합니다.
        
        Args:
            query: 사용자 쿼리
            document: 평가할 문서
            
        Returns:
            평가 결과 (불리언, 점수 또는 상세 결과 딕셔너리)
        """
        pass 