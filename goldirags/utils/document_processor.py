from typing import List, Dict, Any, Tuple, Optional
import asyncio
from pydantic import BaseModel
from langchain.docstore.document import Document

from self_rag.config import DISCARD_THRESHOLD, KEEP_THRESHOLD, MAX_ENHANCEMENT_ATTEMPTS, MIN_SCORE_IMPROVEMENT
from self_rag.utils.common import logger

class DocumentProcessingResult(BaseModel):
    """문서 처리 결과"""
    action: str  # "discard", "keep", "enhance" 중 하나
    document: Dict[str, Any]
    score: float
    enhanced_documents: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

async def process_document_by_relevance(
    query: str,
    document: Dict[str, Any],
    relevance_evaluator,
    metadata: Optional[Dict[str, Any]] = None
) -> DocumentProcessingResult:
    """관련성 점수에 따라 문서를 처리합니다.
    
    Args:
        query: 사용자 쿼리
        document: 평가할 문서
        relevance_evaluator: 관련성 평가기
        metadata: 문서 메타데이터
        
    Returns:
        처리 결과 객체
    """
    # 문서 관련성 평가
    evaluation = await asyncio.to_thread(relevance_evaluator.evaluate, query, document)
    
    # DetailedRelevanceScore 객체인 경우
    if hasattr(evaluation, 'total_score'):
        relevance_score = evaluation.total_score
        evaluation_details = {
            "criteria_scores": evaluation.criteria_scores,
            "criteria_reasoning": evaluation.criteria_reasoning,
            "total_score": evaluation.total_score
        }
    else:
        # 단순 float나 bool인 경우
        relevance_score = float(evaluation) if isinstance(evaluation, (float, int, bool)) else 0.0
        evaluation_details = {"total_score": relevance_score}
    
    # 각 임계값에 따른 처리
    if relevance_score < DISCARD_THRESHOLD:
        # 관련성이 낮은 문서는 폐기
        logger.info(f"문서 폐기 (점수: {relevance_score:.2f} < {DISCARD_THRESHOLD})")
        return DocumentProcessingResult(
            action="discard", 
            document=document, 
            score=relevance_score,
            metadata=metadata
        )
    elif relevance_score >= KEEP_THRESHOLD:
        # 관련성이 높은 문서는 유지
        logger.info(f"문서 유지 (점수: {relevance_score:.2f} >= {KEEP_THRESHOLD})")
        return DocumentProcessingResult(
            action="keep", 
            document=document, 
            score=relevance_score,
            metadata=metadata
        )
    else:
        # 관련성이 중간인 문서는 증강 대상
        logger.info(f"문서 증강 대상 (점수: {relevance_score:.2f}, 구간: {DISCARD_THRESHOLD}~{KEEP_THRESHOLD})")
        return DocumentProcessingResult(
            action="enhance", 
            document=document, 
            score=relevance_score, 
            enhanced_documents={"relevance_evaluation": evaluation_details},
            metadata=metadata
        )

def merge_documents(
    documents: List[Dict[str, Any]], 
    remove_duplicates: bool = True
) -> Dict[str, Any]:
    """여러 문서를 하나로 통합합니다.
    
    Args:
        documents: 통합할 문서 목록
        remove_duplicates: 중복 문장 제거 여부
        
    Returns:
        통합된 문서
    """
    if not documents:
        return {"page_content": "", "metadata": {}}
    
    # 모든 문서의 내용 추출
    contents = []
    metadata = {}
    
    for doc in documents:
        content = doc.get("page_content", "")
        if content:
            contents.append(content)
        
        # 메타데이터 통합
        doc_metadata = doc.get("metadata", {})
        for key, value in doc_metadata.items():
            if key not in metadata:
                metadata[key] = value
            elif isinstance(metadata[key], list):
                if isinstance(value, list):
                    metadata[key].extend(value)
                else:
                    metadata[key].append(value)
            else:
                metadata[key] = [metadata[key], value]
    
    # 내용 통합
    if remove_duplicates:
        # 중복 문장 제거
        unique_sentences = set()
        unique_contents = []
        
        for content in contents:
            sentences = content.split('.')
            filtered_sentences = []
            
            for sentence in sentences:
                clean_sentence = sentence.strip()
                if clean_sentence and clean_sentence not in unique_sentences:
                    unique_sentences.add(clean_sentence)
                    filtered_sentences.append(clean_sentence)
            
            if filtered_sentences:
                unique_contents.append('. '.join(filtered_sentences) + '.')
        
        merged_content = ' '.join(unique_contents)
    else:
        # 단순 연결
        merged_content = ' '.join(contents)
    
    # 통합 정보 추가
    metadata["merged_document"] = True
    metadata["source_count"] = len(documents)
    
    return {
        "page_content": merged_content,
        "metadata": metadata
    }

def restructure_documents(
    main_document: Dict[str, Any],
    supporting_documents: List[Dict[str, Any]],
    relevance_scores: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """핵심 문서와 지원 문서를 구조화하여 재구성합니다.
    
    Args:
        main_document: 핵심 문서
        supporting_documents: 지원 문서 목록
        relevance_scores: 문서별 관련성 점수
        
    Returns:
        재구성된 문서
    """
    # 메인 문서 내용
    main_content = main_document.get("page_content", "")
    if not main_content:
        return merge_documents([main_document] + supporting_documents)
    
    # 지원 문서가 없는 경우 메인 문서만 반환
    if not supporting_documents:
        return main_document
    
    # 문서 내용과 점수 추출
    supporting_contents = []
    for i, doc in enumerate(supporting_documents):
        content = doc.get("page_content", "")
        score = relevance_scores.get(str(i), 0) if relevance_scores else 0
        
        if content:
            supporting_contents.append((content, score))
    
    # 점수에 따라 정렬
    supporting_contents.sort(key=lambda x: x[1], reverse=True)
    
    # 재구성된 내용 생성
    restructured_content = main_content + "\n\n추가 정보:\n"
    for content, _ in supporting_contents:
        restructured_content += f"\n- {content}\n"
    
    # 메타데이터 통합
    metadata = main_document.get("metadata", {}).copy()
    metadata["restructured"] = True
    metadata["supporting_docs_count"] = len(supporting_documents)
    
    return {
        "page_content": restructured_content,
        "metadata": metadata
    }

def identify_redundant_information(documents: List[Dict[str, Any]]) -> List[Tuple[int, int, float]]:
    """문서 간 중복 정보를 식별합니다.
    
    Args:
        documents: 문서 목록
        
    Returns:
        중복 관계 목록 (문서1 인덱스, 문서2 인덱스, 중복도)
    """
    if len(documents) < 2:
        return []
    
    redundancies = []
    
    # 모든 문서 쌍에 대해 중복 검사
    for i in range(len(documents)):
        for j in range(i+1, len(documents)):
            content_i = documents[i].get("page_content", "")
            content_j = documents[j].get("page_content", "")
            
            if not content_i or not content_j:
                continue
            
            # 문장 단위로 분할
            sentences_i = [s.strip() for s in content_i.split('.') if s.strip()]
            sentences_j = [s.strip() for s in content_j.split('.') if s.strip()]
            
            # 공통 문장 수 계산
            common_sentences = set(sentences_i) & set(sentences_j)
            
            # 중복도 계산
            if sentences_i and sentences_j:
                redundancy = len(common_sentences) / min(len(sentences_i), len(sentences_j))
                if redundancy > 0.3:  # 30% 이상 중복시 기록
                    redundancies.append((i, j, redundancy))
    
    return redundancies

def filter_redundant_documents(documents: List[Dict[str, Any]], max_redundancy: float = 0.7) -> List[Dict[str, Any]]:
    """중복 정보가 많은 문서를 필터링합니다.
    
    Args:
        documents: 문서 목록
        max_redundancy: 최대 허용 중복도
        
    Returns:
        필터링된 문서 목록
    """
    if len(documents) < 2:
        return documents
    
    # 중복 관계 식별
    redundancies = identify_redundant_information(documents)
    
    # 중복도가 높은 문서들의 인덱스 추적
    redundant_indices = set()
    for i, j, redundancy in redundancies:
        if redundancy > max_redundancy:
            # 더 짧은 문서를 중복으로 표시
            content_i = documents[i].get("page_content", "")
            content_j = documents[j].get("page_content", "")
            
            if len(content_i) <= len(content_j):
                redundant_indices.add(i)
            else:
                redundant_indices.add(j)
    
    # 중복이 아닌 문서만 선택
    filtered_docs = [doc for i, doc in enumerate(documents) if i not in redundant_indices]
    
    logger.info(f"{len(documents)}개 문서 중 {len(filtered_docs)}개 남음 (중복 {len(redundant_indices)}개 제거)")
    return filtered_docs 