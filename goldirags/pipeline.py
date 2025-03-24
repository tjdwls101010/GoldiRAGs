from typing import List, Dict, Any, Optional, Tuple
import asyncio
import time
import logging

from goldirags.config import TOP_K, DISCARD_THRESHOLD, KEEP_THRESHOLD, MAX_ENHANCEMENT_ATTEMPTS, MIN_SCORE_IMPROVEMENT
from goldirags.utils.common import logger
from goldirags.utils.document_processor import (
    process_document_by_relevance, 
    merge_documents, 
    restructure_documents,
    filter_redundant_documents
)
from goldirags.retriever.base_retriever import BaseRetriever
from goldirags.retriever.query_rewriter import QueryRewriter
from goldirags.evaluator.relevance_evaluator import RelevanceEvaluator
from goldirags.evaluator.support_evaluator import SupportEvaluator
from goldirags.evaluator.use_evaluator import UseEvaluator
from goldirags.generator.base_generator import BaseGenerator

class GoldiRAGsPipeline:
    """GoldiRAGs 파이프라인"""
    
    def __init__(
        self,
        retriever: BaseRetriever,
        relevance_evaluator: RelevanceEvaluator,
        support_evaluator: SupportEvaluator,
        use_evaluator: UseEvaluator,
        generator: BaseGenerator
    ):
        """
        Args:
            retriever: 문서 검색기
            relevance_evaluator: 관련성 평가기 (ISREL)
            support_evaluator: 지원성 평가기 (ISSUP)
            use_evaluator: 사용 결정 평가기 (ISUSE)
            generator: 응답 생성기
        """
        self.retriever = retriever
        self.relevance_evaluator = relevance_evaluator
        self.support_evaluator = support_evaluator
        self.use_evaluator = use_evaluator
        self.generator = generator
        self.query_rewriter = QueryRewriter()
        
        logger.info("GoldiRAGsPipeline 초기화 완료")
    
    async def enhance_document(
        self, 
        query: str, 
        document: Dict[str, Any], 
        relevance_score: float,
        relevance_evaluation: Optional[Dict] = None
    ) -> Tuple[Dict[str, Any], float]:
        """문서의 관련성을 강화하기 위해 후속 질문을 생성하고 추가 문서를 검색합니다.
        
        Args:
            query: 사용자 쿼리
            document: 강화할 문서
            relevance_score: 현재 관련성 점수
            relevance_evaluation: 상세 관련성 평가 결과
            
        Returns:
            (강화된 문서, 새로운 관련성 점수) 튜플
        """
        logger.info(f"문서 강화 시작 (현재 점수: {relevance_score:.2f})")
        
        # 현재 문서 정보 유지
        current_score = relevance_score
        current_document = document
        attempt_count = 0
        
        # 추가 문서를 저장할 목록
        additional_documents = []
        
        # 점수가 기준치 미만이고 최대 시도 횟수에 도달하지 않은 동안 반복
        while current_score < KEEP_THRESHOLD and attempt_count < MAX_ENHANCEMENT_ATTEMPTS:
            attempt_count += 1
            logger.info(f"강화 시도 {attempt_count}/{MAX_ENHANCEMENT_ATTEMPTS}")
            
            # 후속 질문 생성
            follow_up_questions = self.query_rewriter.generate_follow_up_questions(
                query=query,
                document=current_document,
                relevance_evaluation=relevance_evaluation
            )
            
            if not follow_up_questions:
                logger.warning("후속 질문을 생성할 수 없어 증강 중단")
                break
            
            # 각 후속 질문으로 문서 검색
            new_documents = []
            for follow_up in follow_up_questions:
                logger.info(f"후속 질문으로 검색: '{follow_up.question}'")
                
                # 검색 수행
                retrieved_docs = self.retriever.retrieve(follow_up.question)
                if retrieved_docs:
                    # 검색된 문서를 새 문서 목록에 추가
                    new_documents.extend(retrieved_docs)
                    
                    # 메타데이터에 후속 질문 정보 추가
                    for doc in retrieved_docs:
                        if "metadata" not in doc:
                            doc["metadata"] = {}
                        doc["metadata"]["follow_up_question"] = follow_up.question
                        doc["metadata"]["follow_up_reasoning"] = follow_up.reasoning
                        doc["metadata"]["follow_up_focus"] = follow_up.focus_area
            
            if not new_documents:
                logger.warning("추가 문서를 찾을 수 없어 증강 중단")
                break
            
            # 중복 문서 필터링
            filtered_new_docs = filter_redundant_documents(new_documents)
            additional_documents.extend(filtered_new_docs)
            
            # 원본 문서와 추가 문서 통합
            all_docs = [current_document] + additional_documents
            merged_doc = merge_documents(all_docs)
            
            # 통합 문서 재평가
            evaluation = await asyncio.to_thread(
                self.relevance_evaluator.evaluate, 
                query, 
                merged_doc
            )
            
            # DetailedRelevanceScore 객체인 경우
            if hasattr(evaluation, 'total_score'):
                new_score = evaluation.total_score
                new_evaluation_details = {
                    "criteria_scores": evaluation.criteria_scores,
                    "criteria_reasoning": evaluation.criteria_reasoning,
                    "total_score": evaluation.total_score
                }
            else:
                # 단순 float나 bool인 경우
                new_score = float(evaluation) if isinstance(evaluation, (float, int, bool)) else 0.0
                new_evaluation_details = {"total_score": new_score}
            
            # 점수 향상 확인
            score_improvement = new_score - current_score
            logger.info(f"재평가 결과: 점수 {current_score:.2f} -> {new_score:.2f} (향상: {score_improvement:.2f})")
            
            # 점수가 충분히 향상되었는지 확인
            if score_improvement < MIN_SCORE_IMPROVEMENT:
                logger.info(f"점수 향상이 불충분함 ({score_improvement:.2f} < {MIN_SCORE_IMPROVEMENT}), 현재 문서 유지")
                break
            
            # 점수가 기준점을 넘었거나 충분히 향상된 경우 업데이트
            current_score = new_score
            current_document = merged_doc
            relevance_evaluation = new_evaluation_details
            
            # 점수가 충분히 높아지면 종료
            if current_score >= KEEP_THRESHOLD:
                logger.info(f"충분한 관련성 달성 (점수: {current_score:.2f} >= {KEEP_THRESHOLD})")
                break
        
        # 증강 정보 추가
        if "metadata" not in current_document:
            current_document["metadata"] = {}
            
        current_document["metadata"].update({
            "enhanced": True,
            "original_score": relevance_score,
            "enhanced_score": current_score,
            "enhancement_attempts": attempt_count,
            "additional_documents_count": len(additional_documents)
        })
        
        return current_document, current_score
    
    async def run(self, query: str, top_k: int = TOP_K) -> Dict[str, Any]:
        """GoldiRAGs 파이프라인을 실행합니다.
        
        Args:
            query: 사용자 쿼리
            top_k: 검색할 최대 문서 수
            
        Returns:
            결과 및 메타데이터를 포함한 딕셔너리
        """
        start_time = time.time()
        logger.info(f"쿼리 '{query}'에 대한 GoldiRAGs 파이프라인 실행")
        
        # 1. 문서 검색 (Retrieve-Token)
        retrieved_docs = self.retriever.retrieve(query, top_k=top_k)
        logger.info(f"{len(retrieved_docs)}개 문서 검색됨")
        
        if not retrieved_docs:
            logger.warning("검색된 문서가 없습니다.")
            # 문서 없이 응답 생성
            response = self.generator.generate(query)
            return {
                "query": query,
                "response": response,
                "retrieved_docs": [],
                "filtered_docs": [],
                "evaluation_results": [],
                "processing_time": time.time() - start_time
            }
        
        # 2. 문서 관련성 평가 및 처리 (병렬 처리)
        processing_tasks = []
        for doc in retrieved_docs:
            task = process_document_by_relevance(
                query=query,
                document=doc,
                relevance_evaluator=self.relevance_evaluator,
                metadata=doc.get("metadata", {})
            )
            processing_tasks.append(task)
        
        # 모든 처리 작업 실행
        processing_results = await asyncio.gather(*processing_tasks)
        
        # 3. 결과 처리 및 문서 증강
        kept_docs = []
        enhanced_docs = []
        discard_docs = []
        evaluation_results = []
        
        for result in processing_results:
            evaluation_results.append({
                "document": result.document,
                "action": result.action,
                "score": result.score,
                "metadata": result.metadata
            })
            
            if result.action == "keep":
                # 관련성 높은 문서는 바로 추가
                kept_docs.append(result.document)
            elif result.action == "enhance":
                # 관련성 중간 문서는 증강 대상
                enhanced_docs.append((
                    result.document, 
                    result.score, 
                    result.enhanced_documents
                ))
            else:
                # 관련성 낮은 문서는 폐기
                discard_docs.append(result.document)
        
        logger.info(f"문서 처리 결과: 유지 {len(kept_docs)}개, 증강 {len(enhanced_docs)}개, 폐기 {len(discard_docs)}개")
        
        # 4. 문서 증강 (병렬 처리)
        if enhanced_docs:
            enhancement_tasks = []
            for doc, score, enhanced_info in enhanced_docs:
                relevance_evaluation = enhanced_info.get("relevance_evaluation") if enhanced_info else None
                
                task = self.enhance_document(
                    query=query,
                    document=doc,
                    relevance_score=score,
                    relevance_evaluation=relevance_evaluation
                )
                enhancement_tasks.append(task)
            
            # 모든 증강 작업 실행
            enhancement_results = await asyncio.gather(*enhancement_tasks)
            
            # 증강 결과 처리
            for enhanced_doc, new_score in enhancement_results:
                if new_score >= KEEP_THRESHOLD:
                    # 충분히 증강된 문서만 추가
                    kept_docs.append(enhanced_doc)
                    logger.info(f"증강된 문서 추가 (점수: {new_score:.2f})")
                else:
                    logger.info(f"증강 후에도 점수가 불충분함 (점수: {new_score:.2f}), 문서 제외")
        
        # 5. 임시 응답 생성 (지원성 평가용)
        temp_response = self.generator.generate(query)
        
        # 6. 지원성 및 사용 여부 평가 (ISSUP/ISUSE-Token)
        final_docs = []
        for doc in kept_docs:
            # 지원성 평가 (ISSUP-Token)
            is_supporting = self.support_evaluator.evaluate(query, doc, temp_response)
            
            # 사용 결정 (ISUSE-Token)
            should_use = self.use_evaluator.evaluate(
                query, doc, is_relevant=True, is_supporting=is_supporting
            )
            
            # 결과에 평가 추가
            evaluation_results.append({
                "document": doc,
                "is_supporting": is_supporting,
                "should_use": should_use,
                "enhanced": doc.get("metadata", {}).get("enhanced", False)
            })
            
            # 사용 결정이 True인 문서만 필터링
            if should_use:
                final_docs.append(doc)
        
        logger.info(f"{len(final_docs)}개 문서가 최종 응답 생성에 사용됨")
        
        # 7. 중복 정보 제거 및 문서 구조화
        if len(final_docs) > 1:
            filtered_final_docs = filter_redundant_documents(final_docs)
            
            # 메인 문서와 지원 문서 선택
            if filtered_final_docs:
                # 첫 번째 문서를 메인 문서로 선택
                main_doc = filtered_final_docs[0]
                supporting_docs = filtered_final_docs[1:] if len(filtered_final_docs) > 1 else []
                
                # 문서 재구성
                relevance_scores = {}
                for i, doc in enumerate(supporting_docs):
                    # metadata에서 점수 추출 또는 기본값 사용
                    score = doc.get("metadata", {}).get("enhanced_score", 0.7)
                    relevance_scores[str(i)] = score
                
                # 문서 구조화
                structured_doc = restructure_documents(
                    main_doc, supporting_docs, relevance_scores
                )
                
                # 최종 문서 목록 업데이트
                final_docs = [structured_doc]
            else:
                logger.warning("중복 제거 후 문서가 없습니다.")
        
        # 8. 최종 응답 생성 (Generator)
        final_response = self.generator.generate(query, final_docs)
        
        # 9. 처리 시간 계산 및 결과 반환
        end_time = time.time()
        processing_time = end_time - start_time
        
        result = {
            "query": query,
            "response": final_response,
            "retrieved_docs": retrieved_docs,
            "filtered_docs": final_docs,
            "evaluation_results": evaluation_results,
            "processing_time": processing_time
        }
        
        logger.info(f"GoldiRAGs 파이프라인 완료 (소요시간: {processing_time:.2f}초)")
        return result 