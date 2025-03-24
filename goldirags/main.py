import os
import argparse
import asyncio
from dotenv import load_dotenv

from goldirags.pipeline import GoldiRAGsPipeline
from goldirags.retriever.vector_retriever import VectorRetriever
from goldirags.evaluator.relevance_evaluator import RelevanceEvaluator
from goldirags.evaluator.support_evaluator import SupportEvaluator
from goldirags.evaluator.use_evaluator import UseEvaluator
from goldirags.generator.llm_generator import LLMGenerator
from goldirags.utils.common import logger

def create_pipeline():
    """GoldiRAGs 파이프라인을 생성합니다."""
    # 컴포넌트 초기화
    retriever = VectorRetriever()
    relevance_evaluator = RelevanceEvaluator()
    support_evaluator = SupportEvaluator()
    use_evaluator = UseEvaluator()
    generator = LLMGenerator()
    
    # 파이프라인 생성
    pipeline = GoldiRAGsPipeline(
        retriever=retriever,
        relevance_evaluator=relevance_evaluator,
        support_evaluator=support_evaluator,
        use_evaluator=use_evaluator,
        generator=generator
    )
    
    return pipeline

async def process_query(pipeline, query, top_k=3):
    """쿼리를 처리합니다."""
    result = await pipeline.run(query, top_k=top_k)
    
    # 결과 출력
    print("\n=== GoldiRAGs 응답 ===")
    print(result["response"])
    print(f"\n검색된 문서: {len(result['retrieved_docs'])}개")
    print(f"사용된 문서: {len(result['filtered_docs'])}개")
    print(f"처리 시간: {result['processing_time']:.2f}초")
    
    return result

async def interactive_mode(pipeline, top_k=3):
    """대화형 모드로 실행합니다."""
    print("GoldiRAGs 시스템에 오신 것을 환영합니다! 종료하려면 'exit' 또는 'quit'를 입력하세요.")
    
    while True:
        query = input("\n질문을 입력하세요: ")
        if query.lower() in ["exit", "quit"]:
            break
            
        await process_query(pipeline, query, top_k)

async def main_async():
    """비동기 메인 함수"""
    # 환경 변수 로드
    load_dotenv()
    
    # 인자 파싱
    parser = argparse.ArgumentParser(description="GoldiRAGs 파이프라인 실행")
    parser.add_argument("--query", type=str, help="사용자 쿼리")
    parser.add_argument("--files", nargs="+", help="로드할 텍스트 파일 경로 목록")
    parser.add_argument("--top_k", type=int, default=3, help="검색할 최대 문서 수")
    args = parser.parse_args()
    
    # 파이프라인 생성
    pipeline = create_pipeline()
    
    # 파일 로드 (제공된 경우)
    if args.files:
        logger.info(f"파일 {args.files}를 로드합니다")
        retriever = VectorRetriever.from_text_files(args.files)
        pipeline.retriever = retriever
    
    # 쿼리 처리
    if args.query:
        logger.info(f"쿼리: {args.query}")
        await process_query(pipeline, args.query, args.top_k)
    else:
        # 대화형 모드
        await interactive_mode(pipeline, args.top_k)

def main():
    """메인 함수"""
    # 비동기 메인 함수 실행
    asyncio.run(main_async())

if __name__ == "__main__":
    main() 