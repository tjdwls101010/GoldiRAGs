import numpy as np
from typing import List, Dict, Any, Optional
import logging
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from self_rag.retriever.base_retriever import BaseRetriever
from self_rag.config import OPENAI_API_KEY, EMBEDDING_MODEL, RETRIEVER_THRESHOLD
from self_rag.utils.common import logger

class VectorRetriever(BaseRetriever):
    """벡터 저장소를 사용한 검색기 구현"""
    
    def __init__(
        self, 
        embedding_model: str = EMBEDDING_MODEL,
        similarity_threshold: float = RETRIEVER_THRESHOLD
    ):
        """
        Args:
            embedding_model: 임베딩 모델 이름
            similarity_threshold: 유사도 임계값
        """
        self.embedding = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=OPENAI_API_KEY
        )
        self.similarity_threshold = similarity_threshold
        self.vector_store = None
        logger.info("VectorRetriever 초기화 완료")
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """문서를 벡터 저장소에 추가합니다.
        
        Args:
            documents: 추가할 문서 목록. 각 문서는 'page_content'와 'metadata' 키를 포함해야 함
        """
        if not documents:
            logger.warning("추가할 문서가 없습니다.")
            return
        
        if self.vector_store is None:
            # 벡터 저장소 초기화
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embedding
            )
            logger.info(f"{len(documents)}개 문서로 벡터 저장소 생성 완료")
        else:
            # 기존 벡터 저장소에 문서 추가
            self.vector_store.add_documents(documents)
            logger.info(f"{len(documents)}개 문서를 벡터 저장소에 추가 완료")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """쿼리에 관련된 문서를 검색합니다.
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 최대 문서 수
            
        Returns:
            검색된 문서 목록. 각 문서는 페이지 내용과 메타데이터를 포함
        """
        if self.vector_store is None:
            logger.warning("벡터 저장소가 초기화되지 않았습니다.")
            return []
        
        # 유사도 점수와 함께 문서 검색
        docs_with_scores = self.vector_store.similarity_search_with_score(
            query=query,
            k=top_k
        )
        
        # 임계값 이상의 문서만 필터링
        filtered_docs = []
        for doc, score in docs_with_scores:
            # FAISS의 경우 거리이므로 유사도로 변환
            similarity = 1.0 / (1.0 + score)
            if similarity >= self.similarity_threshold:
                # 문서에 유사도 점수 추가
                if not hasattr(doc, "metadata"):
                    doc.metadata = {}
                doc.metadata["similarity_score"] = similarity
                filtered_docs.append(doc)
        
        logger.info(f"쿼리 '{query}'에 대해 {len(filtered_docs)}개 문서 검색됨")
        return filtered_docs
    
    @classmethod
    def from_text_files(cls, file_paths: List[str], chunk_size: int = 1000, chunk_overlap: int = 200) -> "VectorRetriever":
        """텍스트 파일로부터 검색기를 생성합니다.
        
        Args:
            file_paths: 텍스트 파일 경로 목록
            chunk_size: 청크 크기
            chunk_overlap: 청크 오버랩 크기
            
        Returns:
            초기화된 VectorRetriever 객체
        """
        retriever = cls()
        documents = []
        
        # 각 파일 처리
        for file_path in file_paths:
            try:
                loader = TextLoader(file_path)
                docs = loader.load()
                
                # 텍스트 분할
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                split_docs = text_splitter.split_documents(docs)
                documents.extend(split_docs)
                
                logger.info(f"파일 '{file_path}'에서 {len(split_docs)}개 문서 로드")
            except Exception as e:
                logger.error(f"파일 '{file_path}' 처리 중 오류: {str(e)}")
        
        # 문서 추가
        retriever.add_documents(documents)
        return retriever 