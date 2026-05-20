import os
import chromadb
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


class VectorDB:
    """
    A simple vector database wrapper using ChromaDB with HuggingFace embeddings.
    """

    def __init__(self, collection_name: str = None, embedding_model: str = None):
        """
        Initialize the vector database.

        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: HuggingFace model name for embeddings
        """
        from chromadb.config import Settings

        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "rag_documents"
        )
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Calculate absolute project root path dynamically
        base_dir = os.path.dirname(os.path.abspath(__file__)) # points to src/
        project_root = os.path.join(base_dir, "..")          # points to root/
        chroma_path = os.path.join(project_root, "chroma_db")
        
        # FIXED: Explicitly disable anonymous telemetry to prevent internal len() crashes
        self.client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False)
        )

        # Load embedding model
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document collection"},
        )

        print(f"Vector database initialized with collection: {self.collection_name}")

    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """
        Splits text cleanly by boundaries using RecursiveCharacterTextSplitter to
        preserve syntactic context.

        Args:
            text: Input text to chunk
            chunk_size: Approximate number of characters per chunk

        Returns:
            List of text chunks
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size * 0.15),  # 15% semantic overlap context
            length_function=lambda x: len(x),     # FIXED: Uses explicit lambda to protect function resolution
            separators=["\n\n", "\n", " ", ""]
        )
        return splitter.split_text(text)

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the vector database.

        Args:
            documents: List of dictionaries where each dict has 'content' and 'metadata' keys
        """
        print(f"Processing {len(documents)} documents...")
        
        all_chunks = []
        all_embeddings = []
        all_metadatas = []
        all_ids = []

        for doc_idx, doc in enumerate(documents):
            # Gracefully handle string-only lists if provided, else parse dictionaries
            content = doc.get("content", "") if isinstance(doc, dict) else str(doc)
            base_metadata = doc.get("metadata", {}) if isinstance(doc, dict) else {}

            if not content.strip():
                continue

            chunks = self.chunk_text(content)
            
            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"doc_{doc_idx}_chunk_{chunk_idx}"
                
                # Create embeddings locally using the sentence transformer model
                embedding = self.embedding_model.encode(chunk).tolist()
                
                # Preserve origin tracks in metadata for reference traceability
                metadata = base_metadata.copy()
                metadata["chunk_index"] = chunk_idx
                metadata["source_doc_index"] = doc_idx

                all_chunks.append(chunk)
                all_embeddings.append(embedding)
                all_metadatas.append(metadata)
                all_ids.append(chunk_id)

        if all_chunks:
            self.collection.add(
                embeddings=all_embeddings,
                documents=all_chunks,
                metadatas=all_metadatas,
                ids=all_ids
            )
            print(f"Successfully added {len(all_chunks)} text chunks to vector database.")
        else:
            print("No valid content found to inject.")

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Search for similar documents in the vector database.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            Dictionary containing search results with keys: 'documents', 'metadatas', 'distances', 'ids'
        """
        if not query.strip():
            return {"documents": [], "metadatas": [], "distances": [], "ids": []}

        # Vectorize incoming query text
        query_embedding = self.embedding_model.encode(query).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        # Simplify structure response format to guarantee safety margins if null
        return {
            "documents": results.get("documents", [[]])[0],
            "metadatas": results.get("metadatas", [[]])[0],
            "distances": results.get("distances", [[]])[0],
            "ids": results.get("ids", [[]])[0],
        }