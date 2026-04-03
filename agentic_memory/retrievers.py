import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional
import ast
import tempfile
import atexit

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from nltk.tokenize import word_tokenize


def simple_tokenize(text):
    return word_tokenize(text)


def _clone_collection(
    src: chromadb.Collection,
    dest: chromadb.Collection,
    batch_size: int = 10
):
    """
    Copies one ChromaDB collection to another. 
    Enables duplicating of collections.
    This seemed to be the only (best) way to do this as the official ChromaDB
        docs also suggest this method:
    """
    existing_count = src.count()
    for i in range(0, existing_count, batch_size):
        batch = src.get(
            include=["metadatas", "documents", "embeddings"],
            limit=batch_size,
            offset=i)
        dest.add(
            ids=batch["ids"],
            documents=batch["documents"],
            metadatas=batch["metadatas"],
            embeddings=batch["embeddings"])


class ChromaRetriever:
    """Vector database retrieval using ChromaDB"""

    def __init__(
        self, 
        collection_name: str = "memories", 
        model_name: str = "all-MiniLM-L6-v2"
    ):
        """Initialize ChromaDB retriever.

        Args:
            collection_name: Name of the ChromaDB collection
        """
        self.client = chromadb.Client(Settings(allow_reset=True))
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.embedding_function
        )

    def add_document(self, document: str, metadata: Dict, doc_id: str):
        """Add a document to ChromaDB.

        Args:
            document: Text content to add
            metadata: Dictionary of metadata
            doc_id: Unique identifier for the document
        """
        # Convert MemoryNote object to serializable format
        processed_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, list):
                processed_metadata[key] = json.dumps(value)
            elif isinstance(value, dict):
                processed_metadata[key] = json.dumps(value)
            else:
                processed_metadata[key] = str(value)

        self.collection.add(
            documents=[document], metadatas=[processed_metadata], ids=[doc_id]
        )

    def delete_document(self, doc_id: str):
        """Delete a document from ChromaDB.

        Args:
            doc_id: ID of document to delete
        """
        self.collection.delete(ids=[doc_id])

    def search(self, query: str, k: int = 5):
        """Search for similar documents.

        Args:
            query: Query text
            k: Number of results to return

        Returns:
            Dict with documents, metadatas, ids, and distances
        """
        results = self.collection.query(query_texts=[query], n_results=k)
        
        if (results is not None) and (results.get("metadatas", [])):
            results["metadatas"] = self._convert_metadata_types(
                results["metadatas"])
        
        return results

    def search_filtered(
        self,
        query: str,
        k: int = 5,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """Search with optional ChromaDB metadata filtering.

        Args:
            query: Query text
            k: Number of results to return
            where: ChromaDB where clause dict (e.g. {"class_name": "Foo"})

        Returns:
            List of dicts with document, metadata, id, and distance
        """
        kwargs = {"query_texts": [query], "n_results": k}
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        if (results is not None) and (results.get("metadatas", [])):
            results["metadatas"] = self._convert_metadata_types(
                results["metadatas"])

        # Flatten into list of dicts for API consumption
        output = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for i in range(len(docs)):
            entry = {
                "id": ids[i] if i < len(ids) else None,
                "document": docs[i] if i < len(docs) else None,
                "distance": dists[i] if i < len(dists) else None,
            }
            if i < len(metas):
                entry.update(metas[i])
            output.append(entry)

        return output

    def _convert_metadata_types(
        self,
        metadatas: List[List[Dict]]
    ) -> List[List[Dict]]:
        """Convert string metadata back to original types.
        
        Args:
            metadatas: List of metadata lists from query results
            
        Returns:
            Converted metadata structure
        """
        for query_metadatas in metadatas:
            if isinstance(query_metadatas, List):
                for metadata_dict in query_metadatas:
                    if isinstance(metadata_dict, Dict):
                        self._convert_metadata_dict(metadata_dict)
        return metadatas

    def _convert_metadata_dict(self, metadata: Dict) -> None:
        """Convert metadata values from strings to appropriate types in-place.
        
        Args:
            metadata: Single metadata dictionary to convert
        """
        for key, value in metadata.items():
            # only attempt to convert strings
            if not isinstance(value, str):
                continue
            else:
                try:
                    metadata[key] = ast.literal_eval(value)
                except Exception:
                    pass


class BM25Retriever:
    """Simple BM25 (Okapi) term-frequency search over in-memory documents.

    Complements vector search with keyword matching — useful for exact term
    lookups (codec names, function signatures, error messages) that embedding
    models may miss.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: List[Dict] = []  # [{id, text, metadata}]
        self.doc_freqs: Counter = Counter()  # term -> number of docs containing it
        self.doc_lens: List[int] = []
        self.avg_dl: float = 0.0
        self.N: int = 0

    def add_document(self, doc_id: str, text: str, metadata: Optional[Dict] = None):
        """Index a document for BM25 search."""
        tokens = simple_tokenize(text.lower())
        self.docs.append({"id": doc_id, "text": text, "metadata": metadata or {}, "tokens": tokens})
        self.doc_lens.append(len(tokens))
        self.N += 1
        self.avg_dl = sum(self.doc_lens) / self.N if self.N else 0.0
        # Update document frequencies
        unique_terms = set(tokens)
        for term in unique_terms:
            self.doc_freqs[term] += 1

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """BM25 search returning top-k results sorted by score."""
        query_tokens = simple_tokenize(query.lower())
        scores = []

        for idx, doc in enumerate(self.docs):
            score = self._score_document(query_tokens, doc["tokens"], self.doc_lens[idx])
            if score > 0:
                scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in scores[:k]:
            doc = self.docs[idx]
            entry = {
                "id": doc["id"],
                "document": doc["text"],
                "bm25_score": round(score, 4),
            }
            entry.update(doc["metadata"])
            results.append(entry)
        return results

    def _score_document(self, query_tokens: List[str], doc_tokens: List[str], dl: int) -> float:
        """Compute BM25 score for a single document."""
        tf_doc = Counter(doc_tokens)
        score = 0.0
        for term in query_tokens:
            if term not in tf_doc:
                continue
            tf = tf_doc[term]
            df = self.doc_freqs.get(term, 0)
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl) if self.avg_dl > 0 else tf + self.k1
            score += idf * (numerator / denominator)
        return score

    @property
    def count(self) -> int:
        return self.N


class PersistentChromaRetriever(ChromaRetriever):
    """
    Persistent ChromaDB client/retriever to facilitate sharing of memory from
        multiple agents across sessions.
    Simply changes how the client and collection are initialized. Other
        functionality is inherited from ChromaRetriever.
    """

    def __init__(
        self, 
        directory: Optional[str] = None, 
        collection_name: str = "memories", 
        model_name: str = "all-MiniLM-L6-v2",
        extend: bool = False
    ):
        """
        Initialize persistent ChromaDB retriever.
        
        :param directory: Directory path for ChromaDB storage. Defaults to
            '~/.chromadb' if None.
        :collection_name: Name of the ChromaDB collection.
        :model_name: SentenceTransformer model name for embeddings.
        :extend: If True, allows initializes client and retriever from
            collection if it exists. Raises error if False and collection
            already exists. This prevents accidental overwriting of
            existing collections.
        """
        if directory is None:
            directory = Path.home() / '.chromadb'
            directory.mkdir(parents=True, exist_ok=True)
        elif isinstance(directory, str):
            directory = Path(directory)

        try:
            directory.resolve(strict=True)
        except FileNotFoundError:
            directory.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValueError(f'Error accessing directory: {e}')        

        # Use PersistentClient instead of regular Client
        self.client = chromadb.PersistentClient(path=str(directory))
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=model_name)
        
        existing_collections = [col.name for col in self.client.list_collections()]
        
        if collection_name in existing_collections:
            if extend:
                self.collection = self.client.get_collection(name=collection_name)
            else:
                raise ValueError(
                    f"Collection '{collection_name}' already exists. "
                    "Use extend=True to add to it."
                )
        else:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
        self.collection_name = collection_name


class CopiedChromaRetriever(PersistentChromaRetriever):
    """
    ChromaDB retriever that creates a copy of an existing collection
        under to a temporary ChromaDB instance.
    Useful for creating isolated copies of shared starting memory collections.
    """

    def __init__(
        self,
        directory: Optional[str] = None, 
        collection_name: str = "memories", 
        model_name: str = "all-MiniLM-L6-v2",
        _dest_collection_name: Optional[str] = None,
        _copy_batch_size: int = 10,
    ):
        """
        Initialize the CopiedChromaDB retriever.

        :param directory: Directory path for source ChromaDB storage. If None,
            defaults to '~/.chromadb'.
        :param collection_name: Name of the source ChromaDB collection to copy.
        :param model_name: SentenceTransformer model name for embeddings.
        :param _dest_collection_name: Optional name for the destination
            collection. If None, defaults to '{collection_name}__clone'.
            This parameter is marked as private as the class itself is meant
            for single use and discard db that exists in a temporary so naming
            the copied collection is most likely not needed. 
        :param _copy_batch_size: Number of documents to copy per batch.
            Shouldn't need to be changed normally. 
        """

        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=model_name)

        # ensure source is valid
        if directory is None:
            directory = Path.home() / '.chromadb'
            directory.mkdir(parents=True, exist_ok=True)
        elif isinstance(directory, str):
            directory = Path(directory)
        self._src_client = chromadb.PersistentClient(path=str(directory))

        self._src = self._src_client.get_collection(name=collection_name)
        existing_collections = [
            col.name for col in self._src_client.list_collections()]
        if collection_name not in existing_collections:
            raise ValueError(
                f"Collection '{collection_name}' to be copied does not exist."
            )        

        # use temp directory for destination collection
        try:
            self._tmpdir = tempfile.TemporaryDirectory(
                prefix='chromadb_ephemeral_')
            self._tmp_path = Path(self._tmpdir.name)
            self._dst_client = chromadb.PersistentClient(
                path=str(self._tmp_path)
            )
            self.collection_name = (
                _dest_collection_name 
                or f"{collection_name}__clone"
            )
            self.collection = self._dst_client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata=self._src.metadata
            )
        except Exception as e:
            raise ValueError(f"Error creating temporary ChromaDB: {e}")
        
        try:
            _clone_collection(
                src=self._src,
                dest=self.collection,
                batch_size=_copy_batch_size,
            )
        except Exception as e:
            raise ValueError(f"Error cloning ChromaDB collection: {e}")
        
        atexit.register(self.close)

    def close(self):
        """Cleanup temporary directory."""
        try:
            self._dst_client.delete_collection(self.collection_name)
        except Exception:
            pass
        try:
            self._tmpdir.cleanup()
        except Exception:
            pass

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
