import keyword
from typing import List, Dict, Optional, Any, Tuple
import uuid
from datetime import datetime
from .llm_controller import LLMController
from .retrievers import ChromaRetriever, PersistentChromaRetriever, BM25Retriever
import json
import logging
import re as _re
import math as _math
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from abc import ABC, abstractmethod
from transformers import AutoModel, AutoTokenizer
from nltk.tokenize import word_tokenize
import pickle
from pathlib import Path
import time

logger = logging.getLogger(__name__)

class MemoryNote:
    """A memory note that represents a single unit of information in the memory system.
    
    This class encapsulates all metadata associated with a memory, including:
    - Core content and identifiers
    - Temporal information (creation and access times)
    - Semantic metadata (keywords, context, tags)
    - Relationship data (links to other memories)
    - Usage statistics (retrieval count)
    - Evolution tracking (history of changes)
    """
    
    def __init__(self, 
                 content: str,
                 id: Optional[str] = None,
                 keywords: Optional[List[str]] = None,
                 links: Optional[Dict] = None,
                 retrieval_count: Optional[int] = None,
                 timestamp: Optional[str] = None,
                 last_accessed: Optional[str] = None,
                 context: Optional[str] = None,
                 evolution_history: Optional[List] = None,
                 category: Optional[str] = None,
                 tags: Optional[List[str]] = None):
        """Initialize a new memory note with its associated metadata.
        
        Args:
            content (str): The main text content of the memory
            id (Optional[str]): Unique identifier for the memory. If None, a UUID will be generated
            keywords (Optional[List[str]]): Key terms extracted from the content
            links (Optional[Dict]): References to related memories
            retrieval_count (Optional[int]): Number of times this memory has been accessed
            timestamp (Optional[str]): Creation time in format YYYYMMDDHHMM
            last_accessed (Optional[str]): Last access time in format YYYYMMDDHHMM
            context (Optional[str]): The broader context or domain of the memory
            evolution_history (Optional[List]): Record of how the memory has evolved
            category (Optional[str]): Classification category
            tags (Optional[List[str]]): Additional classification tags
        """
        # Core content and ID
        self.content = content
        self.id = id or str(uuid.uuid4())
        
        # Semantic metadata
        self.keywords = keywords or []
        self.links = links or []
        self.context = context or "General"
        self.category = category or "Uncategorized"
        self.tags = tags or []
        
        # Temporal information
        current_time = datetime.now().strftime("%Y%m%d%H%M")
        self.timestamp = timestamp or current_time
        self.last_accessed = last_accessed or current_time
        
        # Usage and evolution data
        self.retrieval_count = retrieval_count or 0
        self.evolution_history = evolution_history or []

def _infer_category(content: str) -> str:
    c = content.lower()
    if any(w in c for w in ("lesson", "bug", "error", "failed", "fixed", "incident")):
        return "lesson"
    if any(w in c for w in ("checkpoint", "session", "branch", "commit")):
        return "checkpoint"
    if any(w in c for w in ("plan", "phase", "task", "roadmap")):
        return "plan"
    if any(w in c for w in ("always", "never", "prefer", "avoid", "rule", "standard")):
        return "behavioral-rule"
    if any(w in c for w in ("port", "url", "path", "config", "key-facts", "api key")):
        return "key-facts"
    if any(w in c for w in ("next session", "resume", "continuation")):
        return "session-handoff"
    return "general"


def _extract_keywords(content: str) -> list:
    tokens = _re.findall(
        r'\b[A-Z][A-Za-z]{2,}\b'
        r'|[a-z]+-[a-z]+'
        r'|\b[A-Z_]{3,}\b'
        r'|\bPR\s*#\d+\b'
        r'|\bv\d+\.\d+\b',
        content,
    )
    seen = set()
    result = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            result.append(t)
        if len(result) >= 10:
            break
    return result


def _recency_score(timestamp_str: str, half_life_hours: float = 48.0) -> float:
    """Exponential decay: 1.0 = now, 0.5 = one half-life ago. Returns 0.1 min."""
    try:
        from datetime import datetime as _dt
        ts = _dt.strptime(timestamp_str, "%Y%m%d%H%M")
        hours_old = (_dt.now() - ts).total_seconds() / 3600
        return max(0.1, _math.exp(-0.693 * hours_old / half_life_hours))
    except Exception:
        return 0.5


class AgenticMemorySystem:
    """Core memory system that manages memory notes and their evolution.
    
    This system provides:
    - Memory creation, retrieval, update, and deletion
    - Content analysis and metadata extraction
    - Memory evolution and relationship management
    - Hybrid search capabilities
    """
    
    def __init__(self,
                 model_name: str = 'all-MiniLM-L6-v2',
                 llm_backend: str = "openai",
                 llm_model: str = "gpt-4o-mini",
                 evo_threshold: int = 100,
                 api_key: Optional[str] = None,
                 persistent: bool = True,
                 persist_directory: Optional[str] = None,
                 namespace: str = "default",
                 llm_base_url: Optional[str] = None):
        """Initialize the memory system.

        Args:
            model_name: Name of the sentence transformer model
            llm_backend: LLM backend to use (openai/ollama/litellm)
            llm_model: Name of the LLM model
            evo_threshold: Number of memories before triggering evolution
            api_key: API key for the LLM service
            persistent: If True, use persistent storage (default: True)
            persist_directory: Directory for persistent storage (default: ~/.agentic_memory)
            namespace: Namespace for multi-session isolation (used as collection name suffix)
            llm_base_url: Base URL for LiteLLM gateway (when using litellm backend)
        """
        self.memories = {}
        self.model_name = model_name
        self.namespace = namespace
        self.persistent = persistent

        collection_name = f"memories_{namespace}" if namespace != "default" else "memories"

        if persistent:
            # Use persistent storage
            if persist_directory is None:
                persist_directory = os.path.expanduser("~/.agentic_memory")
            self.persist_directory = persist_directory

            try:
                self.retriever = PersistentChromaRetriever(
                    directory=persist_directory,
                    collection_name=collection_name,
                    model_name=self.model_name,
                    extend=True  # Allow extending existing collections
                )
                logger.info(f"Initialized persistent memory at {persist_directory}")
            except Exception as e:
                logger.warning(f"Could not initialize persistent storage: {e}, falling back to ephemeral")
                self.retriever = ChromaRetriever(collection_name=collection_name, model_name=self.model_name)
                self.persistent = False
        else:
            # Use ephemeral storage
            try:
                # First try to reset the collection if it exists
                temp_retriever = ChromaRetriever(collection_name=collection_name, model_name=self.model_name)
                temp_retriever.client.reset()
            except Exception as e:
                logger.warning(f"Could not reset ChromaDB collection: {e}")

            # Create a fresh retriever instance
            self.retriever = ChromaRetriever(collection_name=collection_name, model_name=self.model_name)

        # BM25 index — must be initialized before _restore_memories_from_chroma()
        self._bm25: BM25Retriever = BM25Retriever()

        # Restore in-memory metadata from ChromaDB on startup
        if self.persistent:
            self._restore_memories_from_chroma()

        # Initialize LLM controller
        self.llm_controller = LLMController(llm_backend, llm_model, api_key, llm_base_url)
        self.evo_cnt = 0
        self.evo_threshold = evo_threshold

        # Evolution system prompt
        self._evolution_system_prompt = '''
                                You are an AI memory evolution agent responsible for managing and evolving a knowledge base.
                                Analyze the the new memory note according to keywords and context, also with their several nearest neighbors memory.
                                Make decisions about its evolution.  

                                The new memory context:
                                {context}
                                content: {content}
                                keywords: {keywords}

                                The nearest neighbors memories:
                                {nearest_neighbors_memories}

                                Based on this information, determine:
                                1. Should this memory be evolved? Consider its relationships with other memories.
                                2. What specific actions should be taken (strengthen, update_neighbor)?
                                   2.1 If choose to strengthen the connection, which memory should it be connected to? Can you give the updated tags of this memory?
                                   2.2 If choose to update_neighbor, you can update the context and tags of these memories based on the understanding of these memories. If the context and the tags are not updated, the new context and tags should be the same as the original ones. Generate the new context and tags in the sequential order of the input neighbors.
                                Tags should be determined by the content of these characteristic of these memories, which can be used to retrieve them later and categorize them.
                                Note that the length of new_tags_neighborhood must equal the number of input neighbors, and the length of new_context_neighborhood must equal the number of input neighbors.
                                The number of neighbors is {neighbor_number}.
                                Return your decision in JSON format with the following structure:
                                {{
                                    "should_evolve": True or False,
                                    "actions": ["strengthen", "update_neighbor"],
                                    "suggested_connections": ["neighbor_memory_ids"],
                                    "tags_to_update": ["tag_1",..."tag_n"], 
                                    "new_context_neighborhood": ["new context",...,"new context"],
                                    "new_tags_neighborhood": [["tag_1",...,"tag_n"],...["tag_1",...,"tag_n"]],
                                }}
                                '''
        
    def _restore_memories_from_chroma(self):
        """Restore in-memory metadata dict from persistent ChromaDB on startup."""
        try:
            collection = self.retriever.collection
            count = collection.count()
            if count == 0:
                return

            logger.info(f"Restoring {count} memories from persistent storage...")
            batch_size = 1000
            restored = 0
            for offset in range(0, count, batch_size):
                batch = collection.get(
                    limit=batch_size,
                    offset=offset,
                    include=["metadatas"]
                )
                for id_, meta in zip(batch["ids"], batch["metadatas"]):
                    try:
                        # Parse JSON-encoded list/dict fields
                        keywords = json.loads(meta.get("keywords", "[]")) if isinstance(meta.get("keywords"), str) else meta.get("keywords", [])
                        links = json.loads(meta.get("links", "[]")) if isinstance(meta.get("links"), str) else meta.get("links", [])
                        tags = json.loads(meta.get("tags", "[]")) if isinstance(meta.get("tags"), str) else meta.get("tags", [])
                        evolution_history = json.loads(meta.get("evolution_history", "[]")) if isinstance(meta.get("evolution_history"), str) else meta.get("evolution_history", [])
                        retrieval_count = int(meta.get("retrieval_count", 0))

                        note = MemoryNote(
                            content=meta.get("content", ""),
                            id=id_,
                            keywords=keywords,
                            links=links,
                            retrieval_count=retrieval_count,
                            timestamp=meta.get("timestamp"),
                            last_accessed=meta.get("last_accessed"),
                            context=meta.get("context"),
                            evolution_history=evolution_history,
                            category=meta.get("category"),
                            tags=tags,
                        )
                        self.memories[id_] = note
                        restored += 1
                    except Exception as e:
                        logger.warning(f"Failed to restore memory {id_}: {e}")

            logger.info(f"Restored {restored}/{count} memories from persistent storage")

            # Rebuild BM25 index from loaded memories
            self._bm25 = BM25Retriever()
            for mem_id, mem in self.memories.items():
                self._bm25.add_document(mem_id, mem.content, {})
            logger.info(f"BM25 index rebuilt: {self._bm25.count} documents")
        except Exception as e:
            logger.warning(f"Could not restore memories from ChromaDB: {e}")

    def analyze_content(self, content: str) -> Dict:            
        """Analyze content using LLM to extract semantic metadata.
        
        Uses a language model to understand the content and extract:
        - Keywords: Important terms and concepts
        - Context: Overall domain or theme
        - Tags: Classification categories
        
        Args:
            content (str): The text content to analyze
            
        Returns:
            Dict: Contains extracted metadata with keys:
                - keywords: List[str]
                - context: str
                - tags: List[str]
        """
        prompt = """Generate a structured analysis of the following content by:
            1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
            2. Extracting core themes and contextual elements
            3. Creating relevant categorical tags

            Format the response as a JSON object:
            {
                "keywords": [
                    // several specific, distinct keywords that capture key concepts and terminology
                    // Order from most to least important
                    // Don't include keywords that are the name of the speaker or time
                    // At least three keywords, but don't be too redundant.
                ],
                "context": 
                    // one sentence summarizing:
                    // - Main topic/domain
                    // - Key arguments/points
                    // - Intended audience/purpose
                ,
                "tags": [
                    // several broad categories/themes for classification
                    // Include domain, format, and type tags
                    // At least three tags, but don't be too redundant.
                ]
            }

            Content for analysis:
            """ + content
        try:
            response = self.llm_controller.llm.get_completion(prompt, response_format={"type": "json_schema", "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "keywords": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "context": {
                                    "type": "string",
                                },
                                "tags": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                }
                            }
                        }
                    }})
            return json.loads(response)
        except Exception as e:
            print(f"Error analyzing content: {e}")
            return {"keywords": [], "context": "General", "tags": []}

    def add_note(self, content: str, time: str = None, **kwargs) -> str:
        """Add a new memory note"""
        # Create MemoryNote without llm_controller
        if time is not None:
            kwargs['timestamp'] = time
        note = MemoryNote(content=content, **kwargs)
        
        # Update retriever with all documents
        evo_label, note = self.process_memory(note)
        self.memories[note.id] = note
        
        # Add to ChromaDB with complete metadata
        metadata = {
            "id": note.id,
            "content": note.content,
            "keywords": note.keywords,
            "links": note.links,
            "retrieval_count": note.retrieval_count,
            "timestamp": note.timestamp,
            "last_accessed": note.last_accessed,
            "context": note.context,
            "evolution_history": note.evolution_history,
            "category": note.category,
            "tags": note.tags
        }
        self.retriever.add_document(note.content, metadata, note.id)
        self._bm25.add_document(note.id, note.content, {})

        if evo_label == True:
            self.evo_cnt += 1
            if self.evo_cnt % self.evo_threshold == 0:
                self.consolidate_memories()
        return note.id
    
    def consolidate_memories(self):
        """Rebuild retriever from self.memories.

        Uses PersistentChromaRetriever when persistent=True so data survives
        service restarts. Respects self.namespace for collection naming.
        Also rebuilds the BM25 index.
        """
        collection_name = f"memories_{self.namespace}" if self.namespace != "default" else "memories"

        if self.persistent:
            self.retriever = PersistentChromaRetriever(
                directory=self.persist_directory,
                collection_name=collection_name,
                model_name=self.model_name,
                extend=False,
            )
        else:
            self.retriever = ChromaRetriever(
                collection_name=collection_name,
                model_name=self.model_name,
            )

        # Rebuild BM25 index
        self._bm25 = BM25Retriever()

        for memory in self.memories.values():
            metadata = {
                "id": memory.id,
                "content": memory.content,
                "keywords": memory.keywords,
                "links": memory.links,
                "retrieval_count": memory.retrieval_count,
                "timestamp": memory.timestamp,
                "last_accessed": memory.last_accessed,
                "context": memory.context,
                "evolution_history": memory.evolution_history,
                "category": memory.category,
                "tags": memory.tags,
            }
            self.retriever.add_document(memory.content, metadata, memory.id)
            self._bm25.add_document(memory.id, memory.content, {})
    
    def _sync_memory_to_chroma(self, memory_id: str) -> None:
        """Write a single memory from self.memories back to ChromaDB.

        Called after process_memory modifies a neighbor's metadata in-place.
        """
        mem = self.memories.get(memory_id)
        if not mem:
            return
        metadata = {
            "id": mem.id,
            "content": mem.content,
            "keywords": mem.keywords,
            "links": mem.links,
            "retrieval_count": mem.retrieval_count,
            "timestamp": mem.timestamp,
            "last_accessed": mem.last_accessed,
            "context": mem.context,
            "evolution_history": mem.evolution_history,
            "category": mem.category,
            "tags": mem.tags,
        }
        try:
            self.retriever.delete_document(memory_id)
        except Exception:
            pass
        self.retriever.add_document(mem.content, metadata, memory_id)

    def find_related_memories(self, query: str, k: int = 5) -> Tuple[str, List[int]]:
        """Find related memories using ChromaDB retrieval"""
        if not self.memories:
            return "", []
            
        try:
            # Get results from ChromaDB
            results = self.retriever.search(query, k)
            
            # Convert to list of memories
            memory_str = ""
            indices = []
            
            if 'ids' in results and results['ids'] and len(results['ids']) > 0 and len(results['ids'][0]) > 0:
                for i, doc_id in enumerate(results['ids'][0]):
                    # Get metadata from ChromaDB results
                    if i < len(results['metadatas'][0]):
                        metadata = results['metadatas'][0][i]
                        # Format memory string
                        memory_str += f"memory index:{i}\ttalk start time:{metadata.get('timestamp', '')}\tmemory content: {metadata.get('content', '')}\tmemory context: {metadata.get('context', '')}\tmemory keywords: {str(metadata.get('keywords', []))}\tmemory tags: {str(metadata.get('tags', []))}\n"
                        indices.append(i)
                    
            return memory_str, indices
        except Exception as e:
            logger.error(f"Error in find_related_memories: {str(e)}")
            return "", []

    def find_related_memories_raw(self, query: str, k: int = 5) -> str:
        """Find related memories using ChromaDB retrieval in raw format"""
        if not self.memories:
            return ""
            
        # Get results from ChromaDB
        results = self.retriever.search(query, k)
        
        # Convert to list of memories
        memory_str = ""
        
        if 'ids' in results and results['ids'] and len(results['ids']) > 0:
            for i, doc_id in enumerate(results['ids'][0][:k]):
                if i < len(results['metadatas'][0]):
                    # Get metadata from ChromaDB results
                    metadata = results['metadatas'][0][i]
                    
                    # Add main memory info
                    memory_str += f"talk start time:{metadata.get('timestamp', '')}\tmemory content: {metadata.get('content', '')}\tmemory context: {metadata.get('context', '')}\tmemory keywords: {str(metadata.get('keywords', []))}\tmemory tags: {str(metadata.get('tags', []))}\n"
                    
                    # Add linked memories if available
                    links = metadata.get('links', [])
                    j = 0
                    for link_id in links:
                        if link_id in self.memories and j < k:
                            neighbor = self.memories[link_id]
                            memory_str += f"talk start time:{neighbor.timestamp}\tmemory content: {neighbor.content}\tmemory context: {neighbor.context}\tmemory keywords: {str(neighbor.keywords)}\tmemory tags: {str(neighbor.tags)}\n"
                            j += 1
                            
        return memory_str

    def read(self, memory_id: str) -> Optional[MemoryNote]:
        """Retrieve a memory note by its ID.
        
        Args:
            memory_id (str): ID of the memory to retrieve
            
        Returns:
            MemoryNote if found, None otherwise
        """
        return self.memories.get(memory_id)
    
    def update(self, memory_id: str, **kwargs) -> bool:
        """Update a memory note.
        
        Args:
            memory_id: ID of memory to update
            **kwargs: Fields to update
            
        Returns:
            bool: True if update successful
        """
        if memory_id not in self.memories:
            return False
            
        note = self.memories[memory_id]
        
        # Update fields
        for key, value in kwargs.items():
            if hasattr(note, key):
                setattr(note, key, value)
                
        # Update in ChromaDB
        metadata = {
            "id": note.id,
            "content": note.content,
            "keywords": note.keywords,
            "links": note.links,
            "retrieval_count": note.retrieval_count,
            "timestamp": note.timestamp,
            "last_accessed": note.last_accessed,
            "context": note.context,
            "evolution_history": note.evolution_history,
            "category": note.category,
            "tags": note.tags
        }
        
        # Delete and re-add to update
        self.retriever.delete_document(memory_id)
        self.retriever.add_document(document=note.content, metadata=metadata, doc_id=memory_id)
        
        return True
    
    def delete(self, memory_id: str) -> bool:
        """Delete a memory note by its ID.
        
        Args:
            memory_id (str): ID of the memory to delete
            
        Returns:
            bool: True if memory was deleted, False if not found
        """
        if memory_id in self.memories:
            # Delete from ChromaDB
            self.retriever.delete_document(memory_id)
            # Delete from local storage
            del self.memories[memory_id]
            return True
        return False
    
    def _search_raw(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Internal search method that returns raw results from ChromaDB.
        
        This is used internally by the memory evolution system to find
        related memories for potential evolution.
        
        Args:
            query (str): The search query text
            k (int): Maximum number of results to return
            
        Returns:
            List[Dict[str, Any]]: Raw search results from ChromaDB
        """
        results = self.retriever.search(query, k)
        return [{'id': doc_id, 'score': score} 
                for doc_id, score in zip(results['ids'][0], results['distances'][0])]
                
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for memories using a hybrid retrieval approach."""
        # Get results from ChromaDB (only do this once)
        search_results = self.retriever.search(query, k)
        memories = []
        
        # Process ChromaDB results
        for i, doc_id in enumerate(search_results['ids'][0]):
            memory = self.memories.get(doc_id)
            if memory:
                memories.append({
                    'id': doc_id,
                    'content': memory.content,
                    'context': memory.context,
                    'keywords': memory.keywords,
                    'score': search_results['distances'][0][i]
                })
        
        return memories[:k]
    
    def _search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for memories using a hybrid retrieval approach.
        
        This method combines results from both:
        1. ChromaDB vector store (semantic similarity)
        2. Embedding-based retrieval (dense vectors)
        
        The results are deduplicated and ranked by relevance.
        
        Args:
            query (str): The search query text
            k (int): Maximum number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of search results, each containing:
                - id: Memory ID
                - content: Memory content
                - score: Similarity score
                - metadata: Additional memory metadata
        """
        # Get results from ChromaDB
        chroma_results = self.retriever.search(query, k)
        memories = []
        
        # Process ChromaDB results
        for i, doc_id in enumerate(chroma_results['ids'][0]):
            memory = self.memories.get(doc_id)
            if memory:
                memories.append({
                    'id': doc_id,
                    'content': memory.content,
                    'context': memory.context,
                    'keywords': memory.keywords,
                    'score': chroma_results['distances'][0][i]
                })
                
        # Get results from embedding retriever
        embedding_results = self.retriever.search(query, k)
        
        # Combine results with deduplication
        seen_ids = set(m['id'] for m in memories)
        for result in embedding_results:
            memory_id = result.get('id')
            if memory_id and memory_id not in seen_ids:
                memory = self.memories.get(memory_id)
                if memory:
                    memories.append({
                        'id': memory_id,
                        'content': memory.content,
                        'context': memory.context,
                        'keywords': memory.keywords,
                        'score': result.get('score', 0.0)
                    })
                    seen_ids.add(memory_id)
                    
        return memories[:k]

    def search_agentic(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search using hybrid BM25+vector scoring with recency decay."""
        if not self.memories:
            return []

        try:
            fetch_k = max(k * 3, 20)

            # Vector scores (ChromaDB cosine distance → similarity)
            chroma = self.retriever.search(query, fetch_k)
            if ('ids' not in chroma or not chroma['ids'] or
                    len(chroma['ids']) == 0 or len(chroma['ids'][0]) == 0):
                return []

            vector_scores: Dict[str, float] = {}
            for i, doc_id in enumerate(chroma['ids'][0]):
                dist = chroma['distances'][0][i] if 'distances' in chroma else 1.0
                vector_scores[doc_id] = max(0.0, 1.0 - dist)

            # BM25 scores (normalized 0–1)
            bm25_raw = self._bm25.search(query, fetch_k)
            bm25_scores: Dict[str, float] = {}
            if bm25_raw:
                max_bm25 = max(r['bm25_score'] for r in bm25_raw)
                if max_bm25 > 0:
                    for r in bm25_raw:
                        bm25_scores[r['id']] = r['bm25_score'] / max_bm25

            # Hybrid + recency
            all_ids = set(vector_scores) | set(bm25_scores)
            scored: List[tuple] = []
            for doc_id in all_ids:
                v_score = vector_scores.get(doc_id, 0.0)
                b_score = bm25_scores.get(doc_id, 0.0)
                base = v_score * 0.6 + b_score * 0.4

                mem = self.memories.get(doc_id)
                recency = _recency_score(mem.timestamp if mem else "") if mem else 0.5
                final = base * 0.80 + recency * 0.20

                scored.append((doc_id, final))

            scored.sort(key=lambda x: x[1], reverse=True)

            # Build result dicts
            memories: List[Dict[str, Any]] = []
            seen_ids: set = set()
            for doc_id, score in scored[:k]:
                if doc_id in seen_ids:
                    continue
                mem = self.memories.get(doc_id)
                if not mem:
                    continue
                memories.append({
                    'id': doc_id,
                    'content': mem.content,
                    'context': mem.context,
                    'keywords': mem.keywords,
                    'tags': mem.tags,
                    'timestamp': mem.timestamp,
                    'category': mem.category,
                    'score': round(1.0 - score, 4),
                    'is_neighbor': False,
                })
                seen_ids.add(doc_id)

            # Add linked neighbors (up to k total)
            neighbor_count = 0
            for memory in list(memories):
                if neighbor_count >= k:
                    break
                mem = self.memories.get(memory['id'])
                links = mem.links if mem else []
                for link_id in links:
                    if link_id not in seen_ids and neighbor_count < k:
                        neighbor = self.memories.get(link_id)
                        if neighbor:
                            memories.append({
                                'id': link_id,
                                'content': neighbor.content,
                                'context': neighbor.context,
                                'keywords': neighbor.keywords,
                                'tags': neighbor.tags,
                                'timestamp': neighbor.timestamp,
                                'category': neighbor.category,
                                'is_neighbor': True,
                            })
                            seen_ids.add(link_id)
                            neighbor_count += 1

            return memories[:k]

        except Exception as e:
            logger.error(f"Error in search_agentic: {str(e)}")
            return []

    def process_memory(self, note: MemoryNote) -> Tuple[bool, MemoryNote]:
        """Process a memory note and determine if it should evolve.
        
        Args:
            note: The memory note to process
            
        Returns:
            Tuple[bool, MemoryNote]: (should_evolve, processed_note)
        """
        # Apply rule-based fallback regardless of whether LLM evolution runs
        if note.category in (None, "", "Uncategorized"):
            note.category = _infer_category(note.content)
        if not note.keywords:
            note.keywords = _extract_keywords(note.content)

        # For first memory, skip neighbor-based LLM evolution
        if not self.memories:
            return False, note
            
        try:
            # Get nearest neighbors
            neighbors_text, indices = self.find_related_memories(note.content, k=5)
            if not neighbors_text or not indices:
                return False, note
                
            # Format neighbors for LLM - in this case, neighbors_text is already formatted
            
            # Query LLM for evolution decision
            response = ""
            prompt = self._evolution_system_prompt.format(
                content=note.content,
                context=note.context,
                keywords=note.keywords,
                nearest_neighbors_memories=neighbors_text,
                neighbor_number=len(indices)
            )
            
            try:
                response = self.llm_controller.llm.get_completion(
                    prompt,
                    response_format={"type": "json_schema", "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "should_evolve": {
                                    "type": "boolean"
                                },
                                "actions": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "suggested_connections": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "new_context_neighborhood": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "tags_to_update": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "new_tags_neighborhood": {
                                    "type": "array",
                                    "items": {
                                        "type": "array",
                                        "items": {
                                            "type": "string"
                                        }
                                    }
                                }
                            },
                            "required": ["should_evolve", "actions", "suggested_connections", 
                                      "tags_to_update", "new_context_neighborhood", "new_tags_neighborhood"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }}
                )
                
                # Some models wrap JSON in markdown code fences despite the
                # json_schema response_format — strip them before parsing.
                cleaned = (response or "").strip()
                if cleaned.startswith("```"):
                    first_nl = cleaned.find("\n")
                    closing = cleaned.rfind("```")
                    if first_nl != -1 and closing > first_nl:
                        cleaned = cleaned[first_nl + 1 : closing].strip()
                response_json = json.loads(cleaned)
                should_evolve = response_json["should_evolve"]
                
                if should_evolve:
                    actions = response_json["actions"]
                    for action in actions:
                        if action == "strengthen":
                            suggest_connections = response_json["suggested_connections"]
                            new_tags = response_json["tags_to_update"]
                            note.links.extend(suggest_connections)
                            note.tags = new_tags
                        elif action == "update_neighbor":
                            new_context_neighborhood = response_json["new_context_neighborhood"]
                            new_tags_neighborhood = response_json["new_tags_neighborhood"]
                            noteslist = list(self.memories.values())
                            notes_id = list(self.memories.keys())
                            
                            for i in range(min(len(indices), len(new_tags_neighborhood))):
                                # Skip if we don't have enough neighbors
                                if i >= len(indices):
                                    continue
                                    
                                tag = new_tags_neighborhood[i]
                                if i < len(new_context_neighborhood):
                                    context = new_context_neighborhood[i]
                                else:
                                    # Since indices are just numbers now, we need to find the memory
                                    # In memory list using its index number
                                    if i < len(noteslist):
                                        context = noteslist[i].context
                                    else:
                                        continue
                                        
                                # Get index from the indices list
                                if i < len(indices):
                                    memorytmp_idx = indices[i]
                                    # Make sure the index is valid
                                    if memorytmp_idx < len(noteslist):
                                        notetmp = noteslist[memorytmp_idx]
                                        notetmp.tags = tag
                                        notetmp.context = context
                                        # Make sure the index is valid
                                        if memorytmp_idx < len(notes_id):
                                            self.memories[notes_id[memorytmp_idx]] = notetmp
                                            self._sync_memory_to_chroma(notes_id[memorytmp_idx])

                return should_evolve, note

            except (json.JSONDecodeError, KeyError, Exception) as e:
                logger.warning(
                    f"Memory evolution LLM call failed ({type(e).__name__}): {e}. "
                    f"Raw response head: {(response or '')[:200]!r}. "
                    "Using rule-based fallback."
                )
                if note.category in (None, "", "Uncategorized"):
                    note.category = _infer_category(note.content)
                if not note.keywords:
                    note.keywords = _extract_keywords(note.content)
                return False, note

        except Exception as e:
            logger.warning(
                f"Memory evolution LLM call failed ({type(e).__name__}): {e}. "
                "Using rule-based fallback."
            )
            if note.category in (None, "", "Uncategorized"):
                note.category = _infer_category(note.content)
            if not note.keywords:
                note.keywords = _extract_keywords(note.content)
            return False, note
