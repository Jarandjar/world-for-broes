"""
SWARM VECTOR DB - Semantic search across all memory

Qdrant integration for institutional memory retrieval
"""

import os
from typing import List, Dict, Optional
from dataclasses import dataclass

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, will use system env vars

try:
    from gpu_embeddings import get_gpu_embedding_engine
except Exception:
    get_gpu_embedding_engine = None  # type: ignore


@dataclass
class VectorDBConfig:
    """Qdrant configuration"""
    host: str = "localhost"
    port: int = 6333
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    distance: str = "Cosine"


class SwarmVectorDB:
    """Vector database for semantic search across swarm memory"""
    
    def __init__(self, config: VectorDBConfig = None):
        self.config = config or VectorDBConfig()
        self.client = None
        self._gpu_embeddings = None
        self._maybe_init_gpu_embeddings()
        self._initialize_client()

    def _maybe_init_gpu_embeddings(self):
        """Initialize GPU embedding engine if enabled via env."""
        if not get_gpu_embedding_engine:
            return
        use_gpu = os.getenv("SWARM_USE_GPU_EMBEDDINGS", "0").lower() in {"1", "true", "yes"}
        if not use_gpu:
            return
        engine = get_gpu_embedding_engine()
        if not engine:
            print("⚠️  GPU embeddings requested but engine unavailable (missing deps or GPU).")
            return
        self._gpu_embeddings = engine
        dim = engine.embedding_dim
        if dim and dim != self.config.embedding_dim:
            print(f"⚠️  Overriding embedding_dim to {dim} to match GPU model.")
            self.config.embedding_dim = dim
    
    def _initialize_client(self):
        """Initialize Qdrant client"""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            
            self.client = QdrantClient(
                host=self.config.host,
                port=self.config.port
            )
            self.Distance = Distance
            self.VectorParams = VectorParams
            print(f"✅ Connected to Qdrant at {self.config.host}:{self.config.port}")
        except ImportError:
            print("⚠️  qdrant-client not installed. Install: pip install qdrant-client")
            self.client = None
        except Exception as e:
            print(f"⚠️  Could not connect to Qdrant: {e}")
            self.client = None
    
    def create_collections(self):
        """Create 6 collections for different content types"""
        if not self.client:
            print("❌ Qdrant client not available")
            return False
        
        collections = {
            "docs": "PDF/text documents and longform content",
            "events": "Log entries, agent actions, system events",
            "lore": "Storylines, timelines, codex chapters",
            "reports": "Analytics summaries, nightly reports",
            "chat_messages": "Individual chat messages for fine-grained Q&A",
            "chat_summaries": "High-level chat summaries for topic search"
        }
        
        print("\n📦 Creating Qdrant Collections")
        print("=" * 60)
        
        for collection_name, description in collections.items():
            try:
                # Check if collection exists
                existing = self.client.get_collections().collections
                if any(c.name == collection_name for c in existing):
                    print(f"✓ {collection_name:<20} (already exists)")
                    continue
                
                # Create collection
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=self.VectorParams(
                        size=self.config.embedding_dim,
                        distance=getattr(self.Distance, self.config.distance.upper())
                    )
                )
                print(f"✅ {collection_name:<20} {description}")
            except Exception as e:
                print(f"❌ {collection_name:<20} Error: {e}")
        
        print()
        return True

    def ensure_collection(self, collection_name: str, *, distance: str | None = None, size: int | None = None):
        """Create a Qdrant collection if it does not yet exist."""
        if not self.client or not getattr(self, "VectorParams", None):
            return False
        try:
            self.client.get_collection(collection_name)
            return True
        except Exception:
            pass

        dist_name = (distance or self.config.distance).upper()
        try:
            vector_distance = getattr(self.Distance, dist_name)
        except AttributeError:
            vector_distance = self.Distance.COSINE

        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=self.VectorParams(
                    size=size or self.config.embedding_dim,
                    distance=vector_distance,
                ),
            )
            return True
        except Exception as exc:
            print(f"⚠️  Failed to create collection {collection_name}: {exc}")
            return False
    
    def embed_text(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API"""
        if self._gpu_embeddings:
            try:
                return self._gpu_embeddings.embed_texts(texts)
            except Exception as exc:
                print(f"⚠️  GPU embedding failure, falling back to OpenAI: {exc}")
        try:
            import openai
            
            # Try to get API key from environment
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                print("⚠️  OPENAI_API_KEY not set. Using mock embeddings.")
                return self._mock_embeddings(texts)
            
            client = openai.OpenAI(api_key=api_key)
            
            response = client.embeddings.create(
                model=self.config.embedding_model,
                input=texts
            )
            
            return [item.embedding for item in response.data]
        
        except ImportError:
            print("⚠️  openai not installed. Install: pip install openai")
            return self._mock_embeddings(texts)
        except Exception as e:
            print(f"⚠️  Embedding error: {e}")
            return self._mock_embeddings(texts)
    
    def _mock_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings for testing"""
        import hashlib
        
        embeddings = []
        for text in texts:
            # Create deterministic "embedding" from text hash
            hash_bytes = hashlib.md5(text.encode()).digest()
            # Convert to floats between -1 and 1
            embedding = [
                (b - 128) / 128.0 
                for b in hash_bytes * (self.config.embedding_dim // 16 + 1)
            ][:self.config.embedding_dim]
            embeddings.append(embedding)
        
        return embeddings
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > chunk_size // 2:
                    chunk = text[start:start + break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
        
        return chunks
    
    def upsert_chat_messages(self, session_id: str, messages: List[Dict], realm: str, 
                            source: str, tags: List[str]):
        """Insert chat messages into vector DB"""
        if not self.client:
            print("⚠️  Qdrant not available - skipping vector upsert")
            return
        
        # Prepare texts for embedding
        texts = [msg['content'] for msg in messages]
        
        # Generate embeddings
        print(f"   Embedding {len(texts)} messages...")
        vectors = self.embed_text(texts)
        
        # Prepare points for Qdrant
        from qdrant_client.models import PointStruct
        import uuid
        
        points = []
        for i, (msg, vector) in enumerate(zip(messages, vectors)):
            # Use UUID instead of hash for point ID
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{session_id}_{i}"))
            
            point = PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "session_id": session_id,
                    "turn_index": i,
                    "role": msg['role'],
                    "content": msg['content'][:500],  # Preview only
                    "realm": realm,
                    "source": source,
                    "tags": tags,
                    "collection": "chat_messages"
                }
            )
            points.append(point)
        
        # Upsert to Qdrant
        try:
            self.client.upsert(
                collection_name="chat_messages",
                points=points
            )
            print(f"   ✅ Upserted {len(points)} vectors to chat_messages")
        except Exception as e:
            print(f"   ❌ Upsert failed: {e}")
    
    def search(self, query: str, collection: str = "chat_messages", 
               limit: int = 10, filters: Dict = None) -> List[Dict]:
        """Semantic search across collection"""
        if not self.client:
            print("⚠️  Qdrant not available")
            return []
        
        # Embed query
        query_vector = self.embed_text([query])[0]
        
        # Build filter if provided
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        search_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            if conditions:
                search_filter = Filter(must=conditions)
        
        # Search
        try:
            response = self.client.query_points(
                collection_name=collection,
                query=query_vector,
                limit=limit,
                query_filter=search_filter
            )
            
            results = response.points if hasattr(response, 'points') else []
            formatted = []
            for result in results:
                payload = getattr(result, 'payload', {}) or {}
                formatted.append(
                    {
                        "score": getattr(result, 'score', 1.0),
                        "content": payload.get("content", ""),
                        "session_id": payload.get("session_id"),
                        "realm": payload.get("realm"),
                        "role": payload.get("role"),
                        "tags": payload.get("tags", []),
                        "payload": payload,
                    }
                )
            return formatted
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return []
    
    def get_stats(self) -> Dict:
        """Get collection statistics"""
        if not self.client:
            return {}
        
        stats = {}
        collections = ["docs", "events", "lore", "reports", "chat_messages", "chat_summaries"]
        
        for collection in collections:
            try:
                info = self.client.get_collection(collection)
                stats[collection] = {
                    "vectors": info.vectors_count,
                    "points": info.points_count
                }
            except:
                stats[collection] = {"vectors": 0, "points": 0}
        
        return stats


def main():
    """CLI for vector DB operations"""
    import sys
    
    if len(sys.argv) < 2:
        print("\n🔍 SWARM VECTOR DB")
        print("=" * 60)
        print("Commands:")
        print("  setup              - Create Qdrant collections")
        print("  search <query>     - Search chat messages")
        print("  stats              - Show collection statistics")
        print("\nRequirements:")
        print("  1. Install: pip install qdrant-client openai")
        print("  2. Run Qdrant: docker run -p 6333:6333 qdrant/qdrant")
        print("  3. Set OPENAI_API_KEY environment variable")
        return
    
    vdb = SwarmVectorDB()
    command = sys.argv[1]
    
    if command == "setup":
        vdb.create_collections()
    
    elif command == "search" and len(sys.argv) >= 3:
        query = " ".join(sys.argv[2:])
        print(f"\n🔍 Searching: {query}")
        print("=" * 60)
        
        results = vdb.search(query, limit=5)
        
        if not results:
            print("No results found.")
        else:
            for i, result in enumerate(results, 1):
                print(f"\n{i}. [Score: {result['score']:.3f}] {result['realm']}")
                print(f"   Role: {result['role']}")
                print(f"   Content: {result['content'][:200]}...")
                if result['tags']:
                    print(f"   Tags: {', '.join(result['tags'])}")
    
    elif command == "stats":
        stats = vdb.get_stats()
        
        print("\n📊 VECTOR DB STATISTICS")
        print("=" * 60)
        
        total_vectors = 0
        for collection, info in stats.items():
            vectors = info.get('vectors', 0)
            total_vectors += vectors
            print(f"{collection:<20} {vectors:>6} vectors")
        
        print("=" * 60)
        print(f"{'TOTAL':<20} {total_vectors:>6} vectors")
        print()
    
    else:
        print("❌ Unknown command or missing arguments")


if __name__ == "__main__":
    main()
