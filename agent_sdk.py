"""
AGENT SDK - Universal Event Logging & Vector Embedding

Every agent action flows through this SDK:
1. Insert to events table (universal event spine)
2. Store content in event_contents if provided
3. Auto-embed to appropriate Qdrant collection
4. Return event_id for tracking

Usage:
    from agent_sdk import log_event, log_report, log_lore
    
    # Basic event
    event_id = log_event("ritual_executed", "agent-ritual", 
                        realm="diabetes-therapy",
                        metadata={"ritual_id": "ritual_123", "outcome": "success"})
    
    # Report with embedding
    event_id = log_report("agent-oracle", "diabetes-therapy", 
                         "Nightly summary: 10 rituals executed, 70% success rate...")
    
    # Lore narrative
    event_id = log_lore("diabetes-therapy", 
                       "Last night, the swarm focused on metformin explorations...")
"""

import os
import json
import uuid
import duckdb
from datetime import datetime
from typing import Dict, Optional, Any

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Optional vector DB integration
try:
    from swarm_vector_db import SwarmVectorDB
    VECTOR_DB_AVAILABLE = True
except ImportError:
    VECTOR_DB_AVAILABLE = False


class AgentSDK:
    """Universal SDK for agent event logging and vector embedding"""
    
    def __init__(self, database_path: str = "evidence.duckdb", enable_vectors: bool = True):
        self.database_path = database_path
        self.enable_vectors = enable_vectors
        self.vector_db = None
        
        # Initialize vector DB if available and enabled
        if enable_vectors and VECTOR_DB_AVAILABLE:
            try:
                self.vector_db = SwarmVectorDB()
                if self.vector_db.client:
                    print("🔍 Agent SDK: Vector DB enabled for automatic embedding")
            except Exception as e:
                print(f"⚠️  Agent SDK: Vector DB initialization failed: {e}")
                self.vector_db = None
    
    def log_event(
        self,
        event_type: str,
        agent_id: str,
        realm: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        content: Optional[str] = None,
        content_type: str = "plain",
        status: str = "pending"
    ) -> str:
        """
        Log an event to the universal event spine
        
        Args:
            event_type: Type of event (ritual_executed, report_generated, lore_update, etc.)
            agent_id: Agent that generated this event
            realm: Realm/domain (diabetes-therapy, gym-supplements, etc.)
            metadata: Additional structured data (JSON)
            content: Text content to store and embed
            content_type: Content format (plain, markdown, json)
            status: Event status (pending, success, error)
            
        Returns:
            event_id: Unique event identifier
        """
        event_id = f"evt_{uuid.uuid4().hex[:12]}"
        content_id = None
        
        try:
            conn = duckdb.connect(self.database_path)
            
            # Prepare metadata with content_id if content provided
            final_metadata = metadata or {}
            if content:
                content_id = f"content_{uuid.uuid4().hex[:12]}"
                final_metadata['content_id'] = content_id
            
            # Determine final status (will be 'success' unless explicitly set to error/pending)
            final_status = 'success' if status == 'pending' else status
            final_completed_at = datetime.now().isoformat() if final_status == 'success' else None
            
            # Insert event
            conn.execute("""
                INSERT INTO events (
                    event_id, source, event_type, realm, agent_id, 
                    status, created_at, completed_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                event_id,
                agent_id,
                event_type,
                realm,
                agent_id,
                final_status,
                datetime.now().isoformat(),
                final_completed_at,
                json.dumps(final_metadata)
            ])
            
            # Store content if provided
            if content:
                conn.execute("""
                    INSERT INTO event_contents (
                        content_id, event_id, content_type, raw_text, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                """, [
                    content_id,
                    event_id,
                    content_type,
                    content,
                    datetime.now().isoformat()
                ])
            
            # Embed content to vector DB if available
            if content and self.vector_db and self.vector_db.client:
                try:
                    collection = self._determine_collection(event_type)
                    self._embed_event_content(
                        event_id=event_id,
                        content=content,
                        collection=collection,
                        realm=realm,
                        agent_id=agent_id,
                        event_type=event_type,
                        metadata=metadata
                    )
                except Exception as e:
                    print(f"⚠️  Failed to embed event {event_id}: {e}")
            
            # Final commit
            conn.commit()
            conn.close()
            return event_id
            
        except Exception as e:
            print(f"❌ Failed to log event: {e}")
            # Try to mark as error
            try:
                conn = duckdb.connect(self.database_path)
                error_metadata = metadata or {}
                error_metadata['error'] = str(e)
                conn.execute("""
                    UPDATE events 
                    SET status = 'error', completed_at = ?, metadata = ?
                    WHERE event_id = ?
                """, [datetime.now().isoformat(), json.dumps(error_metadata), event_id])
                conn.commit()
                conn.close()
            except:
                pass
            raise
    
    def _determine_collection(self, event_type: str) -> str:
        """Determine which Qdrant collection to use based on event type"""
        if event_type in ["lore_update", "lore_generated", "narrative_created"]:
            return "lore"
        elif event_type in ["report_generated", "analytics_complete", "metrics_summary"]:
            return "reports"
        elif event_type in ["file_ingested", "document_processed"]:
            return "docs"
        else:
            return "events"  # Default collection for all other events
    
    def _embed_event_content(
        self,
        event_id: str,
        content: str,
        collection: str,
        realm: Optional[str],
        agent_id: str,
        event_type: str,
        metadata: Optional[Dict]
    ):
        """Embed event content to Qdrant vector DB"""
        # Chunk if content is large
        if len(content) > 1500:
            chunks = self.vector_db.chunk_text(content, chunk_size=1000, overlap=200)
        else:
            chunks = [content]
        
        # Embed chunks
        embeddings = self.vector_db.embed_text(chunks)
        
        # Create points for Qdrant
        from qdrant_client.models import PointStruct
        
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{event_id}_{i}"))
            
            payload = {
                "event_id": event_id,
                "agent_id": agent_id,
                "event_type": event_type,
                "realm": realm,
                "content": chunk[:500],  # Preview
                "chunk_index": i,
                "total_chunks": len(chunks),
                "created_at": datetime.now().isoformat(),
                "collection": collection
            }
            
            # Add metadata if provided
            if metadata:
                payload["metadata"] = metadata
            
            points.append(PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            ))
        
        # Upsert to Qdrant
        self.vector_db.client.upsert(
            collection_name=collection,
            points=points
        )
        
        print(f"✅ Embedded {len(points)} chunks to {collection} collection")


# Convenience functions for common use cases

_sdk_instance = None

def _get_sdk() -> AgentSDK:
    """Get or create SDK instance"""
    global _sdk_instance
    if _sdk_instance is None:
        _sdk_instance = AgentSDK()
    return _sdk_instance


def log_event(
    event_type: str,
    agent_id: str,
    realm: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    content: Optional[str] = None
) -> str:
    """
    Log an event to the universal event spine
    
    Example:
        log_event("ritual_executed", "agent-ritual",
                 realm="diabetes-therapy",
                 metadata={"ritual_id": "ritual_123", "outcome": "success"})
    """
    return _get_sdk().log_event(event_type, agent_id, realm, metadata, content)


def log_report(agent_id: str, realm: str, report_text: str, metadata: Optional[Dict] = None) -> str:
    """
    Log a report with automatic embedding to reports collection
    
    Example:
        log_report("agent-oracle", "diabetes-therapy",
                  "Nightly summary: 10 rituals executed, 70% success rate...")
    """
    return _get_sdk().log_event(
        event_type="report_generated",
        agent_id=agent_id,
        realm=realm,
        metadata=metadata,
        content=report_text
    )


def log_lore(realm: str, lore_text: str, metadata: Optional[Dict] = None) -> str:
    """
    Log a lore narrative with automatic embedding to lore collection
    
    Example:
        log_lore("diabetes-therapy",
                "Last night, the swarm focused on metformin explorations...")
    """
    return _get_sdk().log_event(
        event_type="lore_update",
        agent_id="agent-alchemist",
        realm=realm,
        metadata=metadata,
        content=lore_text
    )


def log_error(agent_id: str, error_message: str, realm: Optional[str] = None, metadata: Optional[Dict] = None) -> str:
    """
    Log an error event
    
    Example:
        log_error("agent-ritual", "Failed to execute chaos_infusion: insufficient mastery",
                 realm="diabetes-therapy")
    """
    error_metadata = metadata or {}
    error_metadata["error"] = error_message
    
    return _get_sdk().log_event(
        event_type="error",
        agent_id=agent_id,
        realm=realm,
        metadata=error_metadata,
        status="error"
    )


# CLI for testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python agent_sdk.py <command>")
        print("Commands:")
        print("  test              - Run test event logging")
        print("  test-report       - Test report logging with embedding")
        print("  test-lore         - Test lore logging with embedding")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "test":
        print("🧪 Testing Agent SDK...")
        event_id = log_event(
            event_type="test_event",
            agent_id="agent-test",
            realm="testing",
            metadata={"test": True, "timestamp": datetime.now().isoformat()},
            content="This is a test event to verify SDK functionality."
        )
        print(f"✅ Test event logged: {event_id}")
        print("Query it: python swarmctl_ask.py 'test event SDK functionality'")
    
    elif command == "test-report":
        print("🧪 Testing report logging...")
        event_id = log_report(
            agent_id="agent-oracle",
            realm="testing",
            report_text="""
# Nightly Oracle Report - Test

## Activity Summary
- 15 events processed
- 3 rituals executed (success rate: 67%)
- 2 zone movements detected

## Key Insights
- Testing realm shows high activity
- Agent SDK integration successful
- Vector embeddings operational

## Recommendations
1. Continue testing with real workloads
2. Monitor vector search quality
3. Verify event spine integrity
""",
            metadata={"report_type": "nightly", "test": True}
        )
        print(f"✅ Test report logged: {event_id}")
        print("Query it: python swarmctl_ask.py 'oracle nightly report'")
    
    elif command == "test-lore":
        print("🧪 Testing lore logging...")
        event_id = log_lore(
            realm="testing",
            lore_text="""
In the twilight hours of the testing realm, the Agent SDK awakened.

The Oracle watched as events flowed like rivers through the event spine,
each action preserved in the institutional memory. The Alchemist wove
these threads into narratives, creating meaning from the chaos.

The swarm learned that night: every action matters, every event tells
a story, and through the SDK, all stories become searchable wisdom.
""",
            metadata={"narrative_type": "mythic", "test": True}
        )
        print(f"✅ Test lore logged: {event_id}")
        print("Query it: python swarmctl_ask.py 'Agent SDK awakened testing realm'")
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
