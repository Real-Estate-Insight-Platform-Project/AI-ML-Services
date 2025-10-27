"""
Session Management with Redis
Handles conversation history, context, and caching
"""
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import hashlib

import redis
from redis.exceptions import RedisError
import structlog

from src.config import settings
from src.tools.tools import convert_for_json

logger = structlog.get_logger()


class SessionManager:
    """Manages conversation sessions with Redis"""
    
    def __init__(self):
        try:
            self.redis_client = redis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_keepalive=True,
                health_check_interval=30
            )
            self.redis_client.ping()
            self.available = True
            logger.info("redis_connected")
        except RedisError as e:
            logger.error("redis_connection_failed", error=str(e))
            self.available = False
            self.memory_store: Dict[str, Any] = {}
    
    def _make_session_key(self, session_id: str) -> str:
        """Generate Redis key for session"""
        return f"session:{session_id}"
    
    def _make_cache_key(self, prefix: str, data: str) -> str:
        """Generate cache key with hash"""
        hash_value = hashlib.md5(data.encode()).hexdigest()
        return f"cache:{prefix}:{hash_value}"
    
    def get_conversation(self, session_id: str) -> List[Dict]:
        """Retrieve conversation history"""
        if not self.available:
            return self.memory_store.get(session_id, {}).get("conversation", [])
        
        try:
            key = self._make_session_key(session_id)
            data = self.redis_client.get(key)
            
            if data:
                session_data = json.loads(data)
                return session_data.get("conversation", [])
            
            return []
            
        except (RedisError, json.JSONDecodeError) as e:
            logger.error("get_conversation_error", session_id=session_id, error=str(e))
            return []
    
    def save_conversation(
        self,
        session_id: str,
        messages: List[Dict],
        metadata: Optional[Dict] = None
    ) -> bool:
        """Save conversation history"""
        try:
            session_data = {
                "conversation": convert_for_json(messages),
                "metadata": metadata or {},
                "updated_at": datetime.now().isoformat(),
                "session_id": session_id
            }
            
            if not self.available:
                if session_id not in self.memory_store:
                    self.memory_store[session_id] = {}
                self.memory_store[session_id].update(session_data)
                return True
            
            key = self._make_session_key(session_id)
            self.redis_client.setex(
                key,
                settings.redis_session_ttl,
                json.dumps(session_data)
            )
            
            logger.info("conversation_saved", session_id=session_id, message_count=len(messages))
            return True
            
        except (RedisError, TypeError, json.JSONEncodeError) as e:
            logger.error("save_conversation_error", session_id=session_id, error=str(e))
            return False
    
    def add_message(self, session_id: str, role: str, content: str, metadata: Optional[Dict] = None) -> bool:
        """Add a single message to conversation"""
        conversation = self.get_conversation(session_id)
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        conversation.append(message)
        return self.save_conversation(session_id, conversation)
    
    def clear_conversation(self, session_id: str) -> bool:
        """Clear conversation history"""
        if not self.available:
            if session_id in self.memory_store:
                del self.memory_store[session_id]
            return True
        
        try:
            key = self._make_session_key(session_id)
            self.redis_client.delete(key)
            logger.info("conversation_cleared", session_id=session_id)
            return True
            
        except RedisError as e:
            logger.error("clear_conversation_error", session_id=session_id, error=str(e))
            return False
    
    def get_all_sessions(self) -> List[str]:
        """Get all active session IDs"""
        if not self.available:
            return list(self.memory_store.keys())
        
        try:
            keys = self.redis_client.keys("session:*")
            return [key.replace("session:", "") for key in keys]
            
        except RedisError as e:
            logger.error("get_all_sessions_error", error=str(e))
            return []
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get session metadata"""
        conversation = self.get_conversation(session_id)
        
        if not conversation:
            return {"exists": False}
        
        info = {
            "exists": True,
            "message_count": len(conversation),
            "created_at": conversation[0].get("timestamp") if conversation else None,
            "last_activity": conversation[-1].get("timestamp") if conversation else None
        }
        
        if self.available:
            try:
                key = self._make_session_key(session_id)
                ttl = self.redis_client.ttl(key)
                info["ttl_seconds"] = ttl if ttl > 0 else None
                info["ttl_minutes"] = round(ttl / 60, 1) if ttl > 0 else None
            except RedisError:
                pass
        
        return info
    
    def extend_session(self, session_id: str) -> bool:
        """Extend session TTL"""
        if not self.available:
            return True
        
        try:
            key = self._make_session_key(session_id)
            return bool(self.redis_client.expire(key, settings.redis_session_ttl))
            
        except RedisError as e:
            logger.error("extend_session_error", session_id=session_id, error=str(e))
            return False
    
    def cache_query_result(self, query: str, result: Any, ttl: Optional[int] = None) -> bool:
        """Cache query result"""
        if not self.available:
            return False
        
        try:
            key = self._make_cache_key("query", query)
            cache_data = {
                "query": query,
                "result": convert_for_json(result),
                "cached_at": datetime.now().isoformat()
            }
            
            cache_ttl = ttl or settings.redis_cache_ttl
            self.redis_client.setex(key, cache_ttl, json.dumps(cache_data))
            
            logger.info("query_cached", query_hash=key[:20])
            return True
            
        except (RedisError, TypeError, json.JSONEncodeError) as e:
            logger.error("cache_query_error", error=str(e))
            return False
    
    def get_cached_query(self, query: str) -> Optional[Any]:
        """Get cached query result"""
        if not self.available:
            return None
        
        try:
            key = self._make_cache_key("query", query)
            data = self.redis_client.get(key)
            
            if data:
                cache_data = json.loads(data)
                logger.info("cache_hit", query_hash=key[:20])
                return cache_data.get("result")
            
            logger.info("cache_miss", query_hash=key[:20])
            return None
            
        except (RedisError, json.JSONDecodeError) as e:
            logger.error("get_cached_query_error", error=str(e))
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Redis statistics"""
        if not self.available:
            return {
                "available": False,
                "mode": "in-memory",
                "active_sessions": len(self.memory_store)
            }
        
        try:
            info = self.redis_client.info()
            session_count = len(self.redis_client.keys("session:*"))
            cache_count = len(self.redis_client.keys("cache:*"))
            
            return {
                "available": True,
                "mode": "redis",
                "redis_version": info.get("redis_version", "unknown"),
                "active_sessions": session_count,
                "cached_queries": cache_count,
                "used_memory_human": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0)
            }
            
        except RedisError as e:
            logger.error("get_stats_error", error=str(e))
            return {"available": False, "error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """Check Redis health"""
        if not self.available:
            return {
                "status": "fallback",
                "message": "Using in-memory storage",
                "available": False
            }
        
        try:
            self.redis_client.ping()
            return {
                "status": "healthy",
                "available": True,
                "latency_ms": "< 1"
            }
            
        except RedisError as e:
            logger.error("redis_health_check_failed", error=str(e))
            return {
                "status": "unhealthy",
                "available": False,
                "error": str(e)
            }


# Global session manager instance
session_manager = SessionManager()
