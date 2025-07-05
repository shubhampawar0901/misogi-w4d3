"""Rate limiting for Discord MCP Server."""

import time
from collections import defaultdict, deque
from typing import Dict, Deque

from .config import settings


class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self):
        # Store request timestamps for each API key
        self.request_history: Dict[str, Deque[float]] = defaultdict(deque)
        self.requests_per_minute = settings.rate_limit.requests_per_minute
        self.burst_limit = settings.rate_limit.burst
    
    def check_rate_limit(self, api_key_hash: str) -> bool:
        """Check if a request is within rate limits.
        
        Args:
            api_key_hash: Hashed API key to check
            
        Returns:
            True if request is allowed, False if rate limited
        """
        current_time = time.time()
        window_start = current_time - 60  # 1 minute window
        
        # Get request history for this API key
        history = self.request_history[api_key_hash]
        
        # Remove old requests outside the window
        while history and history[0] < window_start:
            history.popleft()
        
        # Check if we're within limits
        if len(history) >= self.requests_per_minute:
            return False
        
        # Check burst limit (requests in last 10 seconds)
        burst_window_start = current_time - 10
        recent_requests = sum(1 for timestamp in history if timestamp >= burst_window_start)
        
        if recent_requests >= self.burst_limit:
            return False
        
        # Add current request to history
        history.append(current_time)
        
        return True
    
    def get_rate_limit_status(self, api_key_hash: str) -> Dict[str, int]:
        """Get current rate limit status for an API key.
        
        Args:
            api_key_hash: Hashed API key to check
            
        Returns:
            Dictionary with rate limit information
        """
        current_time = time.time()
        window_start = current_time - 60
        burst_window_start = current_time - 10
        
        history = self.request_history[api_key_hash]
        
        # Count requests in current window
        requests_in_window = sum(1 for timestamp in history if timestamp >= window_start)
        requests_in_burst_window = sum(1 for timestamp in history if timestamp >= burst_window_start)
        
        return {
            "requests_in_minute": requests_in_window,
            "requests_per_minute_limit": self.requests_per_minute,
            "requests_in_burst_window": requests_in_burst_window,
            "burst_limit": self.burst_limit,
            "remaining_requests": max(0, self.requests_per_minute - requests_in_window),
            "remaining_burst": max(0, self.burst_limit - requests_in_burst_window)
        }
    
    def reset_rate_limit(self, api_key_hash: str):
        """Reset rate limit for an API key (admin function)."""
        if api_key_hash in self.request_history:
            del self.request_history[api_key_hash]
