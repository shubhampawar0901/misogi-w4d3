"""Tests for rate limiting."""

import pytest
import time
from unittest.mock import patch

from discord_mcp.rate_limiter import RateLimiter


class TestRateLimiter:
    """Test cases for RateLimiter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.rate_limiter = RateLimiter()
        # Override the settings for testing
        self.rate_limiter.requests_per_minute = 10
        self.rate_limiter.burst_limit = 5
    
    def test_initial_request_allowed(self):
        """Test that initial request is allowed."""
        assert self.rate_limiter.check_rate_limit("test-key")
    
    def test_rate_limit_enforcement(self):
        """Test that rate limit is enforced."""
        api_key = "test-key"

        # Make requests up to the limit (using actual limit from settings)
        limit = self.rate_limiter.requests_per_minute
        for i in range(limit):
            assert self.rate_limiter.check_rate_limit(api_key)

        # Next request should be blocked
        assert not self.rate_limiter.check_rate_limit(api_key)
    
    def test_burst_limit_enforcement(self):
        """Test that burst limit is enforced."""
        api_key = "test-key"

        # Make burst limit requests quickly (using actual burst limit)
        burst_limit = self.rate_limiter.burst_limit
        for i in range(burst_limit):
            assert self.rate_limiter.check_rate_limit(api_key)

        # Next request should be blocked due to burst limit
        assert not self.rate_limiter.check_rate_limit(api_key)
    
    def test_rate_limit_window_sliding(self):
        """Test that rate limit window slides correctly."""
        api_key = "test-key"
        
        # Mock time to control the sliding window
        with patch('time.time') as mock_time:
            # Start at time 0
            mock_time.return_value = 0
            
            # Make requests up to limit
            limit = self.rate_limiter.requests_per_minute
            for i in range(limit):
                assert self.rate_limiter.check_rate_limit(api_key)
            
            # Should be blocked
            assert not self.rate_limiter.check_rate_limit(api_key)
            
            # Move time forward by 61 seconds (past the window)
            mock_time.return_value = 61
            
            # Should be allowed again
            assert self.rate_limiter.check_rate_limit(api_key)
    
    def test_different_keys_independent(self):
        """Test that different API keys have independent rate limits."""
        key1 = "test-key-1"
        key2 = "test-key-2"
        
        # Exhaust limit for key1
        limit = self.rate_limiter.requests_per_minute
        for i in range(limit):
            assert self.rate_limiter.check_rate_limit(key1)
        
        # key1 should be blocked
        assert not self.rate_limiter.check_rate_limit(key1)
        
        # key2 should still be allowed
        assert self.rate_limiter.check_rate_limit(key2)
    
    def test_get_rate_limit_status(self):
        """Test rate limit status reporting."""
        api_key = "test-key"
        
        # Make some requests
        for i in range(3):
            self.rate_limiter.check_rate_limit(api_key)
        
        status = self.rate_limiter.get_rate_limit_status(api_key)
        
        assert status["requests_in_minute"] == 3
        assert status["requests_per_minute_limit"] == self.rate_limiter.requests_per_minute
        assert status["remaining_requests"] == self.rate_limiter.requests_per_minute - 3
        assert status["requests_in_burst_window"] == 3
        assert status["burst_limit"] == self.rate_limiter.burst_limit
        assert status["remaining_burst"] == self.rate_limiter.burst_limit - 3
    
    def test_reset_rate_limit(self):
        """Test rate limit reset functionality."""
        api_key = "test-key"
        
        # Exhaust the rate limit
        limit = self.rate_limiter.requests_per_minute
        for i in range(limit):
            self.rate_limiter.check_rate_limit(api_key)
        
        # Should be blocked
        assert not self.rate_limiter.check_rate_limit(api_key)
        
        # Reset the rate limit
        self.rate_limiter.reset_rate_limit(api_key)
        
        # Should be allowed again
        assert self.rate_limiter.check_rate_limit(api_key)
    
    def test_burst_window_timing(self):
        """Test burst window timing is correct."""
        api_key = "test-key"
        
        with patch('time.time') as mock_time:
            # Start at time 0
            mock_time.return_value = 0
            
            # Make burst limit requests
            burst_limit = self.rate_limiter.burst_limit
            for i in range(burst_limit):
                assert self.rate_limiter.check_rate_limit(api_key)
            
            # Should be blocked due to burst
            assert not self.rate_limiter.check_rate_limit(api_key)
            
            # Move time forward by 11 seconds (past burst window)
            mock_time.return_value = 11
            
            # Should be allowed again (burst window reset)
            assert self.rate_limiter.check_rate_limit(api_key)
    
    def test_empty_history_status(self):
        """Test status for API key with no history."""
        status = self.rate_limiter.get_rate_limit_status("new-key")
        
        assert status["requests_in_minute"] == 0
        assert status["remaining_requests"] == self.rate_limiter.requests_per_minute
        assert status["requests_in_burst_window"] == 0
        assert status["remaining_burst"] == self.rate_limiter.burst_limit
