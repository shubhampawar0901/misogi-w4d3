"""Tests for authentication and authorization."""

import pytest
import time
from unittest.mock import patch

from discord_mcp.auth import AuthManager, Permission, Tenant, APIKey


class TestAuthManager:
    """Test cases for AuthManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.auth_manager = AuthManager()
    
    def test_default_tenant_creation(self):
        """Test that default tenant is created."""
        assert "default" in self.auth_manager.tenants
        default_tenant = self.auth_manager.tenants["default"]
        assert default_tenant.name == "Default Tenant"
        assert Permission.DISCORD_READ in default_tenant.permissions
    
    def test_api_key_hashing(self):
        """Test API key hashing."""
        key = "test-api-key"
        hash1 = self.auth_manager._hash_key(key)
        hash2 = self.auth_manager._hash_key(key)
        
        assert hash1 == hash2
        assert hash1 != key
        assert len(hash1) == 64  # SHA256 hex digest length
    
    def test_authentication_success(self):
        """Test successful authentication."""
        # Use the default API key from settings
        with patch('discord_mcp.auth.settings.auth.api_key', 'test-key'):
            auth_manager = AuthManager()
            result = auth_manager.authenticate('test-key')
            
            assert result is not None
            assert result.tenant_id == "default"
            assert result.last_used is not None
    
    def test_authentication_failure(self):
        """Test failed authentication."""
        result = self.auth_manager.authenticate('invalid-key')
        assert result is None
    
    def test_authorization_success(self):
        """Test successful authorization."""
        api_key = APIKey(
            key_hash="test-hash",
            tenant_id="default",
            permissions={Permission.DISCORD_READ},
            created_at=time.time()
        )
        
        assert self.auth_manager.authorize(api_key, Permission.DISCORD_READ)
    
    def test_authorization_failure(self):
        """Test failed authorization."""
        api_key = APIKey(
            key_hash="test-hash",
            tenant_id="default",
            permissions={Permission.DISCORD_READ},
            created_at=time.time()
        )
        
        assert not self.auth_manager.authorize(api_key, Permission.DISCORD_MODERATE)
    
    def test_api_key_expiration(self):
        """Test API key expiration."""
        expired_key = APIKey(
            key_hash="test-hash",
            tenant_id="default",
            permissions={Permission.DISCORD_READ},
            created_at=time.time(),
            expires_at=time.time() - 3600  # Expired 1 hour ago
        )
        
        assert expired_key.is_expired()
    
    def test_create_tenant(self):
        """Test tenant creation."""
        tenant = self.auth_manager.create_tenant(
            tenant_id="test-tenant",
            name="Test Tenant",
            discord_bot_token="test-token",
            permissions={Permission.DISCORD_READ}
        )
        
        assert tenant.id == "test-tenant"
        assert tenant.name == "Test Tenant"
        assert "test-tenant" in self.auth_manager.tenants
    
    def test_create_duplicate_tenant(self):
        """Test creating duplicate tenant fails."""
        self.auth_manager.create_tenant(
            tenant_id="test-tenant",
            name="Test Tenant",
            discord_bot_token="test-token",
            permissions={Permission.DISCORD_READ}
        )
        
        with pytest.raises(ValueError, match="already exists"):
            self.auth_manager.create_tenant(
                tenant_id="test-tenant",
                name="Another Tenant",
                discord_bot_token="test-token-2",
                permissions={Permission.DISCORD_READ}
            )
    
    def test_create_api_key(self):
        """Test API key creation."""
        # First create a tenant
        self.auth_manager.create_tenant(
            tenant_id="test-tenant",
            name="Test Tenant",
            discord_bot_token="test-token",
            permissions={Permission.DISCORD_READ}
        )
        
        api_key = self.auth_manager.create_api_key(
            key="test-api-key",
            tenant_id="test-tenant",
            permissions={Permission.DISCORD_READ}
        )
        
        assert api_key.tenant_id == "test-tenant"
        assert Permission.DISCORD_READ in api_key.permissions
        
        # Verify it's stored
        key_hash = self.auth_manager._hash_key("test-api-key")
        assert key_hash in self.auth_manager.api_keys
    
    def test_create_api_key_invalid_tenant(self):
        """Test creating API key for non-existent tenant fails."""
        with pytest.raises(ValueError, match="does not exist"):
            self.auth_manager.create_api_key(
                key="test-api-key",
                tenant_id="non-existent",
                permissions={Permission.DISCORD_READ}
            )


class TestPermission:
    """Test cases for Permission enum."""
    
    def test_permission_values(self):
        """Test permission enum values."""
        assert Permission.DISCORD_READ.value == "discord:read"
        assert Permission.DISCORD_WRITE.value == "discord:write"
        assert Permission.DISCORD_MODERATE.value == "discord:moderate"
        assert Permission.DISCORD_ADMIN.value == "discord:admin"


class TestTenant:
    """Test cases for Tenant dataclass."""
    
    def test_tenant_creation(self):
        """Test tenant creation."""
        tenant = Tenant(
            id="test",
            name="Test Tenant",
            discord_bot_token="token",
            permissions={Permission.DISCORD_READ}
        )
        
        assert tenant.id == "test"
        assert tenant.name == "Test Tenant"
        assert tenant.discord_bot_token == "token"
        assert tenant.created_at is not None


class TestAPIKey:
    """Test cases for APIKey dataclass."""
    
    def test_api_key_creation(self):
        """Test API key creation."""
        api_key = APIKey(
            key_hash="hash",
            tenant_id="tenant",
            permissions={Permission.DISCORD_READ},
            created_at=time.time()
        )
        
        assert api_key.key_hash == "hash"
        assert api_key.tenant_id == "tenant"
        assert not api_key.is_expired()
        assert api_key.has_permission(Permission.DISCORD_READ)
        assert not api_key.has_permission(Permission.DISCORD_WRITE)
