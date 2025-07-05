"""Authentication and authorization for Discord MCP Server."""

import hashlib
import hmac
import json
import time
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum

from .config import settings


class Permission(Enum):
    """Available permissions for Discord operations."""
    
    DISCORD_READ = "discord:read"
    DISCORD_WRITE = "discord:write"
    DISCORD_MODERATE = "discord:moderate"
    DISCORD_ADMIN = "discord:admin"


@dataclass
class Tenant:
    """Represents a tenant in multi-tenant setup."""
    
    id: str
    name: str
    discord_bot_token: str
    permissions: Set[Permission]
    rate_limit_override: Optional[int] = None
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


@dataclass
class APIKey:
    """Represents an API key with associated permissions."""
    
    key_hash: str
    tenant_id: str
    permissions: Set[Permission]
    created_at: float
    expires_at: Optional[float] = None
    last_used: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if the API key is expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if the API key has a specific permission."""
        return permission in self.permissions


class AuthManager:
    """Manages authentication and authorization."""
    
    def __init__(self):
        self.tenants: Dict[str, Tenant] = {}
        self.api_keys: Dict[str, APIKey] = {}
        self._setup_default_tenant()
    
    def _setup_default_tenant(self):
        """Set up the default tenant."""
        default_tenant = Tenant(
            id="default",
            name="Default Tenant",
            discord_bot_token=settings.discord.bot_token,
            permissions={
                Permission.DISCORD_READ,
                Permission.DISCORD_WRITE,
                Permission.DISCORD_MODERATE
            }
        )
        self.tenants["default"] = default_tenant
        
        # Create default API key
        default_key_hash = self._hash_key(settings.auth.api_key)
        default_api_key = APIKey(
            key_hash=default_key_hash,
            tenant_id="default",
            permissions=default_tenant.permissions,
            created_at=time.time()
        )
        self.api_keys[default_key_hash] = default_api_key
    
    def _hash_key(self, key: str) -> str:
        """Hash an API key for secure storage."""
        return hashlib.sha256(key.encode()).hexdigest()
    
    def authenticate(self, api_key: str, tenant_id: Optional[str] = None) -> Optional[APIKey]:
        """Authenticate an API key and return the associated APIKey object."""
        key_hash = self._hash_key(api_key)
        
        if key_hash not in self.api_keys:
            return None
        
        api_key_obj = self.api_keys[key_hash]
        
        if api_key_obj.is_expired():
            return None
        
        # Check tenant ID if multi-tenancy is enabled
        if settings.security.enable_multi_tenancy and tenant_id:
            if api_key_obj.tenant_id != tenant_id:
                return None
        
        # Update last used timestamp
        api_key_obj.last_used = time.time()
        
        return api_key_obj
    
    def authorize(self, api_key_obj: APIKey, permission: Permission) -> bool:
        """Check if an API key has the required permission."""
        return api_key_obj.has_permission(permission)
    
    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get a tenant by ID."""
        return self.tenants.get(tenant_id)
    
    def create_tenant(self, tenant_id: str, name: str, discord_bot_token: str, 
                     permissions: Set[Permission]) -> Tenant:
        """Create a new tenant."""
        if len(self.tenants) >= settings.security.max_tenants:
            raise ValueError("Maximum number of tenants reached")
        
        if tenant_id in self.tenants:
            raise ValueError(f"Tenant {tenant_id} already exists")
        
        tenant = Tenant(
            id=tenant_id,
            name=name,
            discord_bot_token=discord_bot_token,
            permissions=permissions
        )
        
        self.tenants[tenant_id] = tenant
        return tenant
    
    def create_api_key(self, key: str, tenant_id: str, permissions: Set[Permission], 
                      expires_at: Optional[float] = None) -> APIKey:
        """Create a new API key."""
        key_hash = self._hash_key(key)
        
        if key_hash in self.api_keys:
            raise ValueError("API key already exists")
        
        if tenant_id not in self.tenants:
            raise ValueError(f"Tenant {tenant_id} does not exist")
        
        api_key_obj = APIKey(
            key_hash=key_hash,
            tenant_id=tenant_id,
            permissions=permissions,
            created_at=time.time(),
            expires_at=expires_at
        )
        
        self.api_keys[key_hash] = api_key_obj
        return api_key_obj


# Global auth manager instance
auth_manager = AuthManager()
