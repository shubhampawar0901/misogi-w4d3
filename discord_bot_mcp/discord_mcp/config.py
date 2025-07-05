"""Configuration management for Discord MCP Server."""

import os
from typing import Optional
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class DiscordConfig(BaseSettings):
    """Discord-specific configuration."""

    bot_token: str = Field("test_token", env="DISCORD_BOT_TOKEN")
    guild_id: Optional[str] = Field(None, env="DISCORD_GUILD_ID")
    
    class Config:
        env_prefix = "DISCORD_"


class MCPConfig(BaseSettings):
    """MCP server configuration."""
    
    server_name: str = Field("Discord MCP Server", env="MCP_SERVER_NAME")
    server_version: str = Field("0.1.0", env="MCP_SERVER_VERSION")
    server_port: int = Field(3001, env="MCP_SERVER_PORT")
    
    class Config:
        env_prefix = "MCP_"


class AuthConfig(BaseSettings):
    """Authentication configuration."""

    api_key: str = Field("test_api_key", env="API_KEY")
    jwt_secret: str = Field("test_jwt_secret", env="JWT_SECRET")
    
    class Config:
        env_prefix = "AUTH_"


class RateLimitConfig(BaseSettings):
    """Rate limiting configuration."""
    
    requests_per_minute: int = Field(60, env="RATE_LIMIT_REQUESTS_PER_MINUTE")
    burst: int = Field(10, env="RATE_LIMIT_BURST")
    
    class Config:
        env_prefix = "RATE_LIMIT_"


class LoggingConfig(BaseSettings):
    """Logging configuration."""
    
    level: str = Field("INFO", env="LOG_LEVEL")
    file: Optional[str] = Field(None, env="LOG_FILE")
    
    class Config:
        env_prefix = "LOG_"


class SecurityConfig(BaseSettings):
    """Security configuration."""
    
    enable_multi_tenancy: bool = Field(True, env="ENABLE_MULTI_TENANCY")
    max_tenants: int = Field(10, env="MAX_TENANTS")
    enable_audit_logging: bool = Field(True, env="ENABLE_AUDIT_LOGGING")
    audit_log_file: str = Field("audit.log", env="AUDIT_LOG_FILE")
    
    class Config:
        env_prefix = "SECURITY_"


class Settings:
    """Main settings container."""
    
    def __init__(self):
        self.discord = DiscordConfig()
        self.mcp = MCPConfig()
        self.auth = AuthConfig()
        self.rate_limit = RateLimitConfig()
        self.logging = LoggingConfig()
        self.security = SecurityConfig()


# Global settings instance
settings = Settings()
