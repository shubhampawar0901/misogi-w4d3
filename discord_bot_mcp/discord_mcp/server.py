"""Main Discord MCP Server implementation."""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP, Context
from mcp.types import TextContent, Tool

from .auth import auth_manager, Permission
from .config import settings
from .discord_client import discord_manager
from .rate_limiter import RateLimiter
from .audit_logger import AuditLogger


# Set up logging
logging.basicConfig(
    level=getattr(logging, settings.logging.level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(settings.logging.file) if settings.logging.file else logging.NullHandler()
    ]
)

logger = logging.getLogger(__name__)


# Initialize components
rate_limiter = RateLimiter()
audit_logger = AuditLogger()


@asynccontextmanager
async def lifespan(server: FastMCP):
    """Manage server lifecycle."""
    logger.info("Starting Discord MCP Server...")
    
    # Initialize Discord clients for all tenants
    for tenant in auth_manager.tenants.values():
        try:
            await discord_manager.get_client(tenant)
            logger.info(f"Initialized Discord client for tenant: {tenant.id}")
        except Exception as e:
            logger.error(f"Failed to initialize Discord client for tenant {tenant.id}: {e}")
    
    yield
    
    logger.info("Shutting down Discord MCP Server...")
    await discord_manager.close_all()


# Create MCP server
mcp = FastMCP(
    name=settings.mcp.server_name,
    lifespan=lifespan
)


def get_auth_context(ctx: Context) -> tuple:
    """Extract authentication context from request."""
    # In a real implementation, you'd extract this from headers
    # For now, we'll use the default tenant and API key
    api_key = settings.auth.api_key
    tenant_id = "default"
    
    # Authenticate
    api_key_obj = auth_manager.authenticate(api_key, tenant_id)
    if not api_key_obj:
        raise ValueError("Authentication failed")
    
    tenant = auth_manager.get_tenant(api_key_obj.tenant_id)
    if not tenant:
        raise ValueError("Tenant not found")
    
    return api_key_obj, tenant


@mcp.tool()
async def send_message(
    channel_id: int,
    content: str,
    embed: Optional[Dict[str, Any]] = None,
    ctx: Context = None
) -> str:
    """Send a message to a Discord channel.
    
    Args:
        channel_id: The ID of the Discord channel
        content: The message content to send
        embed: Optional embed object (Discord embed format)
    
    Returns:
        JSON string with message details
    """
    try:
        # Authentication and authorization
        api_key_obj, tenant = get_auth_context(ctx)
        
        if not auth_manager.authorize(api_key_obj, Permission.DISCORD_WRITE):
            raise ValueError("Insufficient permissions for discord:write")
        
        # Rate limiting
        if not rate_limiter.check_rate_limit(api_key_obj.key_hash):
            raise ValueError("Rate limit exceeded")
        
        # Send message
        result = await discord_manager.send_message(tenant, channel_id, content, embed)
        
        # Audit logging
        audit_logger.log_action(
            tenant_id=tenant.id,
            api_key_hash=api_key_obj.key_hash,
            action="send_message",
            details={
                "channel_id": channel_id,
                "content_length": len(content),
                "has_embed": embed is not None
            }
        )
        
        return f"Message sent successfully: {result}"
    
    except Exception as e:
        logger.error(f"Error sending message: {e}")
        raise


@mcp.tool()
async def get_messages(
    channel_id: int,
    limit: int = 50,
    before: Optional[int] = None,
    after: Optional[int] = None,
    ctx: Context = None
) -> str:
    """Retrieve message history from a Discord channel.
    
    Args:
        channel_id: The ID of the Discord channel
        limit: Maximum number of messages to retrieve (default: 50)
        before: Get messages before this message ID
        after: Get messages after this message ID
    
    Returns:
        JSON string with list of messages
    """
    try:
        # Authentication and authorization
        api_key_obj, tenant = get_auth_context(ctx)
        
        if not auth_manager.authorize(api_key_obj, Permission.DISCORD_READ):
            raise ValueError("Insufficient permissions for discord:read")
        
        # Rate limiting
        if not rate_limiter.check_rate_limit(api_key_obj.key_hash):
            raise ValueError("Rate limit exceeded")
        
        # Validate limit
        if limit > 100:
            limit = 100
        
        # Get messages
        messages = await discord_manager.get_messages(tenant, channel_id, limit, before, after)
        
        # Audit logging
        audit_logger.log_action(
            tenant_id=tenant.id,
            api_key_hash=api_key_obj.key_hash,
            action="get_messages",
            details={
                "channel_id": channel_id,
                "limit": limit,
                "message_count": len(messages)
            }
        )
        
        return f"Retrieved {len(messages)} messages: {messages}"
    
    except Exception as e:
        logger.error(f"Error getting messages: {e}")
        raise


@mcp.tool()
async def get_channel_info(
    channel_id: int,
    ctx: Context = None
) -> str:
    """Get information about a Discord channel.
    
    Args:
        channel_id: The ID of the Discord channel
    
    Returns:
        JSON string with channel information
    """
    try:
        # Authentication and authorization
        api_key_obj, tenant = get_auth_context(ctx)
        
        if not auth_manager.authorize(api_key_obj, Permission.DISCORD_READ):
            raise ValueError("Insufficient permissions for discord:read")
        
        # Rate limiting
        if not rate_limiter.check_rate_limit(api_key_obj.key_hash):
            raise ValueError("Rate limit exceeded")
        
        # Get channel info
        channel_info = await discord_manager.get_channel_info(tenant, channel_id)
        
        # Audit logging
        audit_logger.log_action(
            tenant_id=tenant.id,
            api_key_hash=api_key_obj.key_hash,
            action="get_channel_info",
            details={"channel_id": channel_id}
        )
        
        return f"Channel information: {channel_info}"
    
    except Exception as e:
        logger.error(f"Error getting channel info: {e}")
        raise


@mcp.tool()
async def search_messages(
    query: str,
    channel_id: Optional[int] = None,
    author_id: Optional[int] = None,
    limit: int = 50,
    ctx: Context = None
) -> str:
    """Search for messages with filters.
    
    Args:
        query: Search query string
        channel_id: Optional channel ID to search in
        author_id: Optional author ID to filter by
        limit: Maximum number of results (default: 50)
    
    Returns:
        JSON string with search results
    """
    try:
        # Authentication and authorization
        api_key_obj, tenant = get_auth_context(ctx)
        
        if not auth_manager.authorize(api_key_obj, Permission.DISCORD_READ):
            raise ValueError("Insufficient permissions for discord:read")
        
        # Rate limiting
        if not rate_limiter.check_rate_limit(api_key_obj.key_hash):
            raise ValueError("Rate limit exceeded")
        
        # Validate limit
        if limit > 100:
            limit = 100
        
        # Search messages
        results = await discord_manager.search_messages(tenant, query, channel_id, author_id, limit)
        
        # Audit logging
        audit_logger.log_action(
            tenant_id=tenant.id,
            api_key_hash=api_key_obj.key_hash,
            action="search_messages",
            details={
                "query": query,
                "channel_id": channel_id,
                "author_id": author_id,
                "result_count": len(results)
            }
        )
        
        return f"Found {len(results)} messages: {results}"
    
    except Exception as e:
        logger.error(f"Error searching messages: {e}")
        raise


@mcp.tool()
async def moderate_content(
    action: str,
    target_id: int,
    reason: Optional[str] = None,
    ctx: Context = None
) -> str:
    """Perform moderation actions.
    
    Args:
        action: Moderation action (delete_message, kick_user, ban_user)
        target_id: ID of the target (message ID or user ID)
        reason: Optional reason for the action
    
    Returns:
        JSON string with moderation result
    """
    try:
        # Authentication and authorization
        api_key_obj, tenant = get_auth_context(ctx)
        
        if not auth_manager.authorize(api_key_obj, Permission.DISCORD_MODERATE):
            raise ValueError("Insufficient permissions for discord:moderate")
        
        # Rate limiting
        if not rate_limiter.check_rate_limit(api_key_obj.key_hash):
            raise ValueError("Rate limit exceeded")
        
        # Validate action
        valid_actions = ["delete_message", "kick_user", "ban_user"]
        if action not in valid_actions:
            raise ValueError(f"Invalid action. Must be one of: {valid_actions}")
        
        # Perform moderation
        result = await discord_manager.moderate_content(tenant, action, target_id, reason)
        
        # Audit logging
        audit_logger.log_action(
            tenant_id=tenant.id,
            api_key_hash=api_key_obj.key_hash,
            action="moderate_content",
            details={
                "moderation_action": action,
                "target_id": target_id,
                "reason": reason
            }
        )
        
        return f"Moderation action completed: {result}"
    
    except Exception as e:
        logger.error(f"Error performing moderation: {e}")
        raise


def main():
    """Main entry point for the Discord MCP Server."""
    import argparse

    parser = argparse.ArgumentParser(description="Discord MCP Server")
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio",
                       help="Transport type (default: stdio)")
    parser.add_argument("--port", type=int, default=settings.mcp.server_port,
                       help="Port for SSE transport")
    parser.add_argument("--host", default="localhost",
                       help="Host for SSE transport")

    args = parser.parse_args()

    try:
        logger.info(f"Starting Discord MCP Server on {args.transport}")
        if args.transport == "sse":
            logger.info(f"SSE server will run on http://{args.host}:{args.port}")

        # Run the server
        mcp.run(
            transport=args.transport,
            port=args.port,
            host=args.host if args.transport == "sse" else None
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
