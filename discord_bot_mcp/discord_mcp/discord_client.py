"""Discord client wrapper for MCP server."""

import asyncio
import logging
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

import discord
from discord.ext import commands

from .config import settings
from .auth import Tenant


logger = logging.getLogger(__name__)


class DiscordClientManager:
    """Manages Discord bot clients for multiple tenants."""
    
    def __init__(self):
        self.clients: Dict[str, discord.Client] = {}
        self._setup_intents()
    
    def _setup_intents(self) -> discord.Intents:
        """Set up Discord intents."""
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.guild_messages = True
        intents.guild_reactions = True
        intents.members = True
        return intents
    
    async def get_client(self, tenant: Tenant) -> discord.Client:
        """Get or create a Discord client for a tenant."""
        if tenant.id not in self.clients:
            await self._create_client(tenant)
        
        return self.clients[tenant.id]
    
    async def _create_client(self, tenant: Tenant):
        """Create a new Discord client for a tenant."""
        intents = self._setup_intents()
        client = discord.Client(intents=intents)
        
        @client.event
        async def on_ready():
            logger.info(f"Discord client for tenant {tenant.id} is ready: {client.user}")
        
        @client.event
        async def on_error(event, *args, **kwargs):
            logger.error(f"Discord error in tenant {tenant.id}: {event}", exc_info=True)
        
        # Start the client
        try:
            await client.login(tenant.discord_bot_token)
            # Don't call client.start() here as it's blocking
            # Instead, we'll manage the connection manually
            self.clients[tenant.id] = client
            logger.info(f"Created Discord client for tenant {tenant.id}")
        except Exception as e:
            logger.error(f"Failed to create Discord client for tenant {tenant.id}: {e}")
            raise
    
    async def send_message(self, tenant: Tenant, channel_id: int, content: str, 
                          embed: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send a message to a Discord channel."""
        client = await self.get_client(tenant)
        
        try:
            channel = client.get_channel(channel_id)
            if not channel:
                # Try to fetch the channel if not in cache
                channel = await client.fetch_channel(channel_id)
            
            if not channel:
                raise ValueError(f"Channel {channel_id} not found")
            
            # Create embed if provided
            discord_embed = None
            if embed:
                discord_embed = discord.Embed.from_dict(embed)
            
            message = await channel.send(content=content, embed=discord_embed)
            
            return {
                "id": message.id,
                "content": message.content,
                "author": {
                    "id": message.author.id,
                    "name": message.author.name,
                    "display_name": message.author.display_name
                },
                "channel_id": message.channel.id,
                "guild_id": message.guild.id if message.guild else None,
                "created_at": message.created_at.isoformat(),
                "url": message.jump_url
            }
        
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            raise
    
    async def get_messages(self, tenant: Tenant, channel_id: int, limit: int = 50,
                          before: Optional[int] = None, after: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get messages from a Discord channel."""
        client = await self.get_client(tenant)
        
        try:
            channel = client.get_channel(channel_id)
            if not channel:
                channel = await client.fetch_channel(channel_id)
            
            if not channel:
                raise ValueError(f"Channel {channel_id} not found")
            
            # Convert message IDs to discord.Object if provided
            before_obj = discord.Object(before) if before else None
            after_obj = discord.Object(after) if after else None
            
            messages = []
            async for message in channel.history(limit=limit, before=before_obj, after=after_obj):
                messages.append({
                    "id": message.id,
                    "content": message.content,
                    "author": {
                        "id": message.author.id,
                        "name": message.author.name,
                        "display_name": message.author.display_name,
                        "bot": message.author.bot
                    },
                    "channel_id": message.channel.id,
                    "guild_id": message.guild.id if message.guild else None,
                    "created_at": message.created_at.isoformat(),
                    "edited_at": message.edited_at.isoformat() if message.edited_at else None,
                    "attachments": [
                        {
                            "id": att.id,
                            "filename": att.filename,
                            "url": att.url,
                            "size": att.size
                        } for att in message.attachments
                    ],
                    "embeds": [embed.to_dict() for embed in message.embeds],
                    "reactions": [
                        {
                            "emoji": str(reaction.emoji),
                            "count": reaction.count
                        } for reaction in message.reactions
                    ]
                })
            
            return messages
        
        except Exception as e:
            logger.error(f"Failed to get messages: {e}")
            raise
    
    async def get_channel_info(self, tenant: Tenant, channel_id: int) -> Dict[str, Any]:
        """Get information about a Discord channel."""
        client = await self.get_client(tenant)
        
        try:
            channel = client.get_channel(channel_id)
            if not channel:
                channel = await client.fetch_channel(channel_id)
            
            if not channel:
                raise ValueError(f"Channel {channel_id} not found")
            
            channel_info = {
                "id": channel.id,
                "name": channel.name,
                "type": str(channel.type),
                "guild_id": channel.guild.id if hasattr(channel, 'guild') and channel.guild else None,
                "created_at": channel.created_at.isoformat()
            }
            
            # Add additional info based on channel type
            if isinstance(channel, discord.TextChannel):
                channel_info.update({
                    "topic": channel.topic,
                    "position": channel.position,
                    "nsfw": channel.nsfw,
                    "slowmode_delay": channel.slowmode_delay,
                    "category": channel.category.name if channel.category else None
                })
            
            return channel_info
        
        except Exception as e:
            logger.error(f"Failed to get channel info: {e}")
            raise
    
    async def search_messages(self, tenant: Tenant, query: str, channel_id: Optional[int] = None,
                             author_id: Optional[int] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Search for messages (simplified implementation)."""
        # Note: Discord API doesn't provide direct search functionality
        # This is a simplified implementation that searches recent messages
        client = await self.get_client(tenant)
        
        try:
            channels_to_search = []
            
            if channel_id:
                channel = client.get_channel(channel_id)
                if channel:
                    channels_to_search.append(channel)
            else:
                # Search in all accessible text channels
                for guild in client.guilds:
                    channels_to_search.extend([
                        ch for ch in guild.text_channels 
                        if ch.permissions_for(guild.me).read_message_history
                    ])
            
            matching_messages = []
            query_lower = query.lower()
            
            for channel in channels_to_search:
                if len(matching_messages) >= limit:
                    break
                
                try:
                    async for message in channel.history(limit=100):  # Search recent 100 messages per channel
                        if len(matching_messages) >= limit:
                            break
                        
                        # Check if message matches query
                        if query_lower in message.content.lower():
                            # Check author filter if provided
                            if author_id and message.author.id != author_id:
                                continue
                            
                            matching_messages.append({
                                "id": message.id,
                                "content": message.content,
                                "author": {
                                    "id": message.author.id,
                                    "name": message.author.name,
                                    "display_name": message.author.display_name
                                },
                                "channel_id": message.channel.id,
                                "channel_name": message.channel.name,
                                "guild_id": message.guild.id if message.guild else None,
                                "created_at": message.created_at.isoformat()
                            })
                
                except discord.Forbidden:
                    # Skip channels we can't access
                    continue
            
            return matching_messages
        
        except Exception as e:
            logger.error(f"Failed to search messages: {e}")
            raise
    
    async def moderate_content(self, tenant: Tenant, action: str, target_id: int, 
                              reason: Optional[str] = None) -> Dict[str, Any]:
        """Perform moderation actions."""
        client = await self.get_client(tenant)
        
        try:
            if action == "delete_message":
                # Find and delete message
                for guild in client.guilds:
                    for channel in guild.text_channels:
                        try:
                            message = await channel.fetch_message(target_id)
                            await message.delete(reason=reason)
                            return {
                                "action": "delete_message",
                                "target_id": target_id,
                                "success": True,
                                "reason": reason
                            }
                        except discord.NotFound:
                            continue
                        except discord.Forbidden:
                            continue
                
                raise ValueError(f"Message {target_id} not found or cannot be deleted")
            
            elif action == "kick_user":
                # Kick user from guild
                if not settings.discord.guild_id:
                    raise ValueError("Guild ID not configured for moderation")
                
                guild = client.get_guild(int(settings.discord.guild_id))
                if not guild:
                    raise ValueError("Guild not found")
                
                member = guild.get_member(target_id)
                if not member:
                    member = await guild.fetch_member(target_id)
                
                await member.kick(reason=reason)
                return {
                    "action": "kick_user",
                    "target_id": target_id,
                    "success": True,
                    "reason": reason
                }
            
            elif action == "ban_user":
                # Ban user from guild
                if not settings.discord.guild_id:
                    raise ValueError("Guild ID not configured for moderation")
                
                guild = client.get_guild(int(settings.discord.guild_id))
                if not guild:
                    raise ValueError("Guild not found")
                
                user = client.get_user(target_id)
                if not user:
                    user = await client.fetch_user(target_id)
                
                await guild.ban(user, reason=reason)
                return {
                    "action": "ban_user",
                    "target_id": target_id,
                    "success": True,
                    "reason": reason
                }
            
            else:
                raise ValueError(f"Unknown moderation action: {action}")
        
        except Exception as e:
            logger.error(f"Failed to perform moderation action: {e}")
            raise
    
    async def close_all(self):
        """Close all Discord clients."""
        for client in self.clients.values():
            if not client.is_closed():
                await client.close()
        self.clients.clear()


# Global Discord client manager
discord_manager = DiscordClientManager()
