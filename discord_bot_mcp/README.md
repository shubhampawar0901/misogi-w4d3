# Discord MCP Server

A Model Context Protocol (MCP) server that enables AI models to interact with Discord through a secure, authenticated interface.

## Features

- **Discord Tools**: Send messages, retrieve history, get channel info, search messages, and moderate content
- **Authentication**: API key authentication with granular permission system
- **Multi-tenancy**: Support for multiple Discord bots
- **MCP Inspector**: Integration for debugging and monitoring
- **Security**: Rate limiting, input validation, and audit logging
- **Testing**: Comprehensive unit tests with >80% coverage

## Quick Start

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your Discord bot token and other settings
   ```

3. **Run the server**:
   ```bash
   uv run discord-mcp-server
   ```

4. **Test with MCP Inspector**:
   ```bash
   mcp dev discord_mcp/server.py
   ```

## Discord Bot Setup

1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Create a new application
3. Go to "Bot" section and create a bot
4. Copy the bot token to your `.env` file
5. Enable necessary intents (Message Content Intent, etc.)
6. Invite bot to your server with appropriate permissions

## MCP Tools

### send_message
Send messages to Discord channels
- **Parameters**: `channel_id`, `content`, `embed` (optional)
- **Permissions**: `discord:write`

### get_messages
Retrieve message history from channels
- **Parameters**: `channel_id`, `limit`, `before`, `after`
- **Permissions**: `discord:read`

### get_channel_info
Fetch channel metadata and information
- **Parameters**: `channel_id`
- **Permissions**: `discord:read`

### search_messages
Search messages with filters
- **Parameters**: `query`, `channel_id`, `author_id`, `limit`
- **Permissions**: `discord:read`

### moderate_content
Delete messages and manage users
- **Parameters**: `action`, `target_id`, `reason`
- **Permissions**: `discord:moderate`

## Authentication

The server uses API key authentication with a permission system:

```python
# Example client usage
headers = {
    "Authorization": "Bearer your-api-key",
    "X-Tenant-ID": "your-tenant-id"  # Optional for multi-tenancy
}
```

## Development

### Running Tests
```bash
uv run pytest
```

### Code Formatting
```bash
uv run black .
uv run isort .
```

### Type Checking
```bash
uv run mypy .
```

## License

MIT License
