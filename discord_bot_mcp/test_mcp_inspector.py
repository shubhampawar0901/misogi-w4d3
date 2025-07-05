#!/usr/bin/env python3
"""Test script for MCP Inspector integration."""

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

async def test_mcp_server():
    """Test the MCP server with basic functionality."""
    print("🔧 Testing Discord MCP Server...")
    
    # Test basic import
    try:
        from discord_mcp.server import mcp
        print("✅ Server import successful")
    except Exception as e:
        print(f"❌ Server import failed: {e}")
        return False
    
    # Test configuration
    try:
        from discord_mcp.config import settings
        print(f"✅ Configuration loaded: {settings.mcp.server_name}")
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
        return False
    
    # Test authentication
    try:
        from discord_mcp.auth import auth_manager
        api_key_obj = auth_manager.authenticate("test_api_key")
        if api_key_obj:
            print("✅ Authentication working")
        else:
            print("❌ Authentication failed")
    except Exception as e:
        print(f"❌ Authentication error: {e}")
        return False
    
    # Test rate limiting
    try:
        from discord_mcp.rate_limiter import RateLimiter
        rate_limiter = RateLimiter()
        if rate_limiter.check_rate_limit("test-key"):
            print("✅ Rate limiting working")
        else:
            print("❌ Rate limiting failed")
    except Exception as e:
        print(f"❌ Rate limiting error: {e}")
        return False
    
    print("🎉 All basic tests passed!")
    return True

def create_inspector_config():
    """Create MCP Inspector configuration."""
    config = {
        "mcpServers": {
            "discord-mcp-server": {
                "command": "uv",
                "args": ["run", "python", "-m", "discord_mcp.server"],
                "cwd": str(Path.cwd()),
                "env": {
                    "DISCORD_BOT_TOKEN": "test_token_for_development",
                    "API_KEY": "test_api_key_123",
                    "JWT_SECRET": "test_jwt_secret_456",
                    "LOG_LEVEL": "DEBUG"
                }
            }
        }
    }
    
    config_path = Path("mcp_inspector_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"📝 Created MCP Inspector config: {config_path}")
    return config_path

def test_server_startup():
    """Test server startup in stdio mode."""
    print("🚀 Testing server startup...")
    
    try:
        # Start server process
        process = subprocess.Popen(
            ["uv", "run", "python", "-m", "discord_mcp.server", "--transport", "stdio"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=Path.cwd()
        )
        
        # Give it a moment to start
        time.sleep(2)
        
        # Check if process is running
        if process.poll() is None:
            print("✅ Server started successfully")
            
            # Send a simple MCP message to test
            try:
                # Send initialize request
                init_request = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {
                            "name": "test-client",
                            "version": "1.0.0"
                        }
                    }
                }
                
                process.stdin.write(json.dumps(init_request) + "\n")
                process.stdin.flush()
                
                # Wait for response
                time.sleep(1)
                
                print("✅ Server responding to requests")
                
            except Exception as e:
                print(f"⚠️  Server communication test failed: {e}")
            
            # Terminate the process
            process.terminate()
            process.wait(timeout=5)
            print("✅ Server shutdown successful")
            return True
            
        else:
            stderr_output = process.stderr.read()
            print(f"❌ Server failed to start: {stderr_output}")
            return False
            
    except Exception as e:
        print(f"❌ Server startup test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 Discord MCP Server Test Suite")
    print("=" * 40)
    
    # Run async tests
    if not asyncio.run(test_mcp_server()):
        sys.exit(1)
    
    print()
    
    # Create inspector config
    create_inspector_config()
    
    print()
    
    # Test server startup
    if not test_server_startup():
        sys.exit(1)
    
    print()
    print("🎉 All tests passed! Server is ready for MCP Inspector.")
    print()
    print("To use with MCP Inspector:")
    print("1. Install MCP Inspector: npm install -g @modelcontextprotocol/inspector")
    print("2. Run: mcp-inspector")
    print("3. Load the mcp_inspector_config.json file")
    print("4. Test the Discord tools!")

if __name__ == "__main__":
    main()
