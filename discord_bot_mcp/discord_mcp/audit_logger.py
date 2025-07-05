"""Audit logging for Discord MCP Server."""

import json
import logging
import time
from typing import Any, Dict, Optional

from .config import settings


class AuditLogger:
    """Handles audit logging for all MCP operations."""
    
    def __init__(self):
        self.enabled = settings.security.enable_audit_logging
        
        if self.enabled:
            # Set up audit logger
            self.audit_logger = logging.getLogger("audit")
            self.audit_logger.setLevel(logging.INFO)
            
            # Create file handler for audit logs
            if settings.security.audit_log_file:
                handler = logging.FileHandler(settings.security.audit_log_file)
                formatter = logging.Formatter('%(asctime)s - %(message)s')
                handler.setFormatter(formatter)
                self.audit_logger.addHandler(handler)
                
                # Prevent audit logs from going to root logger
                self.audit_logger.propagate = False
    
    def log_action(self, tenant_id: str, api_key_hash: str, action: str, 
                   details: Optional[Dict[str, Any]] = None):
        """Log an audit event.
        
        Args:
            tenant_id: ID of the tenant performing the action
            api_key_hash: Hash of the API key used
            action: Action being performed
            details: Additional details about the action
        """
        if not self.enabled:
            return
        
        audit_entry = {
            "timestamp": time.time(),
            "iso_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "tenant_id": tenant_id,
            "api_key_hash": api_key_hash[:8] + "...",  # Only log first 8 chars for security
            "action": action,
            "details": details or {}
        }
        
        # Log as JSON for easy parsing
        self.audit_logger.info(json.dumps(audit_entry))
    
    def log_authentication(self, api_key_hash: str, tenant_id: Optional[str], 
                          success: bool, reason: Optional[str] = None):
        """Log authentication attempts.
        
        Args:
            api_key_hash: Hash of the API key used
            tenant_id: Tenant ID if provided
            success: Whether authentication was successful
            reason: Reason for failure if applicable
        """
        if not self.enabled:
            return
        
        auth_entry = {
            "timestamp": time.time(),
            "iso_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event_type": "authentication",
            "api_key_hash": api_key_hash[:8] + "..." if api_key_hash else None,
            "tenant_id": tenant_id,
            "success": success,
            "reason": reason
        }
        
        self.audit_logger.info(json.dumps(auth_entry))
    
    def log_authorization(self, api_key_hash: str, tenant_id: str, permission: str, 
                         granted: bool):
        """Log authorization checks.
        
        Args:
            api_key_hash: Hash of the API key
            tenant_id: Tenant ID
            permission: Permission being checked
            granted: Whether permission was granted
        """
        if not self.enabled:
            return
        
        authz_entry = {
            "timestamp": time.time(),
            "iso_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event_type": "authorization",
            "api_key_hash": api_key_hash[:8] + "...",
            "tenant_id": tenant_id,
            "permission": permission,
            "granted": granted
        }
        
        self.audit_logger.info(json.dumps(authz_entry))
    
    def log_rate_limit(self, api_key_hash: str, action: str, blocked: bool):
        """Log rate limiting events.
        
        Args:
            api_key_hash: Hash of the API key
            action: Action being rate limited
            blocked: Whether the request was blocked
        """
        if not self.enabled:
            return
        
        rate_limit_entry = {
            "timestamp": time.time(),
            "iso_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event_type": "rate_limit",
            "api_key_hash": api_key_hash[:8] + "...",
            "action": action,
            "blocked": blocked
        }
        
        self.audit_logger.info(json.dumps(rate_limit_entry))
    
    def log_error(self, tenant_id: str, api_key_hash: str, action: str, 
                  error: str, details: Optional[Dict[str, Any]] = None):
        """Log error events.
        
        Args:
            tenant_id: Tenant ID
            api_key_hash: Hash of the API key
            action: Action that caused the error
            error: Error message
            details: Additional error details
        """
        if not self.enabled:
            return
        
        error_entry = {
            "timestamp": time.time(),
            "iso_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event_type": "error",
            "tenant_id": tenant_id,
            "api_key_hash": api_key_hash[:8] + "..." if api_key_hash else None,
            "action": action,
            "error": error,
            "details": details or {}
        }
        
        self.audit_logger.info(json.dumps(error_entry))
