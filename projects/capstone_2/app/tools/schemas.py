"""Argument schemas for every tool.

The field descriptions are prompt surface: the model reads them when choosing
arguments. Write them for the model, not for a reviewer.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class UserIdArgs(BaseModel):
    """Arguments for a tool addressed to one employee."""

    user_id: str = Field(description="Employee id in the form U-1234, for example U-1042.")


class TextArgs(BaseModel):
    """Arguments for a free-text similarity search."""

    text: str = Field(description="The ticket text, or a short paraphrase of the problem.")


class QueryArgs(BaseModel):
    """Arguments for a knowledge-base search."""

    query: str = Field(description="A short natural-language description of the problem.")


class CiArgs(BaseModel):
    """Arguments for a configuration-item health check."""

    ci_name: str = Field(description="Configuration item name, for example 'app-prd-01'.")


class ServiceArgs(BaseModel):
    """Arguments for a service status check."""

    service: str = Field(description="Service name, for example 'mfa-push' or 'vpn-gateway'.")


class HostArgs(BaseModel):
    """Arguments for a host disk check."""

    host: str = Field(description="Hostname, for example 'file-prd-01'.")


class RestartArgs(BaseModel):
    """Arguments for restarting a service."""

    host: str = Field(description="The host the service runs on.")
    service: str = Field(description="The service to restart.")


class GroupArgs(BaseModel):
    """Arguments for a group access grant."""

    user_id: str = Field(description="Employee id in the form U-1234.")
    group: str = Field(description="Exact directory group name, for example 'vpn-users'.")


class EscalateArgs(BaseModel):
    """Arguments for handing a ticket to L2."""

    ticket_id: str = Field(description="The ticket reference, for example 'INC-88120'.")
    summary: str = Field(description="The structured handover, rendered for an engineer.")
