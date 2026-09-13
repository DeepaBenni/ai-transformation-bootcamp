"""Tool registry.

The only module that decides which agent may reach which tool. Diagnostic cannot
write because no writer appears in its list - the restriction is structural, not
a promise made in a prompt.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from app.tools import read_tools, write_tools

READ_TOOLS: list[BaseTool] = [
    read_tools.get_user,
    read_tools.check_account_lockout,
    read_tools.get_ticket_history,
    read_tools.get_ci_health,
    read_tools.check_service_status,
    read_tools.get_disk_usage,
    read_tools.search_kb,
    read_tools.find_similar_tickets,
]

WRITE_TOOLS: list[BaseTool] = [
    write_tools.unlock_account,
    write_tools.reset_password,
    write_tools.restart_service,
    write_tools.grant_group_access,
    write_tools.escalate_to_l2,
]

# Per-agent scopes, consumed in Phase 2. Diagnostic gets facts only; Knowledge
# gets the two search tools; Action gets the writers (each of which still
# self-guards on the approval ledger).
DIAGNOSTIC_TOOLS: list[BaseTool] = [
    read_tools.get_user,
    read_tools.check_account_lockout,
    read_tools.get_ticket_history,
    read_tools.get_ci_health,
    read_tools.check_service_status,
    read_tools.get_disk_usage,
]
KNOWLEDGE_TOOLS: list[BaseTool] = [read_tools.search_kb, read_tools.find_similar_tickets]
ACTION_TOOLS: list[BaseTool] = WRITE_TOOLS

BY_NAME: dict[str, BaseTool] = {tool.name: tool for tool in READ_TOOLS + WRITE_TOOLS}
