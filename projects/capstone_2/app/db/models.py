"""SQLAlchemy models over the Phase 0 schema.

These mirror db/schema.sql exactly. The SQL file remains the source of truth -
these classes exist so tools can query with types instead of raw strings.

Boundary: only the estate tables and the approval/action_log tables are modelled
here. The high-volume `runs` and `audit_events` tables are written with raw SQL
from app/audit/trace.py and must not gain ORM models.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import JSON, ForeignKey, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for every OpsMate model."""


class User(Base):
    """An employee in the synthetic estate."""

    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    full_name: Mapped[str] = mapped_column(String(128))
    email: Mapped[str] = mapped_column(String(128))
    department: Mapped[str] = mapped_column(String(64))
    manager: Mapped[str | None] = mapped_column(String(128))
    employment_status: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime]

    account: Mapped[Account | None] = relationship(back_populates="user", uselist=False)


class Account(Base):
    """Directory account state for a user."""

    __tablename__ = "accounts"

    user_id: Mapped[str] = mapped_column(String(16), ForeignKey("users.user_id"), primary_key=True)
    state: Mapped[str] = mapped_column(String(16))
    failed_logins_24h: Mapped[int]
    disabled_reason: Mapped[str | None] = mapped_column(String(128))
    last_login_at: Mapped[datetime | None]
    password_set_at: Mapped[datetime | None]

    user: Mapped[User] = relationship(back_populates="account")


class Host(Base):
    """A server or network device."""

    __tablename__ = "hosts"

    host: Mapped[str] = mapped_column(String(64), primary_key=True)
    os: Mapped[str] = mapped_column(String(64))
    environment: Mapped[str] = mapped_column(String(16))
    site: Mapped[str] = mapped_column(String(64))


class Service(Base):
    """A service running on a host."""

    __tablename__ = "services"

    service: Mapped[str] = mapped_column(String(64), primary_key=True)
    host: Mapped[str] = mapped_column(String(64), ForeignKey("hosts.host"))
    status: Mapped[str] = mapped_column(String(16))
    depends_on: Mapped[str | None] = mapped_column(String(128))
    status_since: Mapped[datetime | None]
    restart_window: Mapped[str] = mapped_column(String(32))


class ConfigItem(Base):
    """A CMDB configuration item."""

    __tablename__ = "config_items"

    ci_name: Mapped[str] = mapped_column(String(64), primary_key=True)
    ci_type: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16))
    cpu_pct: Mapped[Decimal]
    mem_pct: Mapped[Decimal]
    last_checkin_at: Mapped[datetime | None]
    owner_team: Mapped[str] = mapped_column(String(64))


class DiskUsage(Base):
    """One mount point on one host."""

    __tablename__ = "disk_usage"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    host: Mapped[str] = mapped_column(String(64), ForeignKey("hosts.host"))
    mount: Mapped[str] = mapped_column(String(64))
    used_pct: Mapped[Decimal]
    free_gb: Mapped[Decimal]


class AccessGroup(Base):
    """A directory group that access can be granted to."""

    __tablename__ = "access_groups"

    group_name: Mapped[str] = mapped_column(String(64), primary_key=True)
    description: Mapped[str] = mapped_column(String(255))
    owner: Mapped[str] = mapped_column(String(128))
    manager_approval: Mapped[bool]


class GroupMembership(Base):
    """A user's membership of an access group."""

    __tablename__ = "group_memberships"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(16), ForeignKey("users.user_id"))
    group_name: Mapped[str] = mapped_column(String(64), ForeignKey("access_groups.group_name"))
    granted_at: Mapped[datetime]


class Ticket(Base):
    """A historical service-desk ticket."""

    __tablename__ = "tickets"

    ticket_id: Mapped[str] = mapped_column(String(24), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(16), ForeignKey("users.user_id"))
    body: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16))
    resolution: Mapped[str | None] = mapped_column(String(512))
    created_at: Mapped[datetime]
    closed_at: Mapped[datetime | None]


class Approval(Base):
    """A recorded human decision on a proposed state change."""

    __tablename__ = "approvals"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(40))
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    action: Mapped[str] = mapped_column(String(64))
    # Bare `dict` has no SQLAlchemy Core type; JSON maps it to the MySQL JSON column.
    args_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    decision: Mapped[str] = mapped_column(String(16))
    approver: Mapped[str] = mapped_column(String(128))
    reason: Mapped[str | None] = mapped_column(Text)
    decided_at: Mapped[datetime]


class ActionLogEntry(Base):
    """A state change that actually happened. One row per real modification."""

    __tablename__ = "action_log"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(40))
    action: Mapped[str] = mapped_column(String(64))
    args_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    approver: Mapped[str] = mapped_column(String(128))
    result: Mapped[str] = mapped_column(String(512))
    executed_at: Mapped[datetime]
