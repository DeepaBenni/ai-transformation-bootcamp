"""Populate the OpsMate synthetic estate from data/seed/estate.json.

Idempotent: every insert is an upsert keyed on the table's primary or unique key,
so running this script twice leaves the database in exactly the same state.

    python scripts/populate_database.py            # load or refresh
    python scripts/populate_database.py --reset    # truncate estate tables first
    python scripts/populate_database.py --verify   # counts only, no writes

This script deliberately depends only on pymysql and python-dotenv. It must run
before the application package exists.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Final

import pymysql
from dotenv import load_dotenv

ROOT: Final[Path] = Path(__file__).resolve().parent.parent
# Seed data lives in the repo-wide data/ folder, two levels above this project.
SEED_FILE: Final[Path] = ROOT.parent.parent / "data" / "seed" / "estate.json"

# Parents before children: foreign keys are enforced, so order matters.
LOAD_ORDER: Final[tuple[str, ...]] = (
    "users",
    "accounts",
    "hosts",
    "services",
    "config_items",
    "disk_usage",
    "access_groups",
    "group_memberships",
    "tickets",
)

LOGGER: Final[logging.Logger] = logging.getLogger("populate_database")


def configure_logging(verbose: bool = False) -> None:
    """Send readable, timestamped logs to stdout."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def connect() -> pymysql.connections.Connection:
    """Open a connection using the values in .env.

    Raises:
        SystemExit: if MySQL is unreachable, with the likely cause spelled out.
    """
    # Repo-root .env first, project-local .env second (the later call wins).
    load_dotenv(ROOT.parent.parent / ".env")
    load_dotenv(ROOT / ".env", override=True)
    try:
        return pymysql.connect(
            host=os.getenv("MYSQL_HOST", "127.0.0.1"),
            port=int(os.getenv("MYSQL_PORT", "3306")),
            user=os.getenv("MYSQL_USER", "opsmate"),
            password=os.getenv("MYSQL_PASSWORD", "opsmate"),
            database=os.getenv("MYSQL_DATABASE", "opsmate"),
            charset="utf8mb4",
            autocommit=False,
            cursorclass=pymysql.cursors.DictCursor,
        )
    except pymysql.err.OperationalError as exc:
        raise SystemExit(
            f"Cannot reach MySQL: {exc}\n"
            "Is the container running?  docker ps\n"
            "If not:                    ./scripts/docker_setup_db.ps1"
        ) from exc


def load_seed() -> dict[str, list[dict[str, Any]]]:
    """Read and validate the seed file.

    Raises:
        SystemExit: if the file is missing, or a table's rows have inconsistent keys.
    """
    if not SEED_FILE.exists():
        raise SystemExit(f"Seed file not found: {SEED_FILE}")

    data: dict[str, list[dict[str, Any]]] = json.loads(SEED_FILE.read_text(encoding="utf-8"))

    for table, rows in data.items():
        if not rows:
            continue
        expected = set(rows[0])
        for index, row in enumerate(rows):
            if set(row) != expected:
                raise SystemExit(
                    f"{table}[{index}] has keys {sorted(row)}, expected {sorted(expected)}. "
                    "Every row in a table must have identical keys."
                )
    return data


def upsert(cursor: pymysql.cursors.Cursor, table: str, rows: list[dict[str, Any]]) -> int:
    """Insert rows, updating any that already exist. Returns the number sent.

    Uses the MySQL 8.0.19+ row-alias form of ON DUPLICATE KEY UPDATE. The older
    VALUES() function is deprecated and emits warnings on MySQL 8.4.
    """
    if not rows:
        return 0

    columns = list(rows[0])
    column_sql = ", ".join(f"`{column}`" for column in columns)
    placeholders = ", ".join(["%s"] * len(columns))
    assignments = ", ".join(f"`{column}` = new.`{column}`" for column in columns)

    statement = (
        f"INSERT INTO `{table}` ({column_sql}) VALUES ({placeholders}) AS new "
        f"ON DUPLICATE KEY UPDATE {assignments}"
    )
    payload = [tuple(row[column] for column in columns) for row in rows]
    cursor.executemany(statement, payload)
    return len(rows)


def reset_estate(cursor: pymysql.cursors.Cursor) -> None:
    """Empty the estate tables, leaving the runtime tables untouched."""
    LOGGER.warning("Reset requested: truncating %d estate tables", len(LOAD_ORDER))
    cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
    for table in reversed(LOAD_ORDER):
        cursor.execute(f"TRUNCATE TABLE `{table}`")
    cursor.execute("SET FOREIGN_KEY_CHECKS = 1")


def report_counts(cursor: pymysql.cursors.Cursor) -> dict[str, int]:
    """Return a row count per estate table."""
    counts: dict[str, int] = {}
    for table in LOAD_ORDER:
        cursor.execute(f"SELECT COUNT(*) AS n FROM `{table}`")
        counts[table] = int(cursor.fetchone()["n"])
    return counts


def populate(*, reset: bool = False, verify_only: bool = False) -> dict[str, int]:
    """Load the estate. Returns the resulting row count per table."""
    seed = load_seed()
    connection = connect()
    try:
        with connection.cursor() as cursor:
            if verify_only:
                return report_counts(cursor)

            if reset:
                reset_estate(cursor)

            for table in LOAD_ORDER:
                sent = upsert(cursor, table, seed.get(table, []))
                LOGGER.info("%-20s %3d rows upserted", table, sent)

            connection.commit()
            return report_counts(cursor)
    except Exception:
        connection.rollback()
        LOGGER.exception("Populate failed; the transaction was rolled back")
        raise
    finally:
        connection.close()


def main() -> int:
    """Entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description="Populate the OpsMate synthetic estate.")
    parser.add_argument("--reset", action="store_true", help="truncate estate tables first")
    parser.add_argument("--verify", action="store_true", help="report counts without writing")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    args = parser.parse_args()

    configure_logging(args.verbose)
    counts = populate(reset=args.reset, verify_only=args.verify)

    total = sum(counts.values())
    LOGGER.info("-" * 44)
    for table, count in counts.items():
        LOGGER.info("%-20s %4d", table, count)
    LOGGER.info("%-20s %4d", "TOTAL", total)
    return 0 if total > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
