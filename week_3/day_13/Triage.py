
import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

# ------------------------------------------------------------------ paths --
HERE = Path(__file__).resolve().parent               # week_3/day_13
REPO_ROOT = HERE.parents[1]                           # ai-transformation-bootcamp
KB_JSON = HERE.parent / "day_12" / "kb.json"          # Day 12's output, reused unchanged
CHROMA_DIR = HERE / "chroma_day13"                    # reuse the store the notebook built
TRACE_PATH = HERE / "day13_trace.jsonl"

load_dotenv(REPO_ROOT / ".env")

MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
COLLECTION = "kb_day13"
RECURSION_LIMIT = 12
TRUNCATED = "Sorry, need more steps"      # the sentinel an even recursion limit returns


# ------------------------------------------------------------------ trace --
TRACE = []
_T0 = time.perf_counter()


def reset_trace():
    global _T0
    TRACE.clear()
    _T0 = time.perf_counter()


def log(event, **fields):
    TRACE.append({"t_ms": round((time.perf_counter() - _T0) * 1000, 1),
                  "event": event, **fields})


def _traced(name, args, fn, verbose):
    log("tool_call", tool=name, args=args)
    if verbose:
        print(f"    -> {name}({', '.join(f'{k}={v!r}' for k, v in args.items())})")
    t = time.perf_counter()
    try:
        out = fn()
        log("observation", tool=name,
            latency_ms=round((time.perf_counter() - t) * 1000, 1), result=out)
        return out
    except Exception as e:
        log("tool_error", tool=name,
            latency_ms=round((time.perf_counter() - t) * 1000, 1), error=repr(e))
        raise


# -------------------------------------------------------------- mock data --
USERS = {
    "U-1042": {"user_id": "U-1042", "name": "Priya Raman", "email": "priya.raman@example.com",
               "department": "Finance", "manager": "M-07 Alan Whitfield",
               "employment_status": "active", "hire_date": "2021-03-15"},
    "U-2087": {"user_id": "U-2087", "name": "Daniel Okafor", "email": "daniel.okafor@example.com",
               "department": "Sales", "manager": "M-12 Rosa Delgado",
               "employment_status": "leaver", "hire_date": "2019-08-01",
               "leave_date": "2026-08-29"},
    "U-3311": {"user_id": "U-3311", "name": "Mei Lin", "email": "mei.lin@example.com",
               "department": "Engineering", "manager": "M-04 Sanjay Iyer",
               "employment_status": "active", "hire_date": "2023-11-06"},
}
ACCOUNTS = {
    "U-1042": {"user_id": "U-1042", "account_state": "locked", "failed_logins_24h": 5,
               "last_lockout": "2026-09-06T08:12:00Z", "mfa_enrolled": True,
               "password_age_days": 88},
    "U-2087": {"user_id": "U-2087", "account_state": "disabled", "failed_logins_24h": 0,
               "last_lockout": None, "mfa_enrolled": True, "password_age_days": 402,
               "disabled_on": "2026-08-29T18:00:00Z", "disabled_reason": "offboarding"},
    "U-3311": {"user_id": "U-3311", "account_state": "active", "failed_logins_24h": 0,
               "last_lockout": None, "mfa_enrolled": True, "password_age_days": 12},
}
TICKETS = {
    "U-1042": [{"ticket": "INC-88120", "opened": "2026-09-06", "priority": "P3",
                "summary": "Cannot sign in, account locked", "state": "open"}],
    "U-2087": [{"ticket": "INC-88090", "opened": "2026-09-05", "priority": "P3",
                "summary": "Cannot access email since Monday", "state": "open"}],
    "U-3311": [{"ticket": "INC-88131", "opened": "2026-09-06", "priority": "P3",
                "summary": "MFA code rejected on every attempt", "state": "open"}],
}

_STORE = None
_VERBOSE = True


def store():
    """Open the Chroma collection, embedding the knowledge base the first time only."""
    global _STORE
    if _STORE is not None:
        return _STORE
    s = Chroma(collection_name=COLLECTION,
               embedding_function=OpenAIEmbeddings(model=EMBED_MODEL),
               persist_directory=str(CHROMA_DIR),
               collection_metadata={"hnsw:space": "cosine"})
    if s._collection.count() == 0:
        if not KB_JSON.exists():
            sys.exit(f"{KB_JSON} is missing. Build it first:\n"
                     f"  python week_3/day_12/ask.py --rebuild")
        docs = json.load(open(KB_JSON, encoding="utf-8"))
        s.add_documents([Document(page_content=d["text"],
                                  metadata={"id": d["id"], "title": d["title"],
                                            "type": d["type"]}) for d in docs],
                        ids=[d["id"] for d in docs])
        print(f"indexed {s._collection.count()} documents", file=sys.stderr)
    _STORE = s
    return s


# ------------------------------------------------------------------ tools --
@tool
def lookup_user(user_id: str) -> dict:
    """Look up an employee's identity record: name, email, department, manager and
    employment status. Use this first to confirm the person exists and is a current
    employee."""
    return _traced("lookup_user", {"user_id": user_id},
                   lambda: USERS.get(user_id, {"error": f"no such user {user_id}"}), _VERBOSE)


@tool
def check_account_status(user_id: str) -> dict:
    """Check the directory account state for a user: locked, active or disabled, plus
    failed login count, MFA enrolment and password age. 'locked' and 'disabled' are
    different states with different procedures."""
    return _traced("check_account_status", {"user_id": user_id},
                   lambda: ACCOUNTS.get(user_id, {"error": f"no account for {user_id}"}), _VERBOSE)


@tool
def get_ticket_history(user_id: str) -> list:
    """Return the user's recent service-desk tickets, newest first. Useful for spotting
    a repeating problem or a related open ticket."""
    return _traced("get_ticket_history", {"user_id": user_id},
                   lambda: TICKETS.get(user_id, []), _VERBOSE)


@tool
def search_kb(query: str) -> list:
    """Search the IT knowledge base of articles and runbooks. Returns the top 3 matches
    as {id, title, text}. Use this to find the correct procedure before acting; cite the
    id you used."""
    def _go():
        return [{"id": d.metadata["id"], "title": d.metadata["title"],
                 "score": round(sc, 4), "text": d.page_content}
                for d, sc in store().similarity_search_with_relevance_scores(query, k=3)]
    return _traced("search_kb", {"query": query}, _go, _VERBOSE)


TOOLS = [lookup_user, check_account_status, get_ticket_history, search_kb]

SYSTEM = """You are an IT service-desk agent. You act only through the tools provided.

Procedure:
1. Identify the user with lookup_user before anything else.
2. Check the account state with check_account_status.
3. Find the governing runbook or article with search_kb and follow it exactly.
4. Cite the knowledge-base id you relied on, like [RB-01].

Rules:
- Never invent a user, an account state, a ticket or a procedure. If a tool did not
  tell you something, you do not know it.
- Call each tool at most once per user unless a later step genuinely needs new data.
- A locked account and a disabled account are different. Read the runbook before acting.
- When you have enough to answer, answer. Do not keep calling tools to feel thorough.

End with a short recommendation of the concrete next action for the service desk."""


def cost(in_tok, out_tok):
    return in_tok / 1e6 * 0.15 + out_tok / 1e6 * 0.60


def save_trace(path=None):
    path = Path(path or TRACE_PATH)
    with open(path, "w", encoding="utf-8") as f:
        for e in TRACE:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return path


def main():
    ap = argparse.ArgumentParser(description="Hand an IT ticket to the Day 13 agent.")
    ap.add_argument("ticket", nargs="*", help="ask once and exit; omit for interactive mode")
    ap.add_argument("--quiet", action="store_true", help="hide the tool calls, print the answer only")
    ap.add_argument("--trace", action="store_true", help="also write day13_trace.jsonl")
    args = ap.parse_args()

    global _VERBOSE
    _VERBOSE = not args.quiet

    agent = create_react_agent(model=ChatOpenAI(model=MODEL, temperature=0),
                               tools=TOOLS, prompt=SYSTEM, checkpointer=InMemorySaver())

    def ask(text):
        reset_trace()
        log("user_request", text=text)
        t0 = time.perf_counter()
        res = agent.invoke({"messages": [HumanMessage(content=text)]},
                           {"configurable": {"thread_id": "cli"},
                            "recursion_limit": RECURSION_LIMIT})
        msgs = res["messages"]
        in_tok = sum((m.usage_metadata or {}).get("input_tokens", 0)
                     for m in msgs if isinstance(m, AIMessage))
        out_tok = sum((m.usage_metadata or {}).get("output_tokens", 0)
                      for m in msgs if isinstance(m, AIMessage))
        answer = msgs[-1].content
        calls = [e["tool"] for e in TRACE if e["event"] == "tool_call"]

        print()
        # an even recursion limit returns this as a normal message, with no error raised
        if answer.startswith(TRUNCATED):
            print("  !! agent was cut short by recursion_limit - answer is not real")
        print(answer)
        print()
        dupes = len(calls) - len(set(calls))
        print(f"  {len(calls)} tool calls ({dupes} duplicate)  "
              f"{time.perf_counter()-t0:.1f}s  ${cost(in_tok, out_tok):.6f}")
        if args.trace:
            print("  trace:", save_trace())
        print()

    if args.ticket:
        ask(" ".join(args.ticket))
        return

    print("Day 13 service-desk agent. Known users: U-1042, U-2087, U-3311.")
    print("Describe a ticket. Ctrl-C or an empty line to quit.\n")
    while True:
        try:
            q = input("ticket> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not q:
            return
        try:
            ask(q)
        except Exception as e:
            print(f"  error: {type(e).__name__}: {e}\n")


if __name__ == "__main__":
    main()
