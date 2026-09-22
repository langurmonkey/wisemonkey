"""Smoke test for the phase 2 UDS server/client.

Starts a server in a subprocess, attaches, pings, runs a slash command,
and shuts the server down. Run with:

    uv run python tests/smoke_server.py
"""

import subprocess
import sys
import time

from agent.client import ServerConnection
from agent.ipc import TransportClosed

session = "smoketest"

# Start the server as a normal subprocess (visible output)
proc = subprocess.Popen(
    [sys.executable, "-m", "agent.server", session],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)
try:
    conn = ServerConnection.connect(session=session, spawn=False, timeout=15)
    hs = conn.attach()
    print("attached:", hs.session, "pid:", hs.server_pid, "model:", hs.model)
    print("ping:", conn.ping())

    res = conn.command("/help")
    print("command ok:", res.ok, "| msg:", (res.msg or "")[:80])

    conn.shutdown()
    try:
        conn.transport.close()
    except TransportClosed:
        pass
    print("done")
finally:
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
    out, _ = proc.communicate(timeout=5)
    if out:
        print("--- server output ---")
        print(out[:2000])
