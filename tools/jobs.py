"""Background job execution: spawn, poll, kill.

A long command (a test suite, a build) does not need to block the turn. It
can be started, polled while other work happens, and killed when it is no
longer wanted. This is the alternative to `run_command`, whose `timeout` is
all-or-nothing: it either finishes inside the timeout or returns nothing at
all, and the caller has to guess how long to sleep before looking.

Design decisions worth naming:

* **Output is a file, not a pipe.** A subprocess pipe is a fixed buffer that
  blocks the child when nobody is reading it; a job that is meant to outlive
  the call that started it would deadlock against its own output. A file also
  survives the job, so a `kill` can still report what the command printed
  before it died.
* **Output is read incrementally.** `poll` returns only what appeared since
  the previous poll, so watching a slow build does not re-send the same lines
  every time and does not fill the context with duplicates.
* **Jobs live in the process.** A job is a child of this process and dies
  with it. Persisting jobs across restarts is a different and much larger
  feature; pretending to support it would leave orphans nobody can find.
* **The child gets its own process group.** `kill` signals the whole group,
  so a shell pipeline (`make | tee log`) does not leave the second half
  running after the first half is dead.
"""

from __future__ import annotations

import atexit
import os
import signal
import subprocess
import tempfile
import time

from agent.tools import tool
from agent.output import get_output_or_ipc
from agent.utils import contractuser

from tools.terminal import _is_dangerous, _prompt_user

# How much output a single poll returns. A build can print megabytes, and a
# poll that returns all of it defeats the purpose of polling: the cap is what
# makes "watch it while I work" affordable.
POLL_MAX_CHARS = 20000

# Jobs, keyed by id. Insertion-ordered, which is also oldest-first for
# listing.
_jobs: dict[str, "_Job"] = {}



class _Job:
    """One background command and the cursor into its output."""

    def __init__(self, job_id: str, command: str, popen: subprocess.Popen, path: str):
        self.id = job_id
        self.command = command
        self.popen = popen
        self.path = path
        self.started = time.time()
        # A *byte* offset, not a text-mode cookie: the log is read in binary so
        # that the offset can be compared against the file's size. A text-mode
        # `tell()` is an opaque number that means nothing next to `getsize`,
        # and with multi-byte output it is not even a character count.
        self.read_offset = 0

    def poll_status(self):
        """Reap the child if it has exited; return the exit code or None."""
        return self.popen.poll()

    def size(self) -> int:
        """How many bytes the job has written in total."""
        try:
            return os.path.getsize(self.path)
        except OSError:
            return self.read_offset

    def unread(self) -> int:
        """Bytes produced but not yet handed to a caller."""
        return max(0, self.size() - self.read_offset)

    def read_new(self, max_chars: int = POLL_MAX_CHARS) -> str:
        """Return up to *max_chars* characters written since the last read.

        Binary read, then decode: a decode error must not swallow the rest of
        the log, and the byte offset has to survive the round trip. A read
        that lands mid-character drops the partial one -- the alternative is a
        replacement character in the middle of the output, which reads like
        corruption in the command's own text.
        """
        try:
            with open(self.path, "rb") as fh:
                fh.seek(self.read_offset)
                raw = fh.read(max_chars * 4)
        except OSError:
            return ""
        if not raw:
            return ""
        self.read_offset += len(raw)
        text = raw.decode("utf-8", errors="replace")
        return text[:max_chars]

    def read_all(self, max_chars: int = POLL_MAX_CHARS) -> str:
        """Return the job's output from the beginning, in capped chunks.

        Unlike :meth:`read_new` this ignores the cursor, so it answers "what
        did this job print, all of it" even when a previous poll has already
        shown part of it. The cursor is left at the end, so a later
        incremental poll does not re-send what this returned.
        """
        try:
            self.read_offset = 0
            chunks = []
            while True:
                chunk = self.read_new(max_chars)
                if not chunk:
                    break
                chunks.append(chunk)
            return "".join(chunks)
        except OSError:
            return ""


def _new_id() -> str:
    return f"job{len(_jobs) + 1}"


# How many finished jobs to keep around. A job that has been polled to
# completion is still worth one more poll -- the common shape is "is it done?
# yes -- what did it say?", and a registry that forgets it the moment it exits
# answers the second question with "no such job". Beyond this many, the
# oldest fully-read finished jobs are dropped and their logs deleted.
MAX_FINISHED_JOBS = 8


def _reap_finished() -> None:
    """Delete the log of finished jobs that are no longer worth keeping.

    Every spawn and poll calls this. Without it, a loop of spawn/poll cycles
    leaks one temp file and one process entry per job for the lifetime of the
    session.
    """
    finished = [
        j for j in _jobs.values()
        if j.poll_status() is not None and _is_fully_read(j)
    ]
    # Insertion order, so the tail is the most recently finished.
    excess = max(0, len(finished) - MAX_FINISHED_JOBS)
    for job in finished[:excess]:
        try:
            os.unlink(job.path)
        except OSError:
            pass
        _jobs.pop(job.id, None)


def _is_fully_read(job: _Job) -> bool:
    """Whether the caller has been shown everything *job* printed."""
    return job.unread() == 0


def _job_result(job: _Job, max_chars: int = POLL_MAX_CHARS, include_all: bool = False):
    """The result dict for *job*, reading its unread output.

    `output_bytes` is what the job has written *in total*, not what this poll
    returned. That distinction is the whole reason the field exists: a poll of
    a chatty job returns a truncated tail, and without a total the caller
    cannot tell a job that printed three lines from one that printed three
    hundred thousand. `omitted_bytes` is the part still unread and available
    to the next poll.
    """
    exit_code = job.poll_status()
    if include_all:
        text = job.read_all(max_chars)
        omitted = 0
    else:
        text = job.read_new(max_chars)
        omitted = job.unread()

    elapsed = time.time() - job.started
    return {
        "job": job.id,
        "command": job.command,
        "running": exit_code is None,
        "exit_code": exit_code,
        "elapsed_s": round(elapsed, 1),
        "output": text,
        "output_bytes": job.size(),
        "omitted_bytes": omitted,
    }


@tool(
    name="spawn",
    description=(
        "Start a shell command in the background and return immediately with a "
        "job id. Use this for long work (a test suite, a build, a long "
        "download) that you want to poll with 'poll_job' while doing "
        "something else, instead of blocking 'run_command' on its timeout.\n"
        "\n"
        "Use 'run_command' for anything short: it is simpler and returns the "
        "output directly. Reach for this only when the command is expected to "
        "outlast the turn. Dangerous commands prompt the user first, exactly "
        "as with 'run_command'. The command runs in its own process group, so "
        "'kill_job' takes its children with it, and it is killed when the "
        "agent exits -- a job cannot outlive the session."
    ),
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to run in the background.",
            },
            "_skip_confirmation": {
                "type": "boolean",
                "description": "Internal flag — do not use directly",
            },
        },
        "required": ["command"],
    },
)
def spawn_handler(args):
    """Start *command* in the background; return its job id."""
    command = args.get("command", "")
    skip_confirmation = args.get("_skip_confirmation", False)
    output = get_output_or_ipc()

    if not command:
        return {"error": "No command provided"}

    from agent.config import get_config

    if get_config().get("agent.unsafe", False):
        skip_confirmation = True

    if not skip_confirmation:
        if _is_dangerous(command):
            if not _prompt_user(command, "Command matches dangerous patterns"):
                return {
                    "error": (
                        "Command execution was cancelled by the user. The user "
                        "declined to run this command. Ask what they would "
                        "like to do instead."
                    ),
                    "user_cancelled": True,
                    "command": command,
                }
        else:
            output.print(f"[path][bold]$[/bold] (background) {command}[/path]", indent=2)

    _reap_finished()

    handle, path = tempfile.mkstemp(prefix="wisemonkey-job-", suffix=".log")
    os.close(handle)

    try:
        with open(path, "wb") as sink:
            popen = subprocess.Popen(
                command,
                shell=True,
                stdout=sink,
                stderr=subprocess.STDOUT,
                # Its own process group, so kill can signal the whole pipeline
                # rather than just the shell that started it.
                start_new_session=True,
            )
    except Exception as exc:
        try:
            os.unlink(path)
        except OSError:
            pass
        return {"error": str(exc)}

    job_id = _new_id()
    while job_id in _jobs:  # pragma: no cover - ids are unique by construction
        job_id = f"job{len(_jobs) + 1}-{time.time_ns()}"
    job = _Job(job_id, command, popen, path)
    _jobs[job_id] = job

    return {
        "job": job_id,
        "command": command,
        "pid": popen.pid,
        "running": True,
        "log": contractuser(path),
        "message": (
            f"Started as {job_id} (pid {popen.pid}). Poll with "
            f"poll_job(job='{job_id}'); output accumulates in {contractuser(path)}."
        ),
    }


@tool(
    name="poll_job",
    description=(
        "Check a background job started with 'spawn'. Returns 'running' "
        "(True/False), the 'exit_code' once it has finished, and the output "
        "produced since the previous poll -- not the whole log, so polling a "
        "chatty command repeatedly does not re-send the same lines. Output is "
        "capped per poll; 'omitted_bytes' says how much was left behind if "
        "the command outran it.\n"
        "\n"
        "Omit 'job' to list every job with its status, which is how you "
        "recover a job id you did not keep."
    ),
    parameters={
        "type": "object",
        "properties": {
            "job": {
                "type": "string",
                "description": "Job id returned by 'spawn'. Omit to list all jobs.",
            },
            "all_output": {
                "type": "boolean",
                "description": "If True, return the job's entire output so far instead of only what is new since the last poll. Default: False.",
            },
        },
        "required": [],
    },
)
def poll_handler(args):
    """Report on one job, or list them all."""
    output = get_output_or_ipc()
    _reap_finished()

    job_id = args.get("job", "") or ""
    include_all = bool(args.get("all_output", False))

    if not job_id:
        if not _jobs:
            return {"jobs": [], "count": 0, "message": "No background jobs."}
        listing = [
            {
                "job": j.id,
                "command": j.command[:80],
                "running": j.poll_status() is None,
                "elapsed_s": round(time.time() - j.started, 1),
            }
            for j in _jobs.values()
        ]
        return {"jobs": listing, "count": len(listing)}

    job = _jobs.get(job_id)
    if job is None:
        return {
            "error": (
                f"No such job: {job_id}. It may have finished and been reaped, "
                "in which case its output is in the log file given by 'spawn'. "
                "Call poll_job with no 'job' to list what is still running."
            ),
            "job": job_id,
        }

    result = _job_result(job, include_all=include_all)
    if result["running"]:
        output.print(
            f"[weak]Job[/weak] [path]{job_id}[/path] [weak]still running "
            f"({result['elapsed_s']}s)[/weak]",
            indent=2,
        )
    else:
        output.print(
            f"[weak]Job[/weak] [path]{job_id}[/path] [weak]finished, exit "
            f"{result['exit_code']}[/weak]",
            indent=2,
        )
    if result["omitted_bytes"]:
        result["message"] = (
            f"This poll was capped at {POLL_MAX_CHARS} characters; "
            f"{result['omitted_bytes']} byte(s) of output are still unread. "
            f"Poll again, or pass all_output=True (or read {contractuser(job.path)})."
        )
    return result


@tool(
    name="kill_job",
    description=(
        "Stop a background job started with 'spawn'. Sends SIGTERM to the "
        "job's whole process group, so shell pipelines and child processes go "
        "with it, then reports any output the command produced before dying -- "
        "a killed test run still says which tests failed. Pass "
        "'force': true to send SIGKILL instead, for a command that ignores "
        "SIGTERM. Killing an already-finished job returns its exit code."
    ),
    parameters={
        "type": "object",
        "properties": {
            "job": {
                "type": "string",
                "description": "Job id returned by 'spawn'.",
            },
            "force": {
                "type": "boolean",
                "description": "If True, use SIGKILL instead of SIGTERM. Default: False.",
            },
        },
        "required": ["job"],
    },
)
def kill_handler(args):
    """Terminate *job*'s process group and report what it last printed."""
    job_id = args.get("job", "")
    force = bool(args.get("force", False))
    output = get_output_or_ipc()
    _reap_finished()

    job = _jobs.get(job_id)
    if job is None:
        return {
            "error": (
                f"No such job: {job_id}. It may have already finished and been "
                "reaped; call poll_job with no 'job' to list what remains."
            ),
            "job": job_id,
        }

    already = job.poll_status()
    if already is not None:
        result = _job_result(job, include_all=True)
        result["killed"] = False
        result["message"] = f"Job {job_id} had already finished (exit {already})."
        return result

    sig = signal.SIGKILL if force else signal.SIGTERM
    try:
        os.killpg(os.getpgid(job.popen.pid), sig)
    except (ProcessLookupError, PermissionError) as exc:
        return {"error": f"Could not signal job {job_id}: {exc}", "job": job_id}

    # Give it a moment to die so the exit code is meaningful, then stop
    # insisting. A process that ignores SIGTERM is exactly why 'force' exists.
    deadline = time.time() + (2.0 if force else 5.0)
    while time.time() < deadline and job.poll_status() is None:
        time.sleep(0.05)

    result = _job_result(job, include_all=True)
    result["killed"] = result["running"] is False
    if result["running"]:
        result["message"] = (
            f"Job {job_id} did not exit within "
            f"{'0' if force else '5'}s of SIG{'KILL' if force else 'TERM'}. "
            "It is still running; call kill_job again with force: true."
        )
    else:
        output.print(
            f"[weak]Killed[/weak] [path]{job_id}[/path] "
            f"[weak](exit {result['exit_code']})[/weak]",
            indent=2,
        )
    return result


def _kill_all() -> None:
    """Kill every running job at exit.

    A job is a child of this process. Without this, exiting the agent would
    leave orphaned builds and test runs holding CPU, writing to temp files,
    and never finishing.
    """
    for job in list(_jobs.values()):
        if job.poll_status() is not None:
            continue
        try:
            os.killpg(os.getpgid(job.popen.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            job.popen.wait(timeout=1)
        except Exception:
            try:
                os.killpg(os.getpgid(job.popen.pid), signal.SIGKILL)
            except Exception:
                pass
        try:
            os.unlink(job.path)
        except OSError:
            pass
    _jobs.clear()


atexit.register(_kill_all)