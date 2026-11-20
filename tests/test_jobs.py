"""Tests for tools/jobs.py -- spawn / poll_job / kill_job."""

import os
import subprocess
import time
from typing import cast
from unittest.mock import patch

from agent.output import OutputAdapter, set_output
from agent.tools import discover_tools, get_registry
from tests.conftest import BaseTest


class _FakeOutput:
    def print(self, *a, **k):
        pass

    def err(self, *a, **k):
        pass

    def ok(self, *a, **k):
        pass

    def newline(self, *a, **k):
        pass

    def ask_confirm(self, *a, **k):
        raise AssertionError("a background job must not prompt in these tests")


def _pids_matching(marker: str) -> list[str]:
    """Pids whose command line contains *marker*, excluding this process."""
    out = subprocess.run(["pgrep", "-f", marker], capture_output=True,
                         text=True).stdout.split()
    mine = {str(os.getpid()), str(os.getppid())}
    return [p for p in out if p not in mine]


class TestJobs(BaseTest):
    def setUp(self):
        super().setUp()
        discover_tools()
        registry = get_registry()
        self.spawn = registry["spawn"]["handler"]
        self.poll = registry["poll_job"]["handler"]
        self.kill = registry["kill_job"]["handler"]
        set_output(cast(OutputAdapter, _FakeOutput()))
        self.addCleanup(set_output, None)
        # No job may outlive the test that started it.
        from tools.jobs import _kill_all
        self.addCleanup(_kill_all)

    def _start(self, command):
        result = self.spawn({"command": command, "_skip_confirmation": True})
        assert "job" in result, result
        self.addCleanup(self.kill, {"job": result["job"]})
        return result["job"]

    def _wait(self, job, timeout=15.0):
        """Poll until *job* exits; return the last poll result."""
        deadline = time.time() + timeout
        result = self.poll({"job": job})
        while result["running"] and time.time() < deadline:
            time.sleep(0.05)
            result = self.poll({"job": job})
        return result


class TestSpawn(TestJobs):
    def test_returns_immediately_with_the_session_temp_dir_as_TMPDIR(self):
        job = self._start("sleep 5")
        assert self.poll({"job": job})["running"] is True
        assert "error" in self.spawn({})

        temp = self._tmpdir / "session-tmp"
        with patch("agent.memory.current_session_temp_dir", return_value=temp):
            result = self._wait(self._start('printf "%s" "$TMPDIR"'))
        assert (result["exit_code"], result["output"]) == (0, str(temp))
        assert temp.is_dir()


class TestPoll(TestJobs):
    def test_exit_code_and_stderr_are_reported(self):
        assert self._wait(self._start("exit 3"))["exit_code"] == 3
        result = self._wait(self._start("echo out; echo err >&2"))
        assert "out" in result["output"] and "err" in result["output"]

    def test_only_new_output_is_returned_but_the_job_stays_pollable(self):
        job = self._start("echo one; sleep 0.4; echo two")
        time.sleep(0.2)
        first = self.poll({"job": job})
        assert "one" in first["output"] and "two" not in first["output"]
        result = self._wait(job)
        assert "two" in result["output"] and "one" not in result["output"]
        # The common shape: is it done? yes -- what did it say?
        again = self.poll({"job": job})
        assert (again["job"], again["running"]) == (job, False)
        full = self.poll({"job": job, "all_output": True})
        assert "one" in full["output"] and "two" in full["output"]

    def test_listing_an_unknown_job_and_the_output_cap(self):
        from tools.jobs import POLL_MAX_CHARS
        assert "error" in self.poll({"job": "job999"})
        assert isinstance(self.poll({})["jobs"], list)
        job = self._start(f"python3 -c \"print('x' * {POLL_MAX_CHARS * 3})\"")
        result = self._wait(job)
        # A truncated tail must not read like a complete one.
        assert len(result["output"]) <= POLL_MAX_CHARS
        assert result["output_bytes"] > len(result["output"])


class TestKill(TestJobs):
    def test_kill_stops_the_whole_process_group_and_keeps_the_output(self):
        job = self._start("echo FAIL: test_foo; sleep 30")
        time.sleep(0.3)
        result = self.kill({"job": job})
        assert (result["running"], result["killed"]) == (False, True)
        assert "FAIL: test_foo" in result["output"]

        marker = f"wmkjobgroup{os.getpid()}"
        job = self._start(f"(sleep 30; echo {marker}) | tee /dev/null")
        time.sleep(0.4)
        assert _pids_matching(marker), "the job's child never started"
        self.kill({"job": job})
        time.sleep(0.5)
        assert not _pids_matching(marker), "kill left a child running"

    def test_a_finished_job_reports_its_code_and_a_stubborn_one_is_forced(self):
        job = self._start("exit 7")
        self._wait(job)
        result = self.kill({"job": job})
        assert (result["killed"], result["exit_code"]) == (False, 7)
        assert "error" in self.kill({"job": "job999"})

        job = self._start("trap '' TERM; sleep 30")
        time.sleep(0.3)
        gentle = self.kill({"job": job})
        if gentle.get("running"):
            assert "force" in gentle.get("message", "")
            assert self.kill({"job": job, "force": True})["running"] is False


class TestLifecycle(TestJobs):
    def test_the_exit_hook_kills_every_job_and_finished_jobs_are_reaped(self):
        from tools.jobs import MAX_FINISHED_JOBS, _jobs, _kill_all
        marker = f"wmkatexit{os.getpid()}"
        # The marker must be in the child's argv: not every shell keeps a
        # comment in the command line pgrep sees.
        self.spawn({"command": f'python3 -c "import time; time.sleep(30)  # {marker}"',
                    "_skip_confirmation": True})
        time.sleep(0.5)
        assert _pids_matching(marker), "the job never started"
        _kill_all()
        time.sleep(0.5)
        assert not _pids_matching(marker)

        for _ in range(MAX_FINISHED_JOBS + 2):
            self._wait(self._start("true"))
        assert len(_jobs) <= MAX_FINISHED_JOBS + 1
