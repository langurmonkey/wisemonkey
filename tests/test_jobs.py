"""Tests for tools/jobs.py -- spawn / poll_job / kill_job.

The properties worth pinning are the ones a caller relies on without noticing:
output arrives incrementally (so polling a chatty job is cheap), a job that
has exited can still be polled for its result, and `kill` takes the whole
process group rather than just the shell that started it.

Real subprocesses are used deliberately. A fake would let the test pass while
the process group, the file-backed output, and the reaping were all wrong --
which is precisely where the bugs in this module live.
"""

import os
import subprocess
import time
from typing import cast

from tests.conftest import BaseTest
from agent.tools import discover_tools, get_registry
from agent.output import set_output, OutputAdapter


class _FakeOutput:
    def print(self, *args, **kwargs):
        pass

    def err(self, *args, **kwargs):
        pass

    def ok(self, *args, **kwargs):
        pass

    def newline(self, *args, **kwargs):
        pass

    def ask_confirm(self, *args, **kwargs):
        raise AssertionError("a background job must not prompt in these tests")


def _pids_matching(marker: str) -> list[str]:
    """Pids whose command line contains *marker*, excluding this process.

    `pgrep -f` matches its own command line when the marker is an argument to
    it, and the test runner's own shell holds the script text, so both are
    filtered out to leave only real job descendants.
    """
    out = subprocess.run(
        ["pgrep", "-f", marker], capture_output=True, text=True
    ).stdout.split()
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

        # Every job started here must be gone when the test ends, or it
        # outlives the assertion that was supposed to be checking it.
        from tools.jobs import _kill_all

        self.addCleanup(_kill_all)

    def _start(self, command):
        result = self.spawn(
            {"command": command, "_skip_confirmation": True}
        )
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

    # --- spawn ---

    def test_spawn_returns_a_job_id_and_does_not_wait(self):
        job = self._start("sleep 5")
        assert job
        # If spawn blocked, this test would take five seconds.
        assert self.poll({"job": job})["running"] is True

    def test_spawn_needs_a_command(self):
        assert "error" in self.spawn({})

    def test_spawn_returns_immediately_even_for_a_fast_command(self):
        job = self._start("true")
        assert self._wait(job)["exit_code"] == 0

    # --- poll ---

    def test_poll_reports_the_exit_code(self):
        assert self._wait(self._start("exit 3"))["exit_code"] == 3

    def test_poll_reports_a_zero_exit_distinctly_from_a_failure(self):
        assert self._wait(self._start("true"))["exit_code"] == 0

    def test_stderr_is_captured_too(self):
        """A test run that fails says so on stderr; losing that is blindness."""
        job = self._start("echo out; echo err >&2")
        result = self._wait(job)
        assert "out" in result["output"]
        assert "err" in result["output"]

    def test_poll_returns_only_what_is_new(self):
        """Otherwise every poll re-sends the whole log and fills the context."""
        job = self._start("echo one; sleep 0.4; echo two")
        # The shell has not written anything yet if we poll immediately, so
        # give it a moment: this test is about the second half of the output
        # not repeating the first, not about how fast a shell starts.
        time.sleep(0.2)
        first = self.poll({"job": job})
        assert "one" in first["output"]
        assert "two" not in first["output"]
        result = self._wait(job)
        assert "two" in result["output"]
        # The earlier line is not repeated in the later poll.
        assert "one" not in result["output"]

    def test_poll_after_the_fact_still_returns_the_result(self):
        """The common shape is: is it done? yes -- what did it say?"""
        job = self._start("sleep 0.2; echo late")
        self._wait(job)
        again = self.poll({"job": job})
        assert again["running"] is False
        assert again["exit_code"] == 0
        # Output was already read by the wait; the exit status must still be
        # there, not a "no such job".
        assert again["job"] == job

    def test_all_output_rereads_from_the_start(self):
        job = self._start("sleep 0.2; echo a; echo b")
        self._wait(job)
        full = self.poll({"job": job, "all_output": True})
        assert "a" in full["output"] and "b" in full["output"]

    def test_unknown_job_is_an_error_that_says_what_to_do(self):
        result = self.poll({"job": "job999"})
        assert "error" in result
        assert "poll_job" in result["error"]

    def test_listing_jobs(self):
        listing = self.poll({})
        assert "jobs" in listing
        assert isinstance(listing["jobs"], list)

    # --- kill ---

    def test_kill_stops_a_running_job(self):
        job = self._start("sleep 30")
        time.sleep(0.2)
        result = self.kill({"job": job})
        assert result["running"] is False
        assert result["killed"] is True

    def test_kill_returns_the_output_produced_before_dying(self):
        """A killed test run must still say which test failed."""
        job = self._start("echo FAIL: test_foo; sleep 30")
        time.sleep(0.3)
        result = self.kill({"job": job})
        assert "FAIL: test_foo" in result["output"]

    def test_kill_takes_the_whole_process_group(self):
        """A pipeline's second half must not survive its first half.

        `kill` signals the process group, so `(sleep; work) | tee log` dies
        completely. Signalling only the shell would leave the sleep running,
        holding CPU with nobody watching it.
        """
        marker = f"wmkjobgroup{os.getpid()}"
        job = self._start(f"(sleep 30; echo {marker}) | tee /dev/null")
        time.sleep(0.4)
        assert _pids_matching(marker), "the job's child never started"
        self.kill({"job": job})
        time.sleep(0.5)
        assert not _pids_matching(marker), "kill left a child of the job running"

    def test_killing_a_finished_job_reports_its_exit_code(self):
        job = self._start("exit 7")
        self._wait(job)
        result = self.kill({"job": job})
        assert result["killed"] is False
        assert result["exit_code"] == 7

    def test_kill_of_an_unknown_job_is_an_error(self):
        assert "error" in self.kill({"job": "job999"})

    def test_force_kill_works(self):
        """A command that traps SIGTERM is exactly why 'force' exists."""
        job = self._start("trap '' TERM; sleep 30")
        time.sleep(0.3)
        gentle = self.kill({"job": job})
        if gentle.get("running"):
            # It ignored the first signal; the tool must say so rather than
            # claim success.
            assert "force" in gentle.get("message", "")
            hard = self.kill({"job": job, "force": True})
            assert hard["running"] is False
        else:
            assert gentle["running"] is False

    # --- lifecycle ---

    def test_exit_hook_kills_everything(self):
        """A job must not outlive the agent that started it."""
        from tools.jobs import _kill_all

        marker = f"wmkatexit{os.getpid()}"
        # The marker has to be inside the child's argument vector, not in a
        # shell comment: `sh -c "sleep 30 # x"` puts "# x" in argv, but not
        # every shell keeps the comment text in the command line pgrep sees.
        self.spawn(
            {
                "command": f'python3 -c "import time; time.sleep(30)  # {marker}"',
                "_skip_confirmation": True,
            }
        )
        time.sleep(0.5)
        assert _pids_matching(marker), "the job never started"
        _kill_all()
        time.sleep(0.5)
        assert not _pids_matching(marker)

    def test_a_finished_job_is_reaped_but_a_recent_one_is_not(self):
        """Enough polls to leak a temp file per job is worse than one stale entry."""
        from tools.jobs import MAX_FINISHED_JOBS, _jobs

        for _ in range(MAX_FINISHED_JOBS + 2):
            job = self._start("true")
            self._wait(job)

        assert len(_jobs) <= MAX_FINISHED_JOBS + 1

    def test_a_noisy_job_does_not_bury_the_poll(self):
        """The per-poll cap is what makes polling an affordable thing to do."""
        from tools.jobs import POLL_MAX_CHARS

        job = self._start(
            f"python3 -c \"print('x' * {POLL_MAX_CHARS * 3})\""
        )
        result = self._wait(job)
        assert len(result["output"]) <= POLL_MAX_CHARS
        # And it says the rest exists rather than implying that was all.
        assert result["output_bytes"] > len(result["output"])