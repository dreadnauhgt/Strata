"""serve/test_reuse_trace.py - STRATA_REUSE_TRACE's line: where a prompt that reuses little starts to differ from
the request before it (no engine, no GPU).

    python -m unittest serve.test_reuse_trace -v
"""
from __future__ import annotations

import io
import os
import sys
import unittest
from array import array
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import reuse_trace  # noqa: E402
from serve.reuse_trace import ReuseTrace, common_prefix, report  # noqa: E402


def ids(n, start=0):
    return array("i", range(start, start + n))


class CommonPrefix(unittest.TestCase):
    def test_lengths_and_positions(self):
        a = ids(20000)
        self.assertEqual(common_prefix(a, a), 20000)
        self.assertEqual(common_prefix(a, ids(5)), 5)
        self.assertEqual(common_prefix(a, array("i")), 0)
        for at in (0, 1, 4095, 4096, 4097, 12345, 19999):   # around the chunk edges too
            b = array("i", a)
            b[at] = -1
            self.assertEqual(common_prefix(a, b), at, at)


class Report(unittest.TestCase):
    def test_quiet_cases(self):
        prev = ids(10000)
        self.assertIsNone(report(None, prev, 0))
        self.assertIsNone(report(array("i"), prev, 0))
        self.assertIsNone(report(prev, ids(12000), 9990))              # reused about all it shares
        short = array("i", prev)
        short[100] = -1
        self.assertIsNone(report(prev, short, 0))                       # shares only 100 tokens: cheap to read

    def test_extends_but_engine_lost_it(self):
        prev = ids(10000)
        line = report(prev, ids(12000), 0)
        self.assertIn("prompt 12000 tokens extends the previous prompt (10000 tokens) whole", line)
        self.assertIn("10000 tokens shared, the engine reused 0 (10000 read again)", line)

    def test_changed_at_shows_the_ids_there(self):
        prev = ids(10000)
        cur = ids(12000)
        cur[6000] = -7
        line = report(prev, cur, 0)
        self.assertIn("changed at token 6000 (previous prompt 10000 tokens)", line)
        self.assertIn("previous [5997, 5998, 5999, 6000, 6001, 6002, 6003]", line)
        self.assertIn("now [5997, 5998, 5999, -7, 6001, 6002, 6003]", line)
        self.assertIn("6000 tokens shared, the engine reused 0 (6000 read again)", line)


class Trace(unittest.TestCase):
    def test_keeps_the_previous_prompt_and_prints_once(self):
        out = io.StringIO()
        t = ReuseTrace(out)
        t.note(list(range(10000)), 0)            # the first request: nothing to compare with
        t.note(list(range(10500)), 9990)         # a normal turn: quiet
        t.note(list(range(11000)), 0)            # the loss
        lines = out.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertIn("extends the previous prompt (10500 tokens) whole", lines[0])

    def test_switch(self):
        for value, on in (("", False), ("0", False), ("1", True), ("yes", True)):
            with mock.patch.dict(os.environ, {"STRATA_REUSE_TRACE": value}):
                self.assertEqual(reuse_trace.enabled(), on, value)
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(reuse_trace.enabled())


class ThroughTheEngine(unittest.TestCase):
    """The hook in StrataEngine.generate: the engine's RESUME line, with a scripted engine process."""

    def bare(self):
        import queue
        from serve.server import StrataEngine
        engine = StrataEngine.__new__(StrataEngine)
        engine.proc = mock.Mock()
        engine.proc.stdin = io.StringIO()
        engine.proc.poll.return_value = None
        engine.lines, engine.can_stop, engine.max_context = queue.Queue(), False, 1 << 20
        engine.silence_s, engine.log_path = 5.0, None
        return engine

    def run_request(self, engine, prompt, reused):
        import threading
        for x in (f"RESUME {reused}", "T 7", f"DONE 1 {len(prompt)} 10 1 length"):
            engine.lines.put(x)
        return list(engine.generate(prompt, 1, {}, threading.Event()))

    def test_a_lost_prefix_is_named_and_the_default_says_nothing(self):
        first, second = list(range(9000)), list(range(9500))
        for value, expect in (("1", 1), ("", 0)):
            engine, err = self.bare(), io.StringIO()
            with mock.patch.dict(os.environ, {"STRATA_REUSE_TRACE": value}), mock.patch("sys.stderr", err):
                self.assertEqual(self.run_request(engine, first, 0), [7])
                self.assertEqual(self.run_request(engine, second, 0), [7])
            lines = [x for x in err.getvalue().splitlines() if x.startswith("strata reuse trace:")]
            self.assertEqual(len(lines), expect, value)
            if expect:
                self.assertIn("prompt 9500 tokens extends the previous prompt (9000 tokens) whole", lines[0])
            else:
                self.assertNotIn("reuse_trace", engine.__dict__)    # off: nothing kept


if __name__ == "__main__":
    unittest.main()
