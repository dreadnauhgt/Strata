"""STRATA_REUSE_TRACE=1 (opt-in, off by default): one log line when the engine reuses much less of a prompt than the
prompt shares with the request before it.

The engine reuses a prompt from its live session or a checkpoint whose tokens are exactly the prompt's start. When a
long conversation suddenly reuses 0 tokens, the engine's own line does not say which side changed. This line does:

  * "changed at token X": this prompt and the previous one differ from token X on. The client, the template or the
    encoding produced different tokens there (earlier text re-rendered, a literal tag, a sampling key in the prompt).
  * "extends the previous prompt": the two agree up to X (all of the previous prompt), yet the engine resumed far
    before X. The engine had no checkpoint at or before X that still matched: a retention or eviction question.

The ids around X are printed (no text) so a client's rendering can be compared offline. Only token ids of the
previous prompt are kept, in one compact array; with the variable unset nothing is kept and nothing is printed.
"""
from __future__ import annotations

import os
import sys
from array import array

MIN_SHARED = 4096          # below this a lost prefix costs little: not reported
CHUNK = 4096               # common-prefix search compares slices this long before narrowing down


def enabled() -> bool:
    v = os.environ.get("STRATA_REUSE_TRACE", "")
    return v not in ("", "0")


def common_prefix(a: array, b: array) -> int:
    """The number of leading ids two arrays share."""
    n = min(len(a), len(b))
    i = 0
    while i + CHUNK <= n and a[i:i + CHUNK] == b[i:i + CHUNK]:
        i += CHUNK
    while i < n and a[i] == b[i]:
        i += 1
    return i


def report(prev: array | None, ids: array, reused: int, min_shared: int = MIN_SHARED) -> str | None:
    """The line for this request, or None when there is nothing worth saying (no previous prompt, a short shared
    prefix, or the engine reused about as much as the two prompts share)."""
    if prev is None or not prev:
        return None
    shared = common_prefix(prev, ids)
    if shared < min_shared or reused + min_shared > shared:
        return None
    if shared >= len(prev):
        what = f"extends the previous prompt ({len(prev)} tokens) whole"
    else:
        what = (f"changed at token {shared} (previous prompt {len(prev)} tokens); ids there: previous "
                f"{list(prev[max(0, shared - 3):shared + 4])}, now {list(ids[max(0, shared - 3):shared + 4])}")
    return (f"strata reuse trace: prompt {len(ids)} tokens {what}; {shared} tokens shared, the engine reused {reused} "
            f"({shared - reused} read again)")


class ReuseTrace:
    """Holds the previous prompt's ids for one engine (single slot)."""

    def __init__(self, out=None):
        self.prev: array | None = None
        self.out = out if out is not None else sys.stderr

    def note(self, ids, reused: int) -> None:
        cur = array("i", ids)
        line = report(self.prev, cur, int(reused))
        if line:
            print(line, file=self.out, flush=True)
        self.prev = cur
