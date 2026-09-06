#!/usr/bin/env python3
"""Focused tests for the two Python answers in quality-mtp5-nonthinking.json."""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESPONSES = (
    HERE
    / "artifacts"
    / "2026-09-06-vast-rtx5090"
    / "qwen38-results"
    / "quality-mtp5-nonthinking.json"
)
items = {item["name"]: item for item in json.loads(RESPONSES.read_text())}


def generated_code(name: str) -> str:
    text = items[name]["response"]["choices"][0]["message"]["content"]
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return (match.group(1) if match else text).strip()


namespace: dict = {}
exec(generated_code("python_intervals"), namespace)
merge_intervals = namespace["merge_intervals"]
original = [(5, 1), (2, 3), (3, 4), (8, 8), (8, 10), (20, 21)]
snapshot = original.copy()
assert merge_intervals(original) == [(1, 5), (8, 10), (20, 21)]
assert original == snapshot
assert merge_intervals([]) == []
assert merge_intervals([(1, 1), (1, 1)]) == [(1, 1)]
print("python_intervals: PASS")

namespace = {}
exec(generated_code("python_async"), namespace)
async_map_limited = namespace["async_map_limited"]


async def test_async_map() -> None:
    active = 0
    peak = 0
    lock = asyncio.Lock()

    async def square(value: int) -> int:
        nonlocal active, peak
        async with lock:
            active += 1
            peak = max(peak, active)
        await asyncio.sleep(0.005 * (5 - value % 5))
        async with lock:
            active -= 1
        return value * value

    assert await async_map_limited(range(20), 3, square) == [x * x for x in range(20)]
    assert peak <= 3
    assert await async_map_limited([], 2, square) == []

    try:
        await async_map_limited([1], 0, square)
    except ValueError:
        pass
    else:
        raise AssertionError("limit=0 did not raise ValueError")

    running = 0

    async def fail_on_two(value: int) -> None:
        nonlocal running
        running += 1
        try:
            if value == 2:
                await asyncio.sleep(0.001)
                raise RuntimeError("boom")
            await asyncio.sleep(1)
        finally:
            running -= 1

    try:
        await async_map_limited(range(10), 3, fail_on_two)
    except RuntimeError as error:
        assert str(error) == "boom"
    else:
        raise AssertionError("worker failure did not propagate")
    assert running == 0


asyncio.run(test_async_map())
print("python_async: PASS")
