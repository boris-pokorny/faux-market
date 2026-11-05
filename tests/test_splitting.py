import random
import statistics
from collections.abc import Iterator
from typing import Any

import pytest

import faux_market as fm


class Limits:
    """Emits only bid limit orders."""

    def warm_up(self, ref: int, rng: random.Random) -> Iterator[fm.Limit]:
        yield fm.Limit(fm.Side.ASK, ref + 1, 1)

    def step(
        self, book: fm.BookView, rng: random.Random
    ) -> tuple[float, fm.Intent]:
        return 1.0, fm.Limit(fm.Side.BID, 90, 1)


class Sevens:
    """Deep book, then market orders of size 7 forever."""

    def warm_up(self, ref: int, rng: random.Random) -> Iterator[fm.Limit]:
        yield fm.Limit(fm.Side.ASK, ref + 1, 10**6)
        yield fm.Limit(fm.Side.BID, ref - 1, 10**6)

    def step(
        self, book: fm.BookView, rng: random.Random
    ) -> tuple[float, fm.Intent]:
        return 1.0, fm.Market(fm.Side.BID, 7)


def signs(flow: fm.OrderFlow, n: int, seed: int) -> list[float]:
    """Sign of each market order, grouping fills by timestamp."""
    out: list[float] = []
    last_ts = -1.0
    for ev in fm.Simulation(flow, seed=seed).run(n):
        if ev.action is fm.Action.EXECUTE and ev.ts != last_ts:
            out.append(1.0 if ev.side is fm.Side.ASK else -1.0)
            last_ts = ev.ts
    return out


def lag1(x: list[float]) -> float:
    m = statistics.fmean(x)
    var = sum((a - m) ** 2 for a in x)
    return sum((x[i] - m) * (x[i + 1] - m) for i in range(len(x) - 1)) / var


@pytest.mark.parametrize(
    "kwargs",
    [{"n_traders": 0}, {"alpha": 1.0}, {"min_size": 0}, {"max_meta": 0}],
)
def test_rejects_bad_config(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        fm.OrderSplitting(**kwargs)


def test_deterministic() -> None:
    def run() -> list[fm.L3Event]:
        return list(fm.simulate(500, order_flow=fm.OrderSplitting(), seed=4))

    assert run() == run()


def test_limits_pass_through() -> None:
    inner = list(fm.Simulation(Limits(), seed=1).run(20))
    wrapped = list(fm.Simulation(fm.OrderSplitting(Limits()), seed=1).run(20))
    assert inner == wrapped


def test_child_size_bounded_by_inner_and_metaorder() -> None:
    flow = fm.OrderSplitting(Sevens(), n_traders=1, min_size=3, max_meta=10)
    sim = fm.Simulation(flow, seed=2)
    per_ts: dict[float, int] = {}
    for ev in sim.run(200):
        if ev.action is fm.Action.EXECUTE:
            per_ts[ev.ts] = per_ts.get(ev.ts, 0) + ev.size
    sizes = list(per_ts.values())
    assert sizes and max(sizes) <= 7
    assert min(sizes) >= 1


def test_single_trader_makes_runs() -> None:
    s = signs(fm.OrderSplitting(n_traders=1), 20_000, 5)
    same = sum(a == b for a, b in zip(s, s[1:], strict=False)) / (len(s) - 1)
    assert same > 0.8


def test_sign_memory_exceeds_zero_intelligence() -> None:
    zi = signs(fm.ZeroIntelligence(), 30_000, 6)
    split = signs(fm.OrderSplitting(), 30_000, 6)
    assert lag1(split) > lag1(zi) + 0.1
