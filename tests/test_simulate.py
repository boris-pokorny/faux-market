import random
import statistics
from collections.abc import Iterator
from typing import Any

import pytest

import faux_market as fm


def test_count() -> None:
    assert sum(1 for _ in fm.simulate(250, seed=1)) == 250


def test_deterministic() -> None:
    assert list(fm.simulate(100, seed=7)) == list(fm.simulate(100, seed=7))


def test_event_invariants() -> None:
    live: dict[int, int] = {}
    last_ts = 0.0
    for ev in fm.simulate(5000, seed=3):
        assert ev.ts >= last_ts
        assert ev.price > 0 and ev.size > 0
        if ev.action is fm.Action.ADD:
            assert ev.order_id not in live
            live[ev.order_id] = ev.size
        elif ev.action is fm.Action.CANCEL:
            assert live.pop(ev.order_id) == ev.size
        else:
            live[ev.order_id] -= ev.size
            assert live[ev.order_id] >= 0
        last_ts = ev.ts


def test_book_never_crosses_and_mid_moves() -> None:
    sim = fm.Simulation(fm.ZeroIntelligence(), seed=11)
    mids = []
    for _ in sim.run(5000):
        spread = sim.book.spread
        assert spread is None or spread > 0
        if sim.book.mid is not None:
            mids.append(sim.book.mid)
    assert statistics.pstdev(mids) > 0


def test_spread_widens_with_market_rate() -> None:
    def mean_spread(market_rate: float) -> float:
        order_flow = fm.ZeroIntelligence(market_rate=market_rate)
        sim = fm.Simulation(order_flow, seed=5)
        spreads = [s for _ in sim.run(20000) if (s := sim.book.spread)]
        return statistics.fmean(spreads)

    assert mean_spread(4.0) > mean_spread(0.5)


class PingPong:
    """Alternates one bid add and one market sell that consumes it."""

    def warm_up(self, ref: int, rng: random.Random) -> Iterator[fm.Limit]:
        yield fm.Limit(fm.Side.ASK, ref + 1, 1)

    def step(
        self, book: fm.BookView, rng: random.Random
    ) -> tuple[float, fm.Intent]:
        if book.best_bid is None:
            return 1.0, fm.Limit(fm.Side.BID, 99, 3)
        return 1.0, fm.Market(fm.Side.ASK, 2)


def test_custom_model() -> None:
    events = list(fm.Simulation(PingPong(), tick=1.0).run(3))
    assert [e.action for e in events] == [
        fm.Action.ADD,
        fm.Action.ADD,
        fm.Action.EXECUTE,
    ]
    execute = fm.L3Event(2.0, 2, fm.Side.BID, fm.Action.EXECUTE, 99.0, 2)
    assert events[2] == execute


def test_run_resumes_without_dropping_events() -> None:
    sim = fm.Simulation(fm.ZeroIntelligence(), seed=3)
    piecewise = [ev for _ in range(300) for ev in sim.run(1)]
    assert piecewise == list(fm.simulate(300, seed=3))


def test_iter_is_the_same_stream_as_run() -> None:
    sim = fm.Simulation(fm.ZeroIntelligence(), seed=3)
    first = next(iter(sim))
    rest = list(sim.run(2))
    assert [first, *rest] == list(fm.simulate(3, seed=3))


def test_warm_up_is_lazy() -> None:
    sim = fm.Simulation(fm.ZeroIntelligence(window=2), seed=0)
    assert len(sim.book) == 0
    next(iter(sim))
    assert len(sim.book) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"add_rate": -1.0},
        {"cancel_rate": -0.1},
        {"add_rate": 0.0, "market_rate": 0.0},
        {"window": 0},
        {"max_size": 0},
    ],
)
def test_zero_intelligence_rejects_bad_config(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        fm.ZeroIntelligence(**kwargs)


def test_simulation_rejects_bad_prices() -> None:
    with pytest.raises(ValueError):
        fm.Simulation(fm.ZeroIntelligence(), tick=0.0)
    with pytest.raises(ValueError):
        fm.Simulation(fm.ZeroIntelligence(), mid=-1.0)


def test_intents_reject_non_positive_values() -> None:
    with pytest.raises(ValueError):
        fm.Limit(fm.Side.BID, 0, 1)
    with pytest.raises(ValueError):
        fm.Limit(fm.Side.BID, 1, 0)
    with pytest.raises(ValueError):
        fm.Market(fm.Side.ASK, 0)


def test_simulate_passes_through_order_flow_and_prices() -> None:
    def events() -> list[fm.L3Event]:
        order_flow = fm.ZeroIntelligence(window=1)
        return list(
            fm.simulate(2, order_flow=order_flow, seed=1, mid=50.0, tick=0.5)
        )

    assert [e.price for e in events()] == [49.5, 50.5]
    assert events() == events()
