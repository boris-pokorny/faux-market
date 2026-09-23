"""Measure which stylized facts the zero-intelligence flow reproduces.

Each section prints one statistic and a one-line reading. Only the
standard library is used. Numbers are in ticks and event time.
"""

import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field

import faux_market as fm

DEPTH_EVERY = 1_000


@dataclass(slots=True)
class MarketOrder:
    """One market order: its sign, total size and the mid before/after."""

    ts: float
    sign: int
    size: int
    mid_before: float
    mid_after: float


@dataclass
class Sample:
    """Series collected from one simulation pass."""

    mid_changes: list[float] = field(default_factory=list)
    trade_prices: list[float] = field(default_factory=list)
    orders: list[MarketOrder] = field(default_factory=list)
    spreads: list[int] = field(default_factory=list)
    depth: dict[int, float] = field(default_factory=dict)


def collect(flow: fm.ZeroIntelligence, n_events: int, seed: int) -> Sample:
    """Run ``flow`` for ``n_events`` and record the series in ``Sample``."""
    sim = fm.Simulation(flow, seed=seed)
    book = sim.book
    out = Sample()
    depth_sum: dict[int, int] = {}
    depth_n = 0
    last_mid: float | None = None
    cur: MarketOrder | None = None  # market order being filled
    for i, ev in enumerate(sim.run(n_events)):
        mid = book.mid
        if mid is not None and last_mid is not None and mid != last_mid:
            out.mid_changes.append(mid - last_mid)
        if ev.action is fm.Action.EXECUTE and mid is not None:
            out.trade_prices.append(ev.price / sim.tick)
            sign = 1 if ev.side is fm.Side.ASK else -1
            if cur is None or cur.ts != ev.ts:
                if cur is not None:
                    out.orders.append(cur)
                cur = MarketOrder(ev.ts, sign, 0, last_mid or mid, mid)
            cur.size += ev.size
            cur.mid_after = mid
        if mid is not None:
            last_mid = mid
        spread = book.spread
        if spread is not None:
            out.spreads.append(spread)
        if i % DEPTH_EVERY == 0 and book.best_bid and book.best_ask:
            depth_n += 1
            for price, size in book.depth(fm.Side.BID).items():
                d = book.best_bid - price
                depth_sum[d] = depth_sum.get(d, 0) + size
            for price, size in book.depth(fm.Side.ASK).items():
                d = price - book.best_ask
                depth_sum[d] = depth_sum.get(d, 0) + size
    out.depth = {d: s / depth_n for d, s in sorted(depth_sum.items())}
    return out


def autocorr(x: Sequence[float], lag: int) -> float:
    """Sample autocorrelation of ``x`` at ``lag``."""
    m = statistics.fmean(x)
    var = sum((a - m) ** 2 for a in x)
    cov = sum((x[i] - m) * (x[i + lag] - m) for i in range(len(x) - lag))
    return cov / var


def kurtosis(x: Sequence[float]) -> float:
    """Population kurtosis; 3 for a Gaussian."""
    m = statistics.fmean(x)
    s = statistics.pstdev(x)
    return sum(((a - m) / s) ** 4 for a in x) / len(x)


def aggregate(x: Sequence[float], h: int) -> list[float]:
    """Non-overlapping sums of ``h`` consecutive values."""
    return [sum(x[i : i + h]) for i in range(0, len(x) - h + 1, h)]


def variance_ratio(x: Sequence[float], h: int) -> float:
    """Var of ``h``-step sums over ``h`` times the one-step variance."""
    return statistics.pvariance(aggregate(x, h)) / (h * statistics.pvariance(x))


def section(title: str, source: str) -> None:
    """Print a section header."""
    print(f"\n{title}\n  ({source})")


def print_depth_profile(s: Sample) -> None:
    """Fact 1: mean depth is hump-shaped away from the best quote."""
    section("1. Hump-shaped depth profile", "Bouchaud, Mezard & Potters 2002")
    print("  ticks from best  mean size")
    for d, size in s.depth.items():
        if d <= 24 and d % 2 == 0:
            print(f"  {d:15d}  {size:9.0f}  {'#' * int(size / 20)}")
    peak = max(s.depth, key=lambda d: s.depth[d])
    print(f"  reading: peak at {peak} ticks, not at the best -> hump")


def print_mean_reversion(s: Sample) -> None:
    """Fact 2: mid changes anticorrelate and the price is sub-diffusive."""
    section("2. Short-horizon mean reversion", "Smith et al. 2003")
    r = s.mid_changes
    print(
        f"  mid-change autocorr  lag 1: {autocorr(r, 1):+.3f}"
        f"   lag 2: {autocorr(r, 2):+.3f}"
    )
    print(
        "  variance ratio  "
        + "  ".join(f"h={h}: {variance_ratio(r, h):.2f}" for h in (2, 10, 100))
    )
    print("  reading: negative lag-1 and ratio < 1 -> mean reverting")


def print_bid_ask_bounce(s: Sample) -> None:
    """Fact 3: trade-price changes anticorrelate."""
    section("3. Bid-ask bounce", "Roll 1984")
    p = s.trade_prices
    dp = [p[i + 1] - p[i] for i in range(len(p) - 1)]
    print(f"  trade-price-change autocorr  lag 1: {autocorr(dp, 1):+.3f}")
    print("  reading: negative -> trades bounce between bid and ask")


def print_spread_scaling(seed: int) -> None:
    """Fact 4: the spread widens with market-order pressure."""
    section("4. Spread grows with market-order rate", "Smith et al. 2003")
    print("  market_rate  mean spread")
    for rate in (0.5, 2.0, 4.0):
        s = collect(fm.ZeroIntelligence(market_rate=rate), 100_000, seed)
        print(f"  {rate:11.1f}  {statistics.fmean(s.spreads):11.2f}")
    print("  reading: monotone -> more market orders eat more depth")


def print_price_impact(s: Sample) -> None:
    """Fact 5: mid moves more after larger market orders."""
    section("5. Price impact grows with order size", "Smith et al. 2003")
    buckets: dict[int, list[float]] = {}
    for o in s.orders:
        moves = buckets.setdefault(min(o.size // 10, 9), [])
        moves.append(o.sign * (o.mid_after - o.mid_before))
    print("  size decile  mean signed mid move")
    for k, moves in sorted(buckets.items()):
        print(f"  {k * 10:4d}-{k * 10 + 9:<4d}  {statistics.fmean(moves):8.3f}")
    print("  reading: increasing in size (roughly linear here, not concave)")


def print_fat_tails(dense: Sample, sparse: Sample) -> None:
    """Fact 6: sparse books give fat tails that fade under aggregation."""
    section("6. Fat tails, Gaussian on aggregation", "Smith et al. 2003")
    print("  kurtosis of mid changes   h=1    h=10   h=100")
    for name, s in (("default", dense), ("sparse", sparse)):
        ks = [kurtosis(aggregate(s.mid_changes, h)) for h in (1, 10, 100)]
        print(f"  {name:24s}" + "".join(f"{k:7.2f}" for k in ks))
    print(
        "  reading: sparse book > 3 at h=1, falling toward 3;"
        " dense book is never fat-tailed"
    )


def print_no_clustering(s: Sample) -> None:
    """Absent fact: volatility clustering."""
    section("7. Volatility clustering: absent", "Cont 2001")
    a = [abs(r) for r in s.mid_changes]
    print(
        "  |mid change| autocorr  "
        + "  ".join(f"lag {lag}: {autocorr(a, lag):+.3f}" for lag in (1, 5, 20))
    )
    print("  reading: lag 1 is the bounce artefact; ~0 by lag 20")


def print_no_sign_memory(s: Sample) -> None:
    """Absent fact: long memory in order signs."""
    section("8. Long memory of order signs: absent", "Lillo & Farmer 2004")
    signs = [float(o.sign) for o in s.orders]
    print(
        "  order-sign autocorr  "
        + "  ".join(
            f"lag {lag}: {autocorr(signs, lag):+.3f}" for lag in (1, 2, 10)
        )
    )
    print("  reading: ~0 at every lag; ZI signs are i.i.d.")


def main() -> None:
    """Collect samples and print every section."""
    seed = 7
    dense = collect(fm.ZeroIntelligence(), 300_000, seed)
    sparse = collect(
        fm.ZeroIntelligence(add_rate=0.3, cancel_rate=0.5), 300_000, seed
    )
    print_depth_profile(dense)
    print_mean_reversion(dense)
    print_bid_ask_bounce(dense)
    print_spread_scaling(seed)
    print_price_impact(dense)
    print_fat_tails(dense, sparse)
    print_no_clustering(dense)
    print_no_sign_memory(dense)
    print(
        "\nFacts 7 and 8 need strategic or heterogeneous agents,"
        " which zero intelligence lacks."
    )


if __name__ == "__main__":
    main()
