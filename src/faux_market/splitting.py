"""Order-splitting wrapper around any order flow.

Follows Lillo, Mike & Farmer (2005): each of ``n_traders`` holds a hidden
metaorder of Pareto-distributed size and works it off through the wrapped
flow's market orders. Signs of consecutive market orders are therefore
correlated, decaying roughly as ``lag ** -(alpha - 1)``.
"""

import math
import random
from collections.abc import Iterable
from dataclasses import dataclass

from faux_market.events import Side
from faux_market.order_flow import BookView, Intent, Limit, Market, OrderFlow
from faux_market.zi import ZeroIntelligence

__all__ = ["OrderSplitting"]


@dataclass(slots=True)
class _Trader:
    side: Side = Side.BID
    remaining: int = 0


class OrderSplitting:
    """Rewrites the inner flow's market orders as metaorder children."""

    def __init__(
        self,
        inner: OrderFlow | None = None,
        *,
        n_traders: int = 3,
        alpha: float = 1.5,
        min_size: int = 50,
        max_meta: int = 100_000,
    ) -> None:
        """Wrap ``inner`` (default :class:`ZeroIntelligence`).

        Args:
            inner: Flow supplying limit, cancel and market intents.
            n_traders: Number of traders holding metaorders; lag-1
                sign autocorrelation scales roughly as ``1 / n_traders``.
            alpha: Pareto tail exponent of metaorder size; must exceed 1.
            min_size: Pareto scale, the smallest metaorder, in shares.
                Far below the child size kills the memory; far above it
                flattens the autocorrelation out to many lags.
            max_meta: Cap on metaorder size.

        Raises:
            ValueError: If ``alpha <= 1`` or a size or count is not positive.
        """
        if alpha <= 1:
            raise ValueError("alpha must exceed 1")
        if min(n_traders, min_size, max_meta) < 1:
            raise ValueError("sizes and n_traders must be positive")
        self._inner = inner if inner is not None else ZeroIntelligence()
        self._alpha = alpha
        self._min_size = min_size
        self._max_meta = max_meta
        self._traders = [_Trader() for _ in range(n_traders)]

    def warm_up(self, ref: int, rng: random.Random) -> Iterable[Limit]:
        """Delegate to the inner flow."""
        return self._inner.warm_up(ref, rng)

    def step(self, book: BookView, rng: random.Random) -> tuple[float, Intent]:
        """Pass limits and cancels through; assign markets to a trader."""
        dt, intent = self._inner.step(book, rng)
        if not isinstance(intent, Market):
            return dt, intent
        trader = rng.choice(self._traders)
        if trader.remaining == 0:
            trader.side = rng.choice((Side.BID, Side.ASK))
            trader.remaining = self._meta_size(rng)
        size = min(intent.size, trader.remaining)
        trader.remaining -= size
        return dt, Market(trader.side, size)

    def _meta_size(self, rng: random.Random) -> int:
        u = 1.0 - rng.random()  # in (0, 1]
        scale: float = u ** (-1.0 / self._alpha)
        size = math.ceil(self._min_size * scale)
        return min(size, self._max_meta)
