"""Zero-intelligence order flow.

Follows Smith, Farmer, Gillemot & Krishnamurthy (2003): limit orders arrive
at rate ``add_rate`` per tick within ``window`` ticks of the opposite best
quote, each resting order is cancelled at hazard ``cancel_rate``, and market
orders arrive at rate ``market_rate`` per side. Prices move only through
these flows.
"""

import random
from collections.abc import Iterator

from faux_market.events import Side
from faux_market.order_flow import BookView, Cancel, Intent, Limit, Market

__all__ = ["ZeroIntelligence"]


class ZeroIntelligence:
    """Poisson order flow with no view on price."""

    def __init__(
        self,
        *,
        add_rate: float = 1.0,
        cancel_rate: float = 0.2,
        market_rate: float = 2.0,
        window: int = 20,
        max_size: int = 100,
    ) -> None:
        """Set the flow rates.

        Args:
            add_rate: Limit-order arrival rate per tick per side.
            cancel_rate: Cancellation hazard per resting order.
            market_rate: Market-order arrival rate per side.
            window: Limit orders land within this many ticks of the opposite
                best quote.
            max_size: Limit and market order sizes are uniform in
                ``[1, max_size]``.

        Raises:
            ValueError: If a rate is negative, both ``add_rate`` and
                ``market_rate`` are zero, or ``window`` or ``max_size`` is
                not positive.
        """
        if min(add_rate, cancel_rate, market_rate) < 0:
            raise ValueError("rates must be non-negative")
        if add_rate == 0 and market_rate == 0:
            raise ValueError("add_rate and market_rate cannot both be zero")
        if window < 1 or max_size < 1:
            raise ValueError("window and max_size must be positive")
        self._window = window
        self._max_size = max_size
        self._add_total = add_rate * window * 2
        self._cancel_rate = cancel_rate
        self._market_total = market_rate * 2
        self._ref = 0

    def warm_up(self, ref: int, rng: random.Random) -> Iterator[Limit]:
        """One order per tick on each side, ``window`` ticks deep."""
        self._ref = ref
        for k in range(1, self._window + 1):
            yield Limit(Side.BID, ref - k, self._size(rng))
            yield Limit(Side.ASK, ref + k, self._size(rng))

    def step(self, book: BookView, rng: random.Random) -> tuple[float, Intent]:
        """Draw the next event from competing Poisson clocks."""
        cancel_total = self._cancel_rate * len(book)
        total = self._add_total + cancel_total + self._market_total
        dt = rng.expovariate(total)
        side = rng.choice((Side.BID, Side.ASK))
        u = rng.random() * total
        if u < self._add_total:
            k = rng.randint(1, self._window)
            return dt, Limit(side, self._price(book, side, k), self._size(rng))
        if u < self._add_total + cancel_total:
            return dt, Cancel(rng.choice(book.order_ids))
        return dt, Market(side, self._size(rng))

    def _price(self, book: BookView, side: Side, k: int) -> int:
        """Price ``k`` ticks inside the opposite best quote.

        Falls back to the last trade (or initial reference) when the
        opposite side is empty.
        """
        ref = book.last_trade if book.last_trade is not None else self._ref
        if side is Side.BID:
            ask = book.best_ask
            return (ask if ask is not None else ref + 1) - k
        bid = book.best_bid
        return (bid if bid is not None else ref - 1) + k

    def _size(self, rng: random.Random) -> int:
        return rng.randint(1, self._max_size)
