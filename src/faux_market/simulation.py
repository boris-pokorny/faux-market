"""Event-driven simulation engine.

The engine owns the book, the clock and order ids. An :class:`OrderFlow` decides
what happens next by returning an intent; the engine applies it and emits the
resulting L3 events.
"""

import random
from collections.abc import Iterator
from itertools import islice

from faux_market.book import Book, Order
from faux_market.events import Action, L3Event, Side
from faux_market.order_flow import Cancel, Intent, Limit, Market, OrderFlow

__all__ = ["Simulation"]


class Simulation:
    """Applies an order flow's intents to a book and emits L3 events.

    The event stream is a single lazy iterator: :meth:`run` slices it, so
    repeated calls continue where the previous one stopped and no event is
    ever skipped, even when one intent produces several events.

    Attributes:
        book: The live limit order book, in integer ticks.
        tick: Tick size used to convert book prices to event prices.
    """

    def __init__(
        self,
        order_flow: OrderFlow,
        *,
        seed: int | None = None,
        mid: float = 100.0,
        tick: float = 0.01,
    ) -> None:
        """Create a simulation whose book is seeded by ``order_flow.warm_up``.

        Warm-up orders are emitted as ``ADD`` events at time zero when the
        stream is first consumed.

        Args:
            order_flow: Source of order intents.
            seed: RNG seed; ``None`` means nondeterministic.
            mid: Initial reference price.
            tick: Tick size.

        Raises:
            ValueError: If ``mid`` or ``tick`` is not positive.
        """
        if mid <= 0 or tick <= 0:
            raise ValueError("mid and tick must be positive")
        self.book = Book()
        self.tick = tick
        self._order_flow = order_flow
        self._rng = random.Random(seed)
        self._ref = round(mid / tick)
        self._ts = 0.0
        self._next_id = 1
        self._events = self._generate()

    def __iter__(self) -> Iterator[L3Event]:
        """The unbounded event stream."""
        return self._events

    def run(self, n_events: int) -> Iterator[L3Event]:
        """Yield the next ``n_events`` book events."""
        return islice(self._events, n_events)

    def _generate(self) -> Iterator[L3Event]:
        for limit in self._order_flow.warm_up(self._ref, self._rng):
            yield from self._apply(limit)
        while True:
            dt, intent = self._order_flow.step(self.book, self._rng)
            self._ts += dt
            yield from self._apply(intent)

    def _apply(self, intent: Intent) -> list[L3Event]:
        match intent:
            case Limit(side, price, size):
                order = Order(self._next_id, side, price, size)
                self._next_id += 1
                self.book.add(order)
                return [self._event(order, Action.ADD, size)]
            case Cancel(order_id):
                order = self.book.cancel(order_id)
                return [self._event(order, Action.CANCEL, order.size)]
            case Market(side, size):
                # A market buy hits the asks; its fills are ask-side executes.
                hit = Side.ASK if side is Side.BID else Side.BID
                return [
                    self._event(f.order, Action.EXECUTE, f.size)
                    for f in self.book.match(hit, size)
                ]

    def _event(self, order: Order, action: Action, size: int) -> L3Event:
        return L3Event(
            self._ts,
            order.order_id,
            order.side,
            action,
            round(order.price * self.tick, 10),
            size,
        )
