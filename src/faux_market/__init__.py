"""faux-market: a lightweight financial-market generator."""

from collections.abc import Iterator

from faux_market.book import Book, Fill, Order
from faux_market.events import Action, L3Event, Side
from faux_market.order_flow import (
    BookView,
    Cancel,
    Intent,
    Limit,
    Market,
    OrderFlow,
)
from faux_market.simulation import Simulation
from faux_market.zi import ZeroIntelligence

__all__ = [
    "Action",
    "Book",
    "BookView",
    "Cancel",
    "Fill",
    "Intent",
    "L3Event",
    "Limit",
    "Market",
    "OrderFlow",
    "Order",
    "Side",
    "Simulation",
    "ZeroIntelligence",
    "__version__",
    "simulate",
]
__version__ = "0.1.0"


def simulate(
    n_events: int,
    *,
    order_flow: OrderFlow | None = None,
    seed: int | None = None,
    mid: float = 100.0,
    tick: float = 0.01,
) -> Iterator[L3Event]:
    """Yield ``n_events`` L3 events from a fresh :class:`Simulation`.

    Use :class:`Simulation` directly to inspect the book between events.

    Args:
        n_events: Number of events to yield.
        order_flow: Defaults to a fresh :class:`ZeroIntelligence`.
        seed: RNG seed; ``None`` means nondeterministic.
        mid: Initial reference price.
        tick: Tick size.

    Yields:
        Events in time order.
    """
    if order_flow is None:
        order_flow = ZeroIntelligence()
    return Simulation(order_flow, seed=seed, mid=mid, tick=tick).run(n_events)
