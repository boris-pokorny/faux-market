"""Contract between the simulation engine and order flows."""

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from faux_market.events import Side

__all__ = ["BookView", "Cancel", "Intent", "Limit", "Market", "OrderFlow"]


class BookView(Protocol):
    """Read-only view of a limit order book, as seen by an :class:`OrderFlow`.

    Flows observe the book through this interface; only the engine mutates
    it. :class:`faux_market.Book` satisfies it structurally.
    """

    @property
    def last_trade(self) -> int | None:
        """Price in ticks of the most recent fill, or ``None``."""
        ...

    @property
    def best_bid(self) -> int | None:
        """Highest bid price in ticks, or ``None`` if no bids."""
        ...

    @property
    def best_ask(self) -> int | None:
        """Lowest ask price in ticks, or ``None`` if no asks."""
        ...

    @property
    def mid(self) -> float | None:
        """Mid price in ticks, or ``None`` if either side is empty."""
        ...

    @property
    def spread(self) -> int | None:
        """Best ask minus best bid in ticks, or ``None`` if a side is empty."""
        ...

    @property
    def order_ids(self) -> Sequence[int]:
        """Identifiers of all resting orders; a live read-only view."""
        ...

    def depth(self, side: Side) -> dict[int, int]:
        """Total resting size per price level on ``side``."""
        ...

    def __len__(self) -> int:
        """Number of resting orders."""
        ...

    def __contains__(self, order_id: int) -> bool:
        """Whether ``order_id`` is resting in the book."""
        ...


@dataclass(frozen=True, slots=True)
class Limit:
    """Rest a limit order.

    Attributes:
        side: Book side.
        price: Limit price in ticks; must be positive.
        size: Quantity; must be positive.
    """

    side: Side
    price: int
    size: int

    def __post_init__(self) -> None:
        """Validate price and size."""
        if self.price <= 0:
            raise ValueError(f"price must be positive, got {self.price}")
        if self.size <= 0:
            raise ValueError(f"size must be positive, got {self.size}")


@dataclass(frozen=True, slots=True)
class Cancel:
    """Cancel the resting order ``order_id``."""

    order_id: int


@dataclass(frozen=True, slots=True)
class Market:
    """Send a market order of positive ``size`` on ``side``."""

    side: Side
    size: int

    def __post_init__(self) -> None:
        """Validate size."""
        if self.size <= 0:
            raise ValueError(f"size must be positive, got {self.size}")


type Intent = Limit | Cancel | Market


class OrderFlow(Protocol):
    """Source of order intents driving a :class:`Simulation`.

    The engine calls :meth:`warm_up` once, then :meth:`step` repeatedly,
    applying each intent to the book.
    """

    def warm_up(self, ref: int, rng: random.Random) -> Iterable[Limit]:
        """Limit orders seeding the book around reference price ``ref``."""
        ...

    def step(self, book: BookView, rng: random.Random) -> tuple[float, Intent]:
        """Time until the next intent and the intent itself."""
        ...
