"""Limit order book with price-time priority."""

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

from faux_market.events import Side

__all__ = ["Book", "Fill", "Order"]


@dataclass(eq=False, slots=True)
class Order:
    """A resting limit order. Compared by identity.

    Attributes:
        order_id: Unique identifier.
        side: Book side.
        price: Limit price in ticks; must be positive.
        size: Remaining quantity; positive when added.
    """

    order_id: int
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
class Fill:
    """A resting order (partially) executed by a market order.

    Attributes:
        order: The resting order after the fill (``size`` already reduced).
        size: Quantity executed.
    """

    order: Order
    size: int


class Book:
    """Two-sided limit order book keyed by integer tick prices.

    Orders at the same price fill in arrival order.

    Attributes:
        last_trade: Price in ticks of the most recent fill, or ``None``.
    """

    def __init__(self) -> None:
        """Create an empty book."""
        self.last_trade: int | None = None
        self._levels: dict[Side, dict[int, deque[Order]]] = {
            Side.BID: {},
            Side.ASK: {},
        }
        self._orders: dict[int, Order] = {}
        # Cached best price per side; rescanned only when that level empties.
        self._best: dict[Side, int | None] = {Side.BID: None, Side.ASK: None}
        # Ids as a list (for O(1) random choice) with their positions.
        self._ids: list[int] = []
        self._pos: dict[int, int] = {}

    def __len__(self) -> int:
        """Number of resting orders."""
        return len(self._orders)

    def __contains__(self, order_id: int) -> bool:
        """Whether ``order_id`` is resting in the book."""
        return order_id in self._orders

    @property
    def order_ids(self) -> Sequence[int]:
        """Identifiers of all resting orders, in no particular order.

        A live read-only view; it changes as the book changes.
        """
        return self._ids

    @property
    def best_bid(self) -> int | None:
        """Highest bid price in ticks, or ``None`` if no bids."""
        return self._best[Side.BID]

    @property
    def best_ask(self) -> int | None:
        """Lowest ask price in ticks, or ``None`` if no asks."""
        return self._best[Side.ASK]

    @property
    def mid(self) -> float | None:
        """Mid price in ticks, or ``None`` if either side is empty."""
        bid, ask = self.best_bid, self.best_ask
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2

    @property
    def spread(self) -> int | None:
        """Best ask minus best bid in ticks, or ``None`` if a side is empty."""
        bid, ask = self.best_bid, self.best_ask
        if bid is None or ask is None:
            return None
        return ask - bid

    def depth(self, side: Side) -> dict[int, int]:
        """Total resting size per price level on ``side``."""
        return {
            price: sum(o.size for o in queue)
            for price, queue in self._levels[side].items()
        }

    def add(self, order: Order) -> None:
        """Rest ``order`` in the book.

        Raises:
            ValueError: If the order would cross the book or its id is taken.
        """
        if order.order_id in self._orders:
            raise ValueError(f"duplicate order id {order.order_id}")
        if order.side is Side.BID:
            crosses = self.best_ask is not None and order.price >= self.best_ask
        else:
            crosses = self.best_bid is not None and order.price <= self.best_bid
        if crosses:
            raise ValueError("limit order would cross the book")
        self._levels[order.side].setdefault(order.price, deque()).append(order)
        self._orders[order.order_id] = order
        self._pos[order.order_id] = len(self._ids)
        self._ids.append(order.order_id)
        best = self._best[order.side]
        if best is None or self._better(order.side, order.price, best):
            self._best[order.side] = order.price

    def cancel(self, order_id: int) -> Order:
        """Remove and return the resting order ``order_id``.

        Raises:
            KeyError: If no such order is resting.
        """
        order = self._orders.pop(order_id)
        self._forget(order)
        return order

    def match(self, side: Side, size: int) -> list[Fill]:
        """Execute a market order of ``size`` against the ``side`` queue.

        Fills walk from the best price outward until ``size`` is exhausted or
        the side is empty.

        Args:
            side: Side of the resting orders being hit.
            size: Quantity to execute.

        Returns:
            Fills in execution order.

        Raises:
            ValueError: If ``size`` is not positive.
        """
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")
        fills: list[Fill] = []
        while size > 0:
            best = self.best_bid if side is Side.BID else self.best_ask
            if best is None:
                break
            queue = self._levels[side][best]
            order = queue[0]
            filled = min(size, order.size)
            order.size -= filled
            size -= filled
            if order.size == 0:
                del self._orders[order.order_id]
                self._forget(order)
            fills.append(Fill(order, filled))
            self.last_trade = best
        return fills

    def _forget(self, order: Order) -> None:
        """Drop ``order`` from its level and the id list."""
        levels = self._levels[order.side]
        queue = levels[order.price]
        queue.remove(order)
        if not queue:
            del levels[order.price]
            if order.price == self._best[order.side]:
                self._best[order.side] = self._scan_best(order.side)
        last = self._ids.pop()
        pos = self._pos.pop(order.order_id)
        if last != order.order_id:
            self._ids[pos] = last
            self._pos[last] = pos

    def _scan_best(self, side: Side) -> int | None:
        prices = self._levels[side]
        if not prices:
            return None
        return max(prices) if side is Side.BID else min(prices)

    @staticmethod
    def _better(side: Side, price: int, than: int) -> bool:
        return price > than if side is Side.BID else price < than
