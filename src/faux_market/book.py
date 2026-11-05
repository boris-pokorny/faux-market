"""Limit order book with price-time priority."""

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field

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
    """A resting order (partially) executed by an incoming order.

    Attributes:
        order: The resting order after the fill (``size`` already reduced).
        size: Quantity executed.
    """

    order: Order
    size: int


@dataclass(slots=True)
class _Level:
    """FIFO queue at one price and its total size."""

    orders: OrderedDict[int, Order] = field(default_factory=OrderedDict)
    size: int = 0


class Book:
    """Two-sided limit order book keyed by integer tick prices.

    Orders at the same price fill in arrival order.

    Attributes:
        last_trade: Price in ticks of the most recent fill, or ``None``.
    """

    def __init__(self) -> None:
        """Create an empty book."""
        self.last_trade: int | None = None
        self._levels: dict[Side, dict[int, _Level]] = {
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
            price: level.size for price, level in self._levels[side].items()
        }

    def level_size(self, side: Side, price: int) -> int:
        """Total resting size at ``price``; zero if the level is empty."""
        level = self._levels[side].get(price)
        return 0 if level is None else level.size

    def order(self, order_id: int) -> Order:
        """The resting order ``order_id``; treat it as read-only.

        Raises:
            KeyError: If no such order is resting.
        """
        return self._orders[order_id]

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
        levels = self._levels[order.side]
        level = levels.get(order.price)
        if level is None:
            level = levels[order.price] = _Level()
        level.orders[order.order_id] = order
        level.size += order.size
        self._orders[order.order_id] = order
        self._pos[order.order_id] = len(self._ids)
        self._ids.append(order.order_id)
        best = self._best[order.side]
        if best is None or self._better(order.side, order.price, best):
            self._best[order.side] = order.price

    def submit(self, order: Order) -> list[Fill]:
        """Execute ``order`` up to its limit, then rest any remainder.

        Args:
            order: The incoming order; ``size`` is reduced by its fills.

        Returns:
            Fills of the resting orders, in execution order.

        Raises:
            ValueError: If the order id is taken.
        """
        if order.order_id in self._orders:
            raise ValueError(f"duplicate order id {order.order_id}")
        hit = Side.ASK if order.side is Side.BID else Side.BID
        best = self._best[hit]
        fills: list[Fill] = []
        if best is not None and not self._better(hit, order.price, best):
            fills = self.match(hit, order.size, limit=order.price)
            order.size -= sum(f.size for f in fills)
        if order.size > 0:
            self.add(order)
        return fills

    def reduce(self, order_id: int, size: int) -> None:
        """Lower the order's remaining size to ``size``, keeping its priority.

        Raises:
            KeyError: If no such order is resting.
            ValueError: Unless ``0 < size`` < the remaining size.
        """
        order = self._orders[order_id]
        if not 0 < size < order.size:
            raise ValueError(f"cannot reduce {order.size} to {size}")
        self._levels[order.side][order.price].size -= order.size - size
        order.size = size

    def cancel(self, order_id: int) -> Order:
        """Remove and return the resting order ``order_id``.

        Raises:
            KeyError: If no such order is resting.
        """
        order = self._orders.pop(order_id)
        self._forget(order)
        return order

    def match(
        self, side: Side, size: int, limit: int | None = None
    ) -> list[Fill]:
        """Execute an incoming ``size`` against the ``side`` queue.

        Fills walk from the best price outward until ``size`` is exhausted,
        the side is empty or the best price is worse than ``limit``.

        Args:
            side: Side of the resting orders being hit.
            size: Quantity to execute.
            limit: Worst price to trade at; ``None`` means any.

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
            if limit is not None and self._better(side, limit, best):
                break
            level = self._levels[side][best]
            order = next(iter(level.orders.values()))
            filled = min(size, order.size)
            order.size -= filled
            level.size -= filled
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
        level = levels[order.price]
        del level.orders[order.order_id]
        level.size -= order.size
        if not level.orders:
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
