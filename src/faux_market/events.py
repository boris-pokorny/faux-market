"""L3 (order-level) market data events."""

import enum
from dataclasses import dataclass

__all__ = ["Action", "L3Event", "Side"]


class Side(enum.StrEnum):
    """Order side."""

    BID = "bid"
    ASK = "ask"


class Action(enum.StrEnum):
    """What happened to the order."""

    ADD = "add"
    CANCEL = "cancel"
    REDUCE = "reduce"
    EXECUTE = "execute"


@dataclass(frozen=True, slots=True)
class L3Event:
    """One order-level book event.

    Attributes:
        ts: Event time in seconds since simulation start.
        order_id: Identifier of the affected order.
        side: Book side of the order.
        action: Add, cancel, reduce or execute. Only ``CANCEL`` and a full
            ``EXECUTE`` remove the order; ``REDUCE`` keeps its priority.
        price: Limit price of the order.
        size: Quantity added, cancelled, removed by a reduce or executed.
    """

    ts: float
    order_id: int
    side: Side
    action: Action
    price: float
    size: int
