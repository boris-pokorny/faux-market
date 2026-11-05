import random

import pytest

from faux_market import Book, Order, Side


def make_book() -> Book:
    book = Book()
    book.add(Order(1, Side.BID, 99, 5))
    book.add(Order(2, Side.BID, 98, 5))
    book.add(Order(3, Side.ASK, 101, 5))
    book.add(Order(4, Side.ASK, 101, 5))
    return book


def test_quotes() -> None:
    book = make_book()
    assert (book.best_bid, book.best_ask) == (99, 101)
    assert book.mid == 100 and book.spread == 2
    assert book.depth(Side.ASK) == {101: 10}


def test_empty_side() -> None:
    book = Book()
    book.add(Order(1, Side.BID, 99, 1))
    assert book.best_ask is None and book.mid is None and book.spread is None


def test_add_rejects_crossing_and_duplicates() -> None:
    book = make_book()
    with pytest.raises(ValueError):
        book.add(Order(5, Side.BID, 101, 1))
    with pytest.raises(ValueError):
        book.add(Order(1, Side.ASK, 105, 1))


def test_cancel() -> None:
    book = make_book()
    assert book.cancel(1).order_id == 1
    assert book.best_bid == 98 and 1 not in book
    with pytest.raises(KeyError):
        book.cancel(1)


def test_match_fifo_and_partial() -> None:
    book = make_book()
    fills = book.match(Side.ASK, 7)
    assert [(f.order.order_id, f.size) for f in fills] == [(3, 5), (4, 2)]
    assert 3 not in book and book.depth(Side.ASK) == {101: 3}


def test_match_walks_levels_and_stops_when_empty() -> None:
    book = make_book()
    fills = book.match(Side.BID, 20)
    assert [f.order.price for f in fills] == [99, 98]
    assert book.best_bid is None and len(book) == 2


def test_cancel_mid_queue_keeps_fifo() -> None:
    book = Book()
    for oid in (1, 2, 3):
        book.add(Order(oid, Side.ASK, 101, 1))
    book.cancel(2)
    fills = book.match(Side.ASK, 2)
    assert [f.order.order_id for f in fills] == [1, 3]


@pytest.mark.parametrize(
    ("side", "price", "size", "filled", "rest"),
    [
        pytest.param(Side.BID, 100, 3, [], 3, id="no-cross"),
        pytest.param(Side.BID, 101, 7, [(3, 5), (4, 2)], 0, id="filled"),
        pytest.param(Side.BID, 101, 12, [(3, 5), (4, 5)], 2, id="buy-limit"),
        pytest.param(Side.ASK, 98, 12, [(1, 5), (2, 5)], 2, id="sell-limit"),
    ],
)
def test_submit(
    side: Side, price: int, size: int, filled: list[tuple[int, int]], rest: int
) -> None:
    book = make_book()
    book.add(Order(5, Side.ASK, 102, 5))
    book.add(Order(6, Side.BID, 97, 5))
    fills = book.submit(Order(7, side, price, size))
    assert [(f.order.order_id, f.size) for f in fills] == filled
    assert book.level_size(side, price) == rest
    assert (7 in book) == (rest > 0)


def test_submit_duplicate_id_leaves_book_untouched() -> None:
    book = make_book()
    with pytest.raises(ValueError):
        book.submit(Order(1, Side.BID, 101, 5))
    assert book.depth(Side.ASK) == {101: 10}


def test_reduce_keeps_queue_position() -> None:
    book = make_book()
    book.reduce(3, 2)
    assert book.depth(Side.ASK) == {101: 7}
    fills = book.match(Side.ASK, 3)
    assert [(f.order.order_id, f.size) for f in fills] == [(3, 2), (4, 1)]


@pytest.mark.parametrize("size", [0, 5])
def test_reduce_rejects_bad_sizes(size: int) -> None:
    with pytest.raises(ValueError):
        make_book().reduce(3, size)


def test_order_and_match_reject_non_positive_values() -> None:
    with pytest.raises(ValueError):
        Order(1, Side.BID, 0, 1)
    with pytest.raises(ValueError):
        Order(1, Side.BID, 1, 0)
    with pytest.raises(ValueError):
        make_book().match(Side.ASK, 0)


def test_random_ops_keep_quotes_and_ids_consistent() -> None:
    rng = random.Random(0)
    book = Book()
    expected: set[int] = set()
    next_id = 1
    for _ in range(5000):
        u = rng.random()
        if u < 0.5:
            side = rng.choice((Side.BID, Side.ASK))
            bid = 100 if book.best_bid is None else book.best_bid
            ask = 101 if book.best_ask is None else book.best_ask
            if side is Side.BID:
                price = bid - rng.randint(0, 5)
                crosses = price >= ask
            else:
                price = ask + rng.randint(0, 5)
                crosses = price <= bid
            if price > 0 and not crosses:
                book.add(Order(next_id, side, price, rng.randint(1, 3)))
                expected.add(next_id)
                next_id += 1
        elif u < 0.6 and expected:
            oid = rng.choice(book.order_ids)
            if book.order(oid).size > 1:
                book.reduce(oid, book.order(oid).size - 1)
        elif u < 0.8 and expected:
            oid = rng.choice(book.order_ids)
            book.cancel(oid)
            expected.discard(oid)
        elif expected:
            side = rng.choice((Side.BID, Side.ASK))
            for f in book.match(side, rng.randint(1, 4)):
                if f.order.size == 0:
                    expected.discard(f.order.order_id)
        assert set(book.order_ids) == expected and len(book) == len(expected)
        for side, pick in ((Side.BID, max), (Side.ASK, min)):
            depth = book.depth(side)
            best = book.best_bid if side is Side.BID else book.best_ask
            assert best == (pick(depth) if depth else None)
            totals: dict[int, int] = {}
            for oid in book.order_ids:
                o = book.order(oid)
                if o.side is side:
                    totals[o.price] = totals.get(o.price, 0) + o.size
            assert depth == totals
            assert all(book.level_size(side, p) == totals[p] for p in totals)
