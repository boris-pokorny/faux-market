"""Print a short stream of simulated L3 events."""

import faux_market as fm


def main() -> None:
    """Run the simulation and print each event."""
    for ev in fm.simulate(60, seed=42):
        print(
            f"{ev.ts:8.3f}  #{ev.order_id:<4} {ev.side:<3} "
            f"{ev.action:<7} {ev.price:8.2f} x {ev.size}"
        )


if __name__ == "__main__":
    main()
