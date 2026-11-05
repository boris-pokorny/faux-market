# faux-market

A lightweight financial-market generator.

```python
import faux_market as fm

for event in fm.simulate(100, seed=42):
    print(event)
```

`fm.simulate()` yields `L3Event`s (order add / cancel / reduce / execute). It accepts
`order_flow`, `seed`, `mid` and `tick`; use `fm.Simulation` directly to inspect the
book between events.

Order flows: `fm.ZeroIntelligence` (Smith et al. 2003) and `fm.OrderSplitting`,
which wraps any flow and works its market orders off as Pareto-sized metaorders
(Lillo, Mike & Farmer 2005). Flows return `fm.Limit`, `fm.Cancel`, `fm.Modify` or
`fm.Market` intents.

## Development

```sh
pdm install -G dev
pdm run lint        # ruff
pdm run typecheck   # mypy --strict
pdm run test        # pytest
pdm run example     # examples/basic.py
pdm run facts       # examples/stylized_facts.py
```
