# faux-market

A lightweight financial-market generator.

```python
import faux_market as fm

for event in fm.simulate(100, seed=42):
    print(event)
```

`fm.simulate()` yields `L3Event`s (order add / cancel / execute). It accepts
`order_flow`, `seed`, `mid` and `tick`; use `fm.Simulation` directly to inspect the
book between events.

## Development

```sh
pdm install -G dev
pdm run lint        # ruff
pdm run typecheck   # mypy --strict
pdm run test        # pytest
pdm run example     # examples/basic.py
pdm run facts       # examples/stylized_facts.py
```
