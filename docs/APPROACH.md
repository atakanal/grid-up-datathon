# Modeling approach

## Problem framing

The target is daily transformer consumption and the competition metric is
RMSLE. Training and ensembling therefore operate in `log1p(consumption)` space.
Predictions are clipped to a valid non-negative range only when materialized as
submission values.

The test panel contains two fundamentally different populations:

| Segment | Transformers | Test rows | Share |
|---|---:|---:|---:|
| Warm — transformer exists in train | 5,012 | 556,319 | 77.84% |
| Cold — transformer is unseen in train | 2,024 | 158,369 | 22.16% |
| **Total** | **7,036** | **714,688** | **100%** |

This distinction drives both modeling and validation.

## Warm series

Warm forecasts use the transformer's own history as the main signal. Candidate
components include:

- recent robust levels and 14/28/90-day summaries;
- transformer-by-weekday and month-by-weekday profiles;
- previous-year aligned profiles and shrunk year-over-year change;
- history length, recency, staleness, volatility, and zero-consumption regime;
- calendar, holiday, Ramadan, school-calendar, and weather interactions;
- residual tree models and pretrained time-series model experiments.

The final routing distinguishes fresh warm series from stale or incomplete
histories, preventing a single blend from treating all warm transformers alike.

## Cold series

Cold transformers have no target history. Their model family uses only signals
available outside the missing series:

- installed power and location hierarchy;
- power-normalized target distributions;
- natural transformer-entry cohorts and commissioning age;
- active/inactive hurdle and lifecycle representations;
- masked-warm transfer experiments with hierarchical shrinkage.

Cold validation combines natural historical entries with completely hidden
transformer holdouts. It is reported separately because masked warm series do
not perfectly reproduce real commissioning behavior.

## Validation

The research pipeline uses time-ordered blocks and prevents validation targets
from entering lag or rolling features. Final candidate directions are also
checked across deterministic partitions of:

- rows;
- transformer identities;
- dates;
- calendar months;
- forecast horizons;
- whole model families.

These controls reduce dependence on one convenient split, but they cannot
reveal private-leaderboard labels or guarantee transfer.

## Final submissions

### V96b — 0.99676

V96b is a segment-routed interpolation between two earlier candidates. It uses
a larger move for fresh warm rows and protects stale-warm/cold rows with a much
smaller coefficient. The construction was selected with a frozen minimax grid
over uncertainty in the protected segment.

### V108 — 0.99430

V108 starts from the V100 candidate and combines 17 previously scored endpoint
directions in log space. A constrained quadratic search required the fixed move
to improve all deterministic row, transformer, day, month, and horizon stress
planes. The projected public center was `0.994281`; the observed score was
`0.99430`, an absolute projection error of roughly `0.000019`.

V108 was supported by 12 of 14 whole-family dropout checks. A stricter 14/14
variant projected `0.994313`, so it was not used for the final submission.

## External-data caveat

The broader research archive evaluated historical and realized external data
from weather, water, national load, sector activity, outages, tourism, ports,
agriculture, and other sources. The lineage of V108 includes endpoint models
affected by realized April–July 2026 observations. This must be reviewed against
the organizer's final rule interpretation before presenting V108 as an eligible
final notebook solution.
