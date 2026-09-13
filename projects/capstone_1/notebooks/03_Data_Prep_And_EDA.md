# 03 — Data Prep and EDA

Five findings, each recorded as an assumption made **before** running the query, then checked against what was actually measured. R3 requires at least one assumption to be contradicted — assumption 2 is, clearly.

All numbers below come from `app.data.seal.development_frame()` and `data/processed/flights_features.parquet` (the same 5,696,742-row development set notebook 01 verifies), and from `data/manifest.json`.

---

## 1. Overall positive rate

**Assumption (before querying):** roughly 20% of US domestic flights arrive 15+ minutes late, in line with widely cited BTS on-time performance figures.

**Measured:** 0.2218 (22.18%) across the full development set.

**Verdict:** Confirmed — close to the assumed ballpark.

---

## 2. Positive rate stability across months

**Assumption (before querying):** the late-arrival rate is roughly flat across the 10-month development window — weather and other drivers of lateness average out over a season.

**Measured (per period):**

| Period | Rows | Positive rate |
|---|---|---|
| 2025-07 | 612,811 | 28.89% |
| 2025-08 | 593,733 | 22.56% |
| 2025-09 | 558,328 | 16.63% |
| 2025-10 | 601,570 | 20.32% |
| 2025-11 | 555,296 | 20.56% |
| 2025-12 | 571,924 | 26.77% |
| 2026-01 | 517,222 | 20.78% |
| 2026-02 | 502,396 | 19.76% |
| 2026-03 | 592,256 | 24.32% |
| 2026-04 | 591,206 | 20.20% |

**Verdict:** **Contradicted.** The range is 16.63% (September) to 28.89% (July) — a 12-point spread, not flat. Summer (July) and December are the worst months, September the best. This is exactly why every metric in this project is reported as a *lift over that fold's own baseline* rather than a bare score: the same PR-AUC means something different in a 17%-positive month than a 29%-positive one.

---

## 3. Row count after cleaning

**Assumption (before querying):** dropping cancelled and diverted flights plus rows with a missing label removes on the order of 2-3% of the raw acquired rows.

**Measured:**

- Raw acquired rows (12 months, `data/manifest.json`): 7,043,316
- Development rows (10 months, cleaned): 5,696,742
- Holdout rows (2 months, cleaned, sealed): 1,199,541
- Removed by cleaning: 147,033 (**2.09%** of raw)

**Verdict:** Confirmed — 2.09% falls right in the assumed 2-3% range.

---

## 4. Rare-event feature shares

**Assumption (before querying):** red-eye departures (00:00-05:59) and holiday-window flights are both small minorities of the schedule — under 10% each.

**Measured:**

- Red-eye share: 2.96%
- Holiday-window share: 12.04%
- Weekend share: 27.57%

**Verdict:** **Partially contradicted.** Red-eye is well under 10% as assumed (2.96%), but the holiday-window feature (within 2 days of one of the 11 tracked holidays) is 12.04% — over the assumed ceiling, because the holiday list includes multi-day windows (Thanksgiving, Christmas/New Year) that each cover several dates.

An unplanned finding from the same query: red-eye flights have a **lower** positive rate than the overall average (9.24% vs 22.18%), even though the model's error analysis (see the caveat in `app/serving/predict.py`) flags early-morning departures as where the model is *least reliable*. Those are two different things — red-eyes are rarely late, but when the model is wrong about one, it's wrong confidently — which is exactly why that caveat exists.

---

## 5. Route concentration and rotation depth

**Assumption (before querying):** with 6,000+ distinct routes in the data, the busiest single route carries well under 1% of all flights, and most aircraft fly no more than 3-4 legs in a day.

**Measured:**

- Unique routes: 6,996
- Top route (ORD-LGA): 9,128 flights out of 5,696,742 — **0.16%** of all development rows
- Leg-number distribution: leg 1 (25.4%), leg 2 (23.3%), leg 3 (19.7%), leg 4 (15.3%), leg 5 (9.0%), leg 6 (5.2%), leg 7 (1.4%), leg 8+ (0.6%)

**Verdict:** Confirmed on both counts — the busiest route is well under 1% (0.16%), and the leg-number distribution drops off sharply after leg 4 (roughly 84% of flights are legs 1-4). This also explains why `route_late_rate_n` (how much history exists for a route) earns its place as its own feature rather than being assumed reliable everywhere: most routes sit nowhere near the top-route volume, so a route-level rate for a thin route needs the smoothing `app/features/history.py` applies.
