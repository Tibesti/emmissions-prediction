# Project Notes — Emissions Prediction (Plant Bowen)

## Data sources (`bowen-plant-data/`)

| File | Source | Contents |
|---|---|---|
| `hourly-emissions-facility-aggregation-*.csv` | EPA CAMPD (facility-level hourly emissions) | Plant Bowen (GA, Facility ID 703): Gross Load, SO2/CO2/NOx mass, Heat Input |
| `POWER_Point_Hourly_20170101_20260331_034d12N_084d92W_LST.csv` | NASA POWER (MERRA-2 / CERES), point 34.1231 N, -84.9203 W, LST | Hourly weather: WS50M, WD50M, QV2M, PS, T2M, ALLSKY_SFC_SW_DWN |
| `combined-data.csv` | Derived (merge of the two above) | The working dataset |

## `combined-data.csv`

- Built by joining the emissions file to the weather file on date + hour (emissions rows drive the output; every row found a weather match).
- **Columns (13):** Date (DD-MM-YYYY), Hour, Heat Input (mmBtu), Gross Load (MW), SO2 Mass (lbs), CO2 Mass (short tons), NOx Mass (lbs), WS50M, WD50M, QV2M, PS, T2M, ALLSKY_SFC_SW_DWN
- **Rows:** 77,896 hourly records, 01-01-2017 00:00 → 31-03-2026 23:00.
- The source EPA file's "Steam Load (1000 lb/hr)" column was entirely empty (Bowen doesn't report it), so it was dropped.
- No empty cells; no duplicate or out-of-order timestamps.

## Known data gaps

Missing hours: yes — **3,152 hours are absent, spread over 20 gaps**. The file spans 01-01-2017 00:00 through 31-03-2026 23:00, which would be 81,048 hours if continuous, but only 77,896 are present. There are no duplicate or out-of-order timestamps — the gaps are the only irregularity.

The gaps cluster in 2019–2022 (these come from the original EPA emissions file, so they almost certainly correspond to plant/unit outages when nothing was reported). Weather data exists for all gap hours; only emissions values are missing. The full list, in order:

| Missing hours | Gap starts after | Data resumes at |
|---:|---|---|
| 197 | 04-01-2019 08:00 | 12-01-2019 14:00 |
| 260 | 16-10-2019 07:00 | 27-10-2019 04:00 |
| 3 | 28-10-2019 07:00 | 28-10-2019 11:00 |
| 63 | 28-10-2019 15:00 | 31-10-2019 07:00 |
| 186 | 09-01-2020 12:00 | 17-01-2020 07:00 |
| 384 | 24-01-2020 12:00 | 09-02-2020 13:00 |
| 26 | 13-03-2020 17:00 | 14-03-2020 20:00 |
| 178 | 06-05-2020 23:00 | 14-05-2020 10:00 |
| 213 | 27-05-2020 23:00 | 05-06-2020 21:00 |
| 1 | 05-06-2020 22:00 | 06-06-2020 00:00 |
| 43 | 07-06-2020 03:00 | 08-06-2020 23:00 |
| 582 | 23-09-2020 22:00 | 18-10-2020 05:00 |
| 2 | 21-09-2021 08:00 | 21-09-2021 11:00 |
| 314 | 19-11-2021 22:00 | 03-12-2021 01:00 |
| 12 | 03-12-2021 06:00 | 03-12-2021 19:00 |
| 47 | 04-12-2021 14:00 | 06-12-2021 14:00 |
| 562 | 14-12-2021 23:00 | 07-01-2022 10:00 |
| 12 | 09-10-2022 09:00 | 09-10-2022 22:00 |
| 14 | 23-03-2024 18:00 | 24-03-2024 09:00 |
| 53 | 24-03-2024 10:00 | 26-03-2024 16:00 |

**Modeling note:** for time-series work, either treat contiguous blocks separately or impute the emissions gaps.

## Weather variable reference (NASA POWER)

- `WS50M` — wind speed at 50 m (m/s)
- `WD50M` — wind direction at 50 m (degrees)
- `QV2M` — specific humidity at 2 m (g/kg)
- `PS` — surface pressure (kPa)
- `T2M` — air temperature at 2 m (°C)
- `ALLSKY_SFC_SW_DWN` — all-sky surface shortwave downward irradiance (Wh/m²)
- NASA POWER uses `-999` for missing values (none present in the combined data's date range).
