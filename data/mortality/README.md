# Mortality Table Data

This directory contains the mortality table CSV files loaded by `BPADataLoader`.

## Files

| File | Content |
|---|---|
| `S3PMA.csv` | S3 Pensioner Males Amounts — base q_x rates, ages 16–120 |
| `S3PFA.csv` | S3 Pensioner Females Amounts — base q_x rates, ages 16–120 |
| `CMI_2023_M.csv` | CMI initial improvement rates for males, ages 16–120 |
| `CMI_2023_F.csv` | CMI initial improvement rates for females, ages 16–120 |

## Column format

All files use two columns: `age` (integer, 16–120) and the rate column.

| File | Rate column | Description |
|---|---|---|
| S3PMA.csv | `qx` | Annual mortality rate in (0, 1) |
| S3PFA.csv | `qx` | Annual mortality rate in (0, 1) |
| CMI_2023_M.csv | `initial_rate` | Annual improvement rate in (0, 1) |
| CMI_2023_F.csv | `initial_rate` | Annual improvement rate in (0, 1) |

## Licensing notice

**The values currently in these files are representative placeholders only.**

The official S3 series tables and CMI improvement model factors are published
by the Institute and Faculty of Actuaries (IFoA) / Continuous Mortality
Investigation (CMI) and are subject to CMI licensing terms.

Before using this model in production:
1. Obtain the official S3PMA and S3PFA tables from CMI (cmilimited.co.uk).
2. Obtain CMI_2023 (or later) initial improvement rates.
3. Replace the placeholder values in these files with the licensed data.
4. Record the CMI model year and LTR assumption used in the actuarial report.

## Format example

```csv
age,qx
16,0.000476
17,0.000520
...
120,0.574069
```
