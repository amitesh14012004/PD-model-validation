# PD Model Validation Framework

An independent validation of a retail **Probability of Default (PD)** model, written the way a model risk team would do it: data quality, conceptual soundness, discrimination, calibration, stability, benchmarking against a challenger, sensitivity analysis, and a RAG-rated validation report with recommendations.

The model under validation (the *champion*) is a logistic regression. A gradient-boosting model serves as the *challenger*. The whole pipeline runs with one command and produces a written report.

```bash
pip install -r requirements.txt
python run_validation.py      # writes outputs/validation_report.md, results CSV and figures
pytest                        # unit tests for PSI, KS, Gini, Hosmer-Lemeshow
```

## What it tests

| Area | Tests |
|---|---|
| Data quality | Duplicates, missing values, out-of-range values |
| Conceptual soundness | Coefficient sign vs business expectation, significance, multicollinearity (VIF) |
| Discrimination | AUC with bootstrap CI, Gini, KS; train / test / out-of-time; AUC drop vs development |
| Calibration | Mean PD vs observed default rate, Hosmer-Lemeshow, binomial test per rating grade, Brier score, rank-ordering across grades |
| Stability | PSI of the score, characteristic PSI (CSI) per driver, AUC by vintage |
| Benchmarking | Paired bootstrap of AUC difference, champion vs challenger |
| Sensitivity | Income, DTI, utilization and delinquency shocks; PD must move in the expected direction |

Every test returns a GREEN / AMBER / RED status against thresholds in `src/validation.py` (illustrative, adjustable per model materiality), which roll up into an overall rating.

## Data

The portfolio is **synthetic**: 60,000 loans over 24 quarterly vintages (2019Q1-2024Q4) with a known data-generating process. That lets the framework prove it detects what it should. The generator plants:

- population drift and a macro shock from 2023Q1 (the out-of-time window)
- a non-linear utilization threshold and a DTI x utilization interaction
- missing values, duplicate rows and impossible ages

To validate real data, replace `generate_loans()` in `src/data.py` with any loan-level dataset with the same columns (for example Lending Club).

## Results from the included run

Development data is 2019-2022; out-of-time is 2023-2024.

| Result | Finding |
|---|---|
| Discrimination | Holds up: AUC 0.816 (test), 0.824 (OOT); KS 0.50 |
| Calibration | **Fails out of time**: mean PD 3.76% vs observed default rate 4.87% (-22.7%); Hosmer-Lemeshow p < 0.0001; grades 9 and 10 under-predict (grade 10: 12.0% predicted vs 15.8% observed) |
| Stability | Score PSI 0.166 (AMBER): population has drifted, no single driver breaches 0.10 |
| Benchmarking | Challenger does not beat the champion (AUC uplift -0.007, CI excludes zero on the negative side), so the simpler model is retained |
| Overall | **RED**: the model still ranks borrowers well but its PD levels are no longer reliable. Recalibrate on recent vintages and add monitoring triggers |

Full detail, tables and charts: [`outputs/validation_report.md`](outputs/validation_report.md).

![calibration](outputs/figures/calibration.png)

## Project layout

```
src/data.py          synthetic portfolio, cleaning, dev / out-of-time split
src/model.py         champion (logistic regression) and challenger (gradient boosting)
src/validation.py    all validation tests and RAG thresholds
run_validation.py    runs everything and writes the report and figures
tests/               unit tests for the metric implementations
```

## Limitations

- Synthetic data: results show the framework works, not how a real portfolio behaves.
- Thresholds are illustrative; a real validation sets them under the institution's model risk policy (e.g. SR 11-7 / internal MRM standards).
- No LGD/EAD, IFRS 9 / CECL staging, or Basel capital calculations; PD only.

## Possible extensions

- Run on a real public credit dataset
- Add Kupiec / Christoffersen tests (already in my market-risk dashboard) for a market-risk VaR validation module
- Add a monitoring dashboard in Streamlit
