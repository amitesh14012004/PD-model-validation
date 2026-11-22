"""Run the full independent validation of the champion PD model.

    python run_validation.py

Outputs: outputs/validation_report.md, outputs/validation_results.csv, outputs/figures/*.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve

from src import validation as V
from src.data import TARGET, clean, generate_loans, split_dev_oot
from src.model import ChallengerPD, ChampionPD

OUT = Path("outputs")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def md_table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False) if hasattr(df, "to_markdown") else df.to_string(index=False)


def main():
    # 1. data ------------------------------------------------------------------
    raw = generate_loans()
    results = V.test_data_quality(raw)
    df = clean(raw)
    train, test, oot = split_dev_oot(df)
    dev = pd.concat([train, test])

    # 2. fit models ------------------------------------------------------------
    champ, chal = ChampionPD().fit(train), ChallengerPD().fit(train)
    p = {k: {"champ": champ.predict_proba(d), "chal": chal.predict_proba(d)}
         for k, d in {"train": train, "test": test, "oot": oot}.items()}

    # 3. tests -----------------------------------------------------------------
    r, vifs = V.test_conceptual_soundness(champ)
    results += r

    r, auc_tr = V.test_discrimination("train", train[TARGET], p["train"]["champ"])
    results += r
    r, auc_te = V.test_discrimination("test", test[TARGET], p["test"]["champ"], auc_tr)
    results += r
    r, auc_oot = V.test_discrimination("OOT", oot[TARGET], p["oot"]["champ"], auc_te)
    results += r

    r, grades_dev = V.test_calibration("test", test[TARGET], p["test"]["champ"], p["train"]["champ"])
    results += r
    r, grades_oot = V.test_calibration("OOT", oot[TARGET], p["oot"]["champ"], p["train"]["champ"])
    results += r

    r, csi = V.test_stability(train, oot, p["train"]["champ"], p["oot"]["champ"])
    results += r
    results += V.test_benchmark(oot[TARGET], p["oot"]["champ"], p["oot"]["chal"])

    sens = V.test_sensitivity(champ, chal, oot)
    sens_ok = bool(sens["direction_ok"].all())
    results.append(V.row("Sensitivity", "PD responds in expected direction to all shocks",
                         "all pass" if sens_ok else "FAIL", "all pass",
                         "GREEN" if sens_ok else "RED"))
    overall = V.overall_rating(results)

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT / "validation_results.csv", index=False)

    # 4. figures ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name, y, pp in [("Test", test[TARGET], p["test"]["champ"]),
                        ("OOT", oot[TARGET], p["oot"]["champ"]),
                        ("Challenger OOT", oot[TARGET], p["oot"]["chal"])]:
        fpr, tpr, _ = roc_curve(y, pp)
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "roc.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, g, title in [(axes[0], grades_dev, "Test sample"), (axes[1], grades_oot, "Out-of-time")]:
        ax.plot(g["grade"], g["avg_PD"], "o-", label="Predicted PD")
        ax.plot(g["grade"], g["default_rate"], "s--", label="Observed default rate")
        ax.set(xlabel="Rating grade (1 = best)", title=f"Calibration by grade: {title}")
    axes[0].set_ylabel("Rate")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(FIG / "calibration.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    s = pd.Series(csi).sort_values()
    ax.barh(s.index, s.values)
    ax.axvline(0.10, color="orange", ls="--", label="0.10 (amber)")
    ax.axvline(0.25, color="red", ls="--", label="0.25 (red)")
    ax.set(xlabel="PSI (dev vs OOT)", title="Characteristic stability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "csi.png", dpi=150)
    plt.close(fig)

    av = V.auc_by_vintage(df, champ.predict_proba(df))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(av.index.astype(str), av.values, "o-")
    ax.axvline(list(av.index.astype(str)).index("2023Q1"), color="grey", ls=":")
    ax.set(ylabel="AUC", title="Champion AUC by vintage (dotted line = start of OOT)")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(FIG / "auc_by_vintage.png", dpi=150)
    plt.close(fig)

    # 5. report ----------------------------------------------------------------
    counts = res_df["status"].value_counts().to_dict()
    for c in ("GREEN", "AMBER", "RED"):
        counts.setdefault(c, 0)
    fmt = lambda d: d.round(4)
    bench = next(r for r in results if r["area"] == "Benchmarking")
    uplift_material = bench["status"] != "GREEN"
    rec3 = ("3. **Model form**: the gradient-boosting challenger materially outperforms the champion "
            "out of time, so add non-linear / interaction terms (e.g. utilization bins, DTI x "
            "utilization) before moving to a black-box model."
            if uplift_material else
            "3. **Model form**: the gradient-boosting challenger does not outperform the champion "
            "out of time (AUC uplift CI does not exceed zero), so the simpler, more explainable "
            "logistic form is retained. Re-run this benchmark at each recalibration.")
    findings = [r for r in results if r["status"] in ("RED", "AMBER")]
    lines = [
        "# Independent Validation Report: Retail PD Model (Champion: Logistic Regression)",
        "",
        f"**Overall rating: {overall}**",
        "",
        f"Tests run: {len(res_df)} | GREEN {counts['GREEN']} | AMBER {counts['AMBER']} | RED {counts['RED']}",
        "",
        f"Samples: train {len(train):,} | test {len(test):,} | out-of-time (2023Q1+) {len(oot):,} loans. "
        f"Default rate: dev {dev[TARGET].mean():.2%}, OOT {oot[TARGET].mean():.2%}.",
        "",
        "## 1. Summary of findings (AMBER / RED)",
        "",
        "| Area | Test | Result | Threshold | Status |",
        "|---|---|---|---|---|",
    ]
    lines += [f"| {r['area']} | {r['test']} | {r['value']} | {r['threshold']} | {r['status']} |"
              for r in findings]
    lines += ["", "## 2. Full test results", "",
              "| Area | Test | Result | Threshold | Status |", "|---|---|---|---|---|"]
    lines += [f"| {r['area']} | {r['test']} | {r['value']} | {r['threshold']} | {r['status']} |"
              for r in results]
    lines += ["", "## 3. Calibration by rating grade (out-of-time)", "",
              md_table(fmt(grades_oot)), "",
              "![calibration](figures/calibration.png)", "",
              "## 4. Stability", "", "![csi](figures/csi.png)", "",
              "![auc](figures/auc_by_vintage.png)", "",
              "## 5. Sensitivity / stress (OOT portfolio, mean PD change)", "",
              md_table(fmt(sens)), "",
              "## 6. Discrimination", "", "![roc](figures/roc.png)", "",
              "## 7. Recommendations", "",
              "1. **Recalibrate** the PD scale on recent vintages (intercept / Platt scaling) and add a "
              "macro or through-the-cycle adjustment: the model under-predicts defaults out of time.",
              "2. **Monitoring**: put PSI, AUC by vintage, and mean-PD-vs-default-rate on a quarterly "
              "dashboard with the thresholds used here as triggers.",
              rec3,
              "4. **Data**: fix upstream duplicate and out-of-range-age records; document the median "
              "imputation for income and employment length.",
              "5. **Limitations**: synthetic data with a known generating process; thresholds are "
              "illustrative and should be set per the institution's model risk policy.",
              ]
    (OUT / "validation_report.md").write_text("\n".join(lines))
    print(f"Overall: {overall}")
    print(res_df[["area", "test", "value", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
