import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Any, Tuple

# ---------- CONFIG ----------
CSV_PATH   = "/FinTech/DerwentData_TS/xtra/IndexData.csv"
ID_COL     = "Publication Number"
DATE_COL   = "Application Date"

INDICATORS = [
    "Count of Cited Refs - Patent",
    "Count of Cited Refs - Non-patent",
    "Count of Citing Patents",
    "DWPI Count of Family Members",
    "DWPI Count of Family Countries/Regions",
    "Assignee Count",
    "Inventor Count",
    "Claims Count",
    "Legal Years Remaining",
    "IPC Count",
]

TRAIN_START = pd.Timestamp("2000-01-01")
TRAIN_END   = pd.Timestamp("2022-12-31")
SCORE_START = pd.Timestamp("2023-01-01")
SCORE_END   = pd.Timestamp("2024-12-31")

RESAMPLE_RULE = "M"     # month-end (canonical pandas alias)
MAX_K_FACTORS = 2
FACTOR_ORDER  = 0
ERROR_COV     = "scalar"
TOP_PCT       = 0.05

# Optionally stabilize heavy-tailed counts (leave empty to keep raw)
LOG1P_VARS: list[str] = []  # e.g., ["Count of Citing Patents", "Claims Count", "IPC Count"]

# ---------- UTILS ----------
def winsorize(s: pd.Series, p: float = 0.01) -> pd.Series:
    # Safe quantiles even with all-NaN/constant series
    if s.notna().sum() == 0:
        return s
    lo, hi = s.quantile(p), s.quantile(1 - p)
    if pd.isna(lo) or pd.isna(hi) or lo == hi:
        return s
    return s.clip(lo, hi)

def robust_standardize_train(ts_train: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    mean = ts_train.mean(skipna=True)
    std = ts_train.std(ddof=1, skipna=True).replace(0, np.nan)
    return (ts_train - mean) / std, mean, std

def try_fit_dfa(mod: sm.tsa.DynamicFactor) -> Any:
    start_params = None
    # EM init if available
    try:
        if hasattr(mod, "fit_em"):
            res0 = mod.fit_em(maxiter=100, disp=False)
            start_params = getattr(res0, "params", None)
    except Exception:
        start_params = None
    # Model-provided starts
    if start_params is None:
        try:
            start_params = mod.start_params
        except Exception:
            start_params = None
    # Optimizer cascade
    for method, kw in [
        ("lbfgs", dict(maxiter=2000)),
        ("powell", dict(maxiter=400)),
        ("nm", dict(maxiter=2000)),
    ]:
        try:
            return mod.fit(start_params=start_params, method=method, disp=False, **kw)
        except Exception:
            continue
    # Last-ditch: powell then lbfgs with its params
    try:
        res1 = mod.fit(start_params=start_params, method="powell", maxiter=400, disp=False)
        return mod.fit(start_params=getattr(res1, "params", None), method="lbfgs", maxiter=2000, disp=False)
    except Exception as e:
        raise RuntimeError(f"DFA fit failed with all methods: {e}")

def _safe_converged(res: Any) -> bool:
    ret = getattr(res, "mle_retvals", None)
    if isinstance(ret, dict) and "converged" in ret:
        return bool(ret["converged"])
    return bool(getattr(res, "converged", False))

def _load_name_candidates(f_idx: int, var_idx: int, var_name: str | None) -> list[str]:
    # Common historical parameter name formats across statsmodels versions
    cands = [
        f"loading.f{f_idx}.y{var_idx}",
        f"loading.f{f_idx}.{var_idx}",
        f"loading.L[{var_idx-1},{f_idx-1}]",
    ]
    if var_name:
        cands += [
            f"loading.f{f_idx}.{var_name}",
            f"loading.f{f_idx}.y{var_name}",
            f"loading.{var_name}.f{f_idx}",
        ]
    return cands

def extract_loadings(res: Any, indicators: list[str], k: int, endog_names: list[str]) -> np.ndarray:
    # Map parameter name -> value robustly
    param_names = list(getattr(res, "param_names", []))
    params = getattr(res, "params", None)
    if params is None or len(param_names) != len(params):
        try:
            s = res.params  # pandas Series in some versions
            param_names = list(s.index)
            params = s.values
        except Exception:
            pass
    if params is None:
        raise RuntimeError("Could not access fitted parameter vector.")
    name2param = {pn: float(params[i]) for i, pn in enumerate(param_names)}

    load_mat = np.zeros((len(indicators), k), dtype=float)
    for f in range(1, k + 1):
        for i, _col in enumerate(indicators, start=1):
            var_name = None
            try:
                if isinstance(endog_names, (list, tuple)) and len(endog_names) >= i:
                    var_name = endog_names[i - 1]
            except Exception:
                var_name = None
            candidates = _load_name_candidates(f, i, var_name)
            val = None
            for cname in candidates:
                if cname in name2param:
                    val = name2param[cname]
                    break
            if val is None:
                # Fuzzy fallback
                prefix = f"loading.f{f}"
                matches = [kname for kname in name2param
                           if kname.startswith(prefix) and (
                               kname.endswith(f".y{i}") or
                               kname.endswith(f".{i}") or
                               (var_name and var_name in kname)
                           )]
                if matches:
                    val = name2param[matches[0]]
            if val is None:
                raise RuntimeError(f"Missing loading for factor {f}, variable index {i} ({_col}).")
            load_mat[i - 1, f - 1] = val
    return load_mat

def rowwise_weighted_sum_nan_safe(Z: np.ndarray, weights: np.ndarray) -> np.ndarray:
    W = weights.reshape(1, -1)
    mask = ~np.isnan(Z)
    numer = np.nan_to_num(Z) * W
    denom = (W * mask).sum(axis=1, keepdims=True)
    denom[denom == 0] = np.nan
    return (numer.sum(axis=1, keepdims=True) / denom).ravel()

# ---------- MAIN PIPELINE ----------
def main():
    # --- Load & validate ---
    df = pd.read_csv(CSV_PATH)
    if DATE_COL not in df.columns or ID_COL not in df.columns:
        raise ValueError(f"Missing {DATE_COL} or {ID_COL} in CSV.")
    missing = [c for c in INDICATORS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing indicator columns: {missing}")

    # --- Types & basic cleaning ---
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).copy()

    for c in INDICATORS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = winsorize(df[c])
        if c in LOG1P_VARS:
            with np.errstate(invalid="ignore"):
                df[c] = np.log1p(df[c])

    # --- Monthly time series for DFA (mean) ---
    ts = (
        df.set_index(DATE_COL)[INDICATORS]
          .resample(RESAMPLE_RULE)
          .mean()    # keep NaNs; state-space handles missing
    )

    ts_train = ts.loc[TRAIN_START:TRAIN_END]
    if ts_train.shape[0] < 18:
        raise ValueError(f"Too few monthly periods in training window {TRAIN_START.date()}–{TRAIN_END.date()}.")

    # Missingness sanity (optional)
    nan_rate = ts_train.isna().mean()
    high_nan = nan_rate[nan_rate > 0.30]
    if not high_nan.empty:
        print(f"[WARN] High missingness in train monthly series: {dict(high_nan.round(2))}")

    ts_train_std, ts_mean, ts_std = robust_standardize_train(ts_train)

    # --- Model selection by BIC ---
    bic, best = {}, {}
    for k in range(1, MAX_K_FACTORS + 1):
        mod = sm.tsa.DynamicFactor(
            ts_train_std,
            k_factors=k,
            factor_order=FACTOR_ORDER,
            error_cov_type=ERROR_COV,
            enforce_stationarity=True,
            enforce_invertibility=True,
        )
        try:
            res = try_fit_dfa(mod)
            converged = _safe_converged(res)
            rbic = getattr(res, "bic", np.inf)
            finite_bic = np.isfinite(rbic)
            if converged and finite_bic:
                bic[k] = float(rbic)
                best[k] = res
                print(f"[INFO] k={k}: BIC={rbic:.2f}, converged={converged}")
            else:
                print(f"[WARN] k={k}: skipped (converged={converged}, finite_bic={finite_bic})")
        except Exception as e:
            print(f"[WARN] k={k}: fitting failed -> {e}")

    if not bic:
        raise RuntimeError("All candidate DFA fits failed or did not converge with finite BIC. "
                           "Action: reduce MAX_K_FACTORS, simplify model, or lengthen TRAIN window.")

    optimal_k = min(bic, key=bic.get)
    res = best[optimal_k]
    print(f"[OK] Selected k={optimal_k} with BIC={bic[optimal_k]:.2f}")

    # --- Loadings -> indicator weights ---
    try:
        endog_names = list(getattr(res.model, "endog_names", []))
        if isinstance(endog_names, str):
            endog_names = [endog_names]
    except Exception:
        endog_names = []
    load_mat = extract_loadings(res, INDICATORS, optimal_k, endog_names=endog_names)

    var_contrib = (load_mat ** 2).sum(axis=1)
    total = var_contrib.sum()
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError("Invalid variance contributions from loadings.")
    weights = var_contrib / total

    # --- Export weights (all 10 indicators) ---
    weights_df = pd.DataFrame({
        "Indicator": INDICATORS,
        "VarianceContribution": var_contrib,
        "Weight": weights
    }).sort_values("Weight", ascending=False).reset_index(drop=True)

    # --- Scoring 2023–2024 on TRAIN scale (row-wise) ---
    df_train_rows = df[df[DATE_COL].between(TRAIN_START, TRAIN_END, inclusive="both")]
    train_row_mean = df_train_rows[INDICATORS].mean(skipna=True)
    train_row_std  = df_train_rows[INDICATORS].std(ddof=1, skipna=True).replace(0, np.nan)

    df_eval = df[df[DATE_COL].between(SCORE_START, SCORE_END, inclusive="both")].copy()
    if df_eval.empty:
        raise ValueError(f"No rows in scoring window {SCORE_START.date()}–{SCORE_END.date()}.")

    Zcols = []
    for c in INDICATORS:
        z = (df_eval[c].astype(float) - train_row_mean[c]) / train_row_std[c]
        zname = f"Z::{c}"
        df_eval[zname] = z
        Zcols.append(zname)

    Z = df_eval[Zcols].to_numpy(dtype=float)
    qi = rowwise_weighted_sum_nan_safe(Z, weights)
    df_eval["QualityIndex"] = qi

    df_eval = df_eval.sort_values("QualityIndex", ascending=False).reset_index(drop=True)
    k_top = int(np.ceil(TOP_PCT * len(df_eval))) if len(df_eval) else 0
    df_eval["Top5pct"] = False
    if k_top > 0:
        df_eval.loc[:k_top - 1, "Top5pct"] = True

    # --- Save & print summaries ---
    train_tag = f"{TRAIN_START.year}_{TRAIN_END.year}"
    weights_path = f"dfa_weights_{train_tag}.csv"
    scores_path  = "quality_scores_2023_24.csv"

    weights_df.to_csv(weights_path, index=False)
    df_eval[[ID_COL, DATE_COL, "QualityIndex", "Top5pct"]].to_csv(scores_path, index=False)

    print("\n=== DFA Summary ===")
    print(f"Train window: {TRAIN_START.date()} to {TRAIN_END.date()}")
    print(f"Score window: {SCORE_START.date()} to {SCORE_END.date()}")
    print(f"Candidates tried (k): {list(bic.keys())}")
    print(f"Optimal k: {optimal_k}  |  Converged: {_safe_converged(res)}")
    print(f"Saved weights -> {os.path.abspath(weights_path)}")
    print(f"Saved scores  -> {os.path.abspath(scores_path)}")

    print("\nIndicator weights (variance share; higher = more influence):")
    print(weights_df.to_string(index=False))

    print("\nWeights in original indicator order:")
    for ind, w in zip(INDICATORS, weights):
        print(f"{ind}: {w:.6f}")

    print("\nTop 10 by QualityIndex:")
    print(df_eval[[ID_COL, "QualityIndex", "Top5pct"]].head(10).to_string(index=False))


if __name__ == "__main__":
    # Cleaner console output
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 50)
    main()
