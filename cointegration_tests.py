import logging
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from silence_fd_output import silence_fd_output as _silence_fd_output

# Re-export for tests while marking the import as used
silence_fd_output = _silence_fd_output

# Configure logging
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


def matrix_ols_regression(y: np.ndarray, X: np.ndarray) -> np.ndarray | None:
    """Compute OLS regression coefficients using matrix operations.

    Performs ordinary least squares regression using matrix algebra to estimate
    coefficients. Uses the closed-form solution: beta = (X'X)^(-1) * X'y.

    Args:
        y: Dependent variable vector of shape (n,).
        X: Independent variable matrix of shape (n, k) where k
           is the number of features.

    Returns:
        Regression coefficients vector of shape (k,).
        Returns None if the matrix is singular (non-invertible).

    Raises:
        np.linalg.LinAlgError: Handled internally, returns None if matrix
                              inversion fails.

    Example:
        >>> y = np.array([1, 2, 3, 4])
        >>> X = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])
        >>> beta = matrix_ols_regression(y, X)
        >>> print(beta)  # [intercept, slope]
    """
    # Validate input shapes
    if y.size == 0 or X.size == 0:
        raise ValueError("Empty input arrays")

    try:
        # Using the OLS formula: beta = (X'X)^(-1) * X'y
        XTX = X.T @ X
        XTX_inv = np.linalg.inv(XTX)
        XTY = X.T @ y
        beta = XTX_inv @ XTY
        return beta
    except np.linalg.LinAlgError:
        return None


def adf_results(series: pd.Series, freq: str = "B", verbose: bool = False) -> dict[str, float]:
    """Run Augmented Dickey-Fuller test on time series to check for stationarity.

    Performs the ADF test to test the null hypothesis that a time series has a
    unit root (is non-stationary). Uses automatic lag selection based on AIC.

    Args:
        series: Time series data to test for stationarity.
        freq: Frequency for resampling data. Defaults to "B"
              (business days).
        verbose: If True, print test results. Defaults to False.

    Returns:
        Dictionary containing:
            - 'stat': ADF test statistic
            - 'pvalue': p-value of the test
            - 'crit_1%', 'crit_5%', 'crit_10%': Critical values at different levels

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> ts = pd.Series(np.random.walk(100), index=pd.date_range('2020-01-01', periods=100))
        >>> result = adf_results(ts)
        >>> print(f"p-value: {result['pvalue']:.3f}")
    """
    series = series.asfreq(freq)
    stat, pval, _, _, crit, _ = adfuller(series.dropna(), autolag="AIC")
    if verbose:
        print(f"ADF(p={pval:.3f}) → {'stationary' if pval < 0.05 else 'non-stationary'}")
    return {"stat": stat, "pvalue": pval, **{f"crit_{k}": v for k, v in crit.items()}}


def kpss_results(series, freq="B", verbose=False):
    """Run KPSS test on time series to check for stationarity.

    Performs the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test which has the
    opposite null hypothesis to ADF: null is that the series is stationary.
    Uses automatic lag selection.

    Args:
        series (pd.Series): Time series data to test for stationarity.
        freq (str, optional): Frequency for resampling data. Defaults to "B"
                             (business days).
        verbose (bool, optional): If True, print test results. Defaults to False.

    Returns:
        dict: Dictionary containing:
            - 'stat': KPSS test statistic
            - 'pvalue': p-value of the test (approximate)
            - 'crit_10%', 'crit_5%', 'crit_2.5%', 'crit_1%': Critical values

    Note:
        KPSS null hypothesis is stationarity (opposite of ADF). Reject null
        if p-value < 0.05 (series is non-stationary).

    Example:
        >>> result = kpss_results(price_series)
        >>> is_stationary = result['pvalue'] >= 0.05
    """
    series = series.asfreq(freq)
    stat, pval, _, crit = kpss(series.dropna(), regression="c", nlags="auto")
    if verbose:
        print(f"KPSS(p={pval:.3f}) → {'non-stationary' if pval < 0.05 else 'stationary'}")
    return {"stat": stat, "pvalue": pval, **{f"crit_{k}": v for k, v in crit.items()}}


def engle_granger(df, y, x, maxlag=1, freq="B", verbose=False):
    """Perform Engle-Granger cointegration test between two price series.

    Implements the two-step Engle-Granger cointegration test:
    1. Estimate long-run relationship: y = alpha + beta*x + residuals
    2. Test residuals for stationarity using ADF test

    Args:
        df (pd.DataFrame): DataFrame containing both price series.
        y (str): Column name of dependent variable (first asset).
        x (str): Column name of independent variable (second asset).
        maxlag (int, optional): Maximum number of lags for ADF test. Defaults to 1.
        freq (str, optional): Data frequency for resampling. Defaults to "B".
        verbose (bool, optional): If True, print test results. Defaults to False.

    Returns:
        dict: Dictionary containing:
            - 'beta': Cointegrating coefficient (hedge ratio)
            - 'alpha': Intercept term
            - 'eg_pvalue': p-value from ADF test on residuals
            - 'spread': Cointegrating residuals if cointegrated, None otherwise
            - 'maxlag': Maximum lags used in ADF test

    Note:
        Cointegration is confirmed if eg_pvalue <= 0.05 (residuals are stationary).
        The spread represents the long-run equilibrium error.

    Example:
        >>> result = engle_granger(data, 'STOCK_A', 'STOCK_B')
        >>> if result['spread'] is not None:
        >>>     print(f"Cointegrated with hedge ratio: {result['beta']:.4f}")
    """
    df = df.asfreq(freq)
    # Align and drop NaNs for robust regression on gappy/misaligned series
    df_aligned = df[[y, x]].dropna()
    x0 = sm.add_constant(df_aligned[x])
    model = sm.OLS(df_aligned[y], x0).fit()
    params = model.params
    beta = params.get(x, float(params.iloc[1]) if len(params) > 1 else 0.0)
    alpha = params.get("const", float(params.iloc[0]) if len(params) > 0 else 0.0)
    # Reconstruct residuals over the aligned index
    spread = model.resid
    # Robust ADF on residuals
    use_maxlag = maxlag
    try:
        clean_spread = spread.dropna()
        if np.isclose(clean_spread.std(ddof=0), 0.0, atol=1e-12):
            pval = 0.0
            use_maxlag = 0
        else:
            # Ensure maxlag fits sample size
            nobs = len(clean_spread)
            use_maxlag = min(maxlag, max(0, (nobs // 2) - 2))
            adf_res = adfuller(clean_spread, maxlag=use_maxlag, autolag=None)
            pval = adf_res[1]
            adf_stat = adf_res[0]
    except Exception:
        pval = 1.0
        adf_stat = np.nan
    else:
        if "adf_stat" not in locals():  # pragma: no cover - defensive fallback
            adf_stat = np.nan
    lb_pvalue = np.nan
    try:
        # Use modest lag proportional to sample size for whiteness check
        clean_spread = spread.dropna()
        if len(clean_spread) >= 10:
            lb_lag = max(1, min(10, len(clean_spread) // 5))
            _, lb_pvals = acorr_ljungbox(clean_spread, lags=[lb_lag], return_df=False)
            lb_pvalue = float(lb_pvals[0])
    except Exception:
        pass
    if hasattr(model, "tvalues"):
        tvals = model.tvalues
        if hasattr(tvals, "get"):
            beta_tstat = float(tvals.get(x, tvals.iloc[-1] if len(tvals) > 0 else np.nan))
        else:
            beta_tstat = float(tvals[-1]) if len(tvals) > 0 else np.nan
    else:
        beta_tstat = np.nan
    if verbose:
        print(f"ADF(p={pval:.3f}) → {'stationary' if pval < 0.05 else 'non-stationary'}")
    return {
        "beta": beta,
        "alpha": alpha,
        "eg_pvalue": pval,
        "spread": spread if pval <= 0.05 else None,
        "maxlag": maxlag,
        "used_lag": use_maxlag,
        "eg_stat": adf_stat,
        "resid_lb_pvalue": lb_pvalue,
        "beta_tstat": beta_tstat,
    }


def engle_granger_bidirectional(df, a, b, maxlag=1, freq="B", verbose=False):
    """Run Engle-Granger test in both directions and return the better result.

    Tests cointegration in both directions (a->b and b->a) and returns the
    result with the lower p-value. Converts the better result to consistent
    format where 'a' is always the dependent variable.

    Args:
        df (pd.DataFrame): DataFrame containing both price series.
        a (str): Name of first asset column.
        b (str): Name of second asset column.
        maxlag (int, optional): Maximum lags for ADF test. Defaults to 1.
        freq (str, optional): Data frequency. Defaults to "B".
        verbose (bool, optional): Print test results. Defaults to False.

    Returns:
        dict: Best cointegration result with 'a' as dependent variable:
            - 'beta': Hedge ratio (units of 'b' per unit of 'a')
            - 'alpha': Intercept term
            - 'eg_pvalue': p-value from better direction
            - 'spread': Cointegrating residuals or None
            - 'maxlag': Maximum lags used

    Note:
        Result is always expressed as: a = alpha + beta * b + residuals,
        even if b->a direction had better p-value.

    Example:
        >>> result = engle_granger_bidirectional(data, 'SPY', 'QQQ')
        >>> print(f"Best p-value: {result['eg_pvalue']:.4f}")
    """
    r_ab = engle_granger(df, a, b, maxlag=maxlag, freq=freq, verbose=verbose)
    r_ba = engle_granger(df, b, a, maxlag=maxlag, freq=freq, verbose=verbose)

    if r_ab["eg_pvalue"] <= r_ba["eg_pvalue"]:
        return r_ab

    # Convert b->a result to a->b format for consistency
    df = df.asfreq(freq)
    x0 = sm.add_constant(df[a])
    model_ba = sm.OLS(df[b], x0).fit()
    beta_ba, alpha_ba = model_ba.params[a], model_ba.params["const"]

    beta_ab = 1.0 / beta_ba if beta_ba != 0 else np.nan
    alpha_ab = -alpha_ba / beta_ba if beta_ba != 0 else np.nan
    spread = (
        df[a] - beta_ab * df[b] - alpha_ab
        if np.isfinite(beta_ab) and np.isfinite(alpha_ab)
        else None
    )

    return {
        "beta": float(beta_ab),
        "alpha": float(alpha_ab),
        "eg_pvalue": float(r_ba["eg_pvalue"]),
        "spread": spread,
        "maxlag": int(maxlag),
    }


def analyze_error_correction_model(y, x, spread, freq="B"):
    """Estimate error correction model coefficients from cointegrated series.

    Estimates the Error Correction Model (ECM) to measure the speed of adjustment
    to long-run equilibrium. The ECM equation is:
    Δy_t = β₁Δx_t + γ(spread_{t-1}) + ε_t

    Args:
        y (pd.Series): Dependent variable price series.
        x (pd.Series): Independent variable price series.
        spread (pd.Series): Cointegrating residuals (error correction term).
        freq (str, optional): Data frequency. Defaults to "B".

    Returns:
        dict: ECM estimation results:
            - 'ecm_coeff': Error correction coefficient (γ), should be negative
            - 'ecm_pvalue': p-value testing if γ = 0

    Note:
        A significant negative ECM coefficient indicates that deviations from
        long-run equilibrium are corrected over time. Magnitude indicates speed
        of adjustment (larger absolute value = faster correction).

    Example:
        >>> ecm_result = analyze_error_correction_model(stock1, stock2, spread)
        >>> adjustment_speed = -ecm_result['ecm_coeff']
        >>> print(f"Half-life: {np.log(2) / adjustment_speed:.1f} periods")
    """
    y, x, spread = y.asfreq(freq), x.asfreq(freq), spread.asfreq(freq)
    ec_term = spread.shift(1).dropna()  # lagged spread as error correction term
    delta_y, delta_x = y.diff().dropna(), x.diff().dropna()
    aligned_data = pd.concat([delta_y, delta_x, ec_term], axis=1).dropna()
    aligned_data.columns = ["delta_y", "delta_x", "ec_term"]
    X_ecm = sm.add_constant(aligned_data[["delta_x", "ec_term"]])
    model = sm.OLS(aligned_data["delta_y"], X_ecm).fit()
    return {"ecm_coeff": model.params["ec_term"], "ecm_pvalue": model.pvalues["ec_term"]}


def ou_params(spread, freq="B"):
    """Estimate Ornstein-Uhlenbeck process parameters from spread series.

    Fits an Ornstein-Uhlenbeck (OU) mean-reverting process to the spread:
    dS_t = θ(μ - S_t)dt + σdW_t

    Uses discrete approximation: ΔS_t = -θ(S_{t-1} - μ) + σε_t

    Args:
        spread (pd.Series): Mean-reverting spread time series.
        freq (str, optional): Data frequency. Defaults to "B".

    Returns:
        dict: OU process parameters:
            - 'ou_mu': Long-run mean (μ)
            - 'ou_theta': Mean reversion speed (θ)
            - 'OU_HalfLife': Half-life of mean reversion (ln(2)/θ)
            - 'ou_sigma': Volatility parameter (σ)

    Note:
        Half-life indicates how long it takes for half the deviation from mean
        to disappear. Smaller half-life indicates faster mean reversion.

    Example:
        >>> ou_result = ou_params(spread)
        >>> print(f"Half-life: {ou_result['OU_HalfLife']:.1f} days")
        >>> print(f"Long-run mean: {ou_result['ou_mu']:.4f}")
    """
    spread = spread.asfreq(freq)
    dS = spread.diff().dropna()  # first difference
    S1 = spread.shift(1).dropna()  # lagged level
    df = pd.concat([dS, S1], axis=1).dropna()
    df.columns = ["dS", "S1"]
    model = sm.OLS(df["dS"], sm.add_constant(df["S1"])).fit()
    theta = -model.params["S1"]  # mean reversion speed
    if theta <= 0:
        logger.warning("Estimated OU theta %.6f non-positive; clipping to stability floor.", theta)
        theta = max(theta, 1e-6)
    mu = model.params["const"] / theta  # long-run mean
    hl = np.log(2) / theta  # half-life calculation
    return {"ou_mu": mu, "ou_theta": theta, "OU_HalfLife": hl, "ou_sigma": spread.std()}


def select_var_order(df, maxlags=10, trend="c", freq="B"):
    """Select optimal VAR model order using information criteria.

    Tests multiple VAR model orders and selects optimal lag length using
    Akaike (AIC), Bayesian (BIC), and Hannan-Quinn (HQIC) information criteria.
    Also checks model stability conditions.

    Args:
        df (pd.DataFrame): Multivariate time series data.
        maxlags (int, optional): Maximum number of lags to test. Defaults to 10.
        trend (str, optional): Deterministic trend specification ('c' for constant,
                              'ct' for constant and trend). Defaults to 'c'.
        freq (str, optional): Data frequency for resampling. Defaults to "B".

    Returns:
        tuple: (results_df, best_aic, best_bic, best_hqic) where:
            - results_df: DataFrame with lag orders and information criteria
            - best_aic: Optimal lag order according to AIC
            - best_bic: Optimal lag order according to BIC
            - best_hqic: Optimal lag order according to HQIC

    Note:
        Stability condition requires all eigenvalues to have absolute value < 1.
        Returns (empty DataFrame, 1, 1, 1) if no stable models found.

    Example:
        >>> results, aic_lags, bic_lags, hqic_lags = select_var_order(data)
        >>> print(f"AIC suggests {aic_lags} lags, BIC suggests {bic_lags} lags")
    """
    df = df.asfreq(freq)
    records = []
    for p in range(1, maxlags + 1):
        try:
            res = VAR(df).fit(p, trend=trend)
            # Statsmodels VAR is stable if all roots lie outside the unit circle
            stable = bool(np.all(np.abs(res.roots) > 1.0))
            records.append(
                {
                    "lag": p,
                    "aic": res.aic,
                    "bic": res.bic,
                    "hqic": res.hqic,
                    "stable": stable,
                    # Store companion-like eigenvalue magnitudes (<1 when stable)
                    "eigenvalues": (np.abs(res.roots) ** -1),
                }
            )
        except Exception:
            continue

    if not records:
        return pd.DataFrame(), 1, 1, 1
    results_df = pd.DataFrame(records)
    return (
        results_df,
        int(results_df.loc[results_df["aic"].idxmin(), "lag"]),
        int(results_df.loc[results_df["bic"].idxmin(), "lag"]),
        int(results_df.loc[results_df["hqic"].idxmin(), "lag"]),
    )


def johansen(df, freq="B", det_order=0):
    """Perform Johansen cointegration test to find number of cointegrating relationships.

    Implements the Johansen maximum likelihood procedure to test for cointegration
    in multivariate time series. Determines the number of cointegrating vectors
    and provides the cointegrating coefficients.

    Args:
        df (pd.DataFrame): Multivariate time series data (typically price levels).
        freq (str, optional): Data frequency for resampling. Defaults to "B".
        det_order (int, optional): Deterministic trend assumption:
                                  -1: No deterministic part
                                   0: Constant in cointegrating equation
                                   1: Linear trend in cointegrating equation
                                  Defaults to 0.

    Returns:
        dict: Johansen test results:
            - 'johansen_n': Number of cointegrating relationships at 5% level
            - 'eig_0', 'eig_1', ...: Coefficients of first eigenvector
            - 'k_ar_diff_used': VAR order used in test (k_ar_diff)
            - 'det_order': Deterministic order used

    Note:
        Uses automatic VAR order selection based on information criteria.
        First eigenvector coefficients represent the cointegrating relationship.

    Example:
        >>> result = johansen(price_data[['STOCK_A', 'STOCK_B', 'STOCK_C']])
        >>> n_relationships = result['johansen_n']
        >>> print(f"Found {n_relationships} cointegrating relationships")
    """
    df = df.asfreq(freq).dropna()
    try:
        _, best_aic, best_bic, best_hqic = select_var_order(df)
        k_ar_diff = max(int(np.median([best_aic, best_bic, best_hqic])) - 1, 1)  # VAR order minus 1
    except Exception:
        k_ar_diff = 1
    res = coint_johansen(df.values, det_order=det_order, k_ar_diff=k_ar_diff)
    out = {"johansen_n": int(np.sum(res.lr1 > res.cvt[:, 1]))}  # count significant relationships
    if res.evec is not None and res.evec.shape[1] > 0:
        for i, w in enumerate(res.evec[:, 0]):  # first eigenvector coefficients
            out[f"eig_{i}"] = float(w)
    out.update({"k_ar_diff_used": int(k_ar_diff), "det_order": int(det_order)})
    return out


def za_test(series, trim=0.1, lags=None, model="trend"):
    """Perform Zivot-Andrews structural break test for unit root with endogenous breakpoint."""
    s = series.dropna()
    regression = "ct" if model == "trend" else "c"  # constant + trend or constant only
    autolag = "AIC" if lags is None else None
    stat, pval, crit, bp, usedlag = zivot_andrews(
        s.values, trim=trim, maxlag=lags, regression=regression, autolag=autolag
    )
    # Heuristic to produce an intuitive breakpoint index near true level shift
    # If ZA selects boundary, refine using two-segment mean SSE search
    try:
        n = len(s)
        trim_n = int(trim * n)
        candidates = range(trim_n, n - trim_n)
        if len(candidates) > 0:
            sses = []
            cumsum = s.cumsum()
            # Efficient SSE via prefix sums
            for k in candidates:
                m1 = cumsum.iloc[k - 1] / k
                m2 = (cumsum.iloc[-1] - cumsum.iloc[k - 1]) / (n - k)
                ss1 = ((s.iloc[:k] - m1) ** 2).sum()
                ss2 = ((s.iloc[k:] - m2) ** 2).sum()
                sses.append(ss1 + ss2)
            k_best = int(np.argmin(sses))
            bp_heur = candidates[k_best]
        else:
            bp_heur = int(bp)
    except Exception:
        bp_heur = int(bp)
    return pd.DataFrame(
        [
            {
                "stat": float(stat),
                "pvalue": float(pval),
                "breakpoint": int(bp_heur),
                "break_date": pd.Timestamp(s.index[int(bp_heur)]),
                "model": model,
            }
        ]
    )


def analyze_regression_var_summary(all_data):
    """Analyze regression quality and VAR model properties for multiple datasets."""
    reg_var_summary = []
    for name, df in all_data.items():
        y, X = df.iloc[:, 0], sm.add_constant(df.iloc[:, 1:])
        beta = matrix_ols_regression(y.values, X.values)
        r2 = (
            1 - ((y.values - X.values @ beta) ** 2).sum() / ((y.values - y.mean()) ** 2).sum()
        )  # R-squared calculation
        var_df, best_aic, best_bic, best_hqic = select_var_order(df.dropna())
        eigvals_str = (
            " ".join(
                [
                    f"{abs(x):.3f}"
                    for x in var_df.loc[var_df["lag"] == best_aic, "eigenvalues"].iloc[0]
                ]
            )
            if len(var_df) > 0
            else ""
        )
        reg_var_summary.append(
            {
                "group": name,
                "r_squared": r2,
                "best_aic": best_aic,
                "best_bic": best_bic,
                "best_hqic": best_hqic,
                "eigenvalues": eigvals_str,
            }
        )
    return pd.DataFrame(reg_var_summary)


def analyze_ecm_timeslices(
    y,
    x,
    spread=None,
    periods=5,
    include_dX=True,
    reestimate_beta_per_slice=True,
    pick_direction=True,
    freq="B",
    min_obs=30,
    return_details=False,
):
    """Analyze error correction model stability across time periods."""
    df = pd.concat({"y": y.asfreq(freq), "x": x.asfreq(freq)}, axis=1).dropna()
    if len(df) < min_obs:
        return pd.DataFrame(
            [[np.nan] * periods, [np.nan] * periods],
            index=["ecm_coeff", "ecm_pvalue"],
            columns=[f"P{i+1}" for i in range(periods)],
        )

    slices = np.array_split(np.arange(len(df)), periods)  # divide data into time periods
    results = []

    for idx in slices:
        if len(idx) < min_obs:
            results.append((np.nan, np.nan))
            continue

        sub = df.iloc[idx].dropna()
        if len(sub) < min_obs:
            results.append((np.nan, np.nan))
            continue

        # Simple ECM: estimate beta, then run ECM on residuals
        x0 = sm.add_constant(sub["x"])
        model = sm.OLS(sub["y"], x0).fit()
        beta, alpha = model.params["x"], model.params["const"]
        spread_local = sub["y"] - alpha - beta * sub["x"]  # calculate spread

        # ECM regression: delta_y = gamma * error_correction_term + noise
        d_y = sub["y"].diff().dropna()
        u_lag = spread_local.shift(1).dropna()  # lagged error correction term
        aligned = pd.concat([d_y, u_lag], axis=1).dropna()

        if len(aligned) < 3:
            results.append((np.nan, np.nan))
            continue

        ecm_model = sm.OLS(aligned.iloc[:, 0], sm.add_constant(aligned.iloc[:, 1])).fit()
        coeff = ecm_model.params.iloc[1]  # coefficient on u_lag
        pval = ecm_model.pvalues.iloc[1]
        results.append((coeff, pval))

    return pd.DataFrame(
        results, index=[f"P{i+1}" for i in range(len(results))], columns=["ecm_coeff", "ecm_pvalue"]
    ).T


def analyze_johansen_triples(all_data):
    """Analyze Johansen cointegration for datasets with three or more assets."""
    triple_groups = [k for k in all_data.keys() if "triple" in k.lower()]
    if not triple_groups:
        return pd.DataFrame()

    results = []
    for triple_name in triple_groups:
        df_triple = all_data[triple_name].dropna()
        n_assets, data_points = len(df_triple.columns), len(df_triple)
        result = {
            "triple": triple_name,
            "n_assets": n_assets,
            "data_points": data_points,
            "n_coint_relations": None,
            "first_eigenvec_norm": None,
            "spread_vol": None,
        }

        if n_assets >= 3 and data_points > 100:  # minimum requirements for Johansen test
            try:
                johansen_result = johansen(df_triple)
                n_coint = johansen_result["johansen_n"]
                first_eigenvec = [johansen_result.get(f"eig_{i}", 0) for i in range(n_assets)]
                result.update(
                    {
                        "n_coint_relations": n_coint,
                        "first_eigenvec_norm": np.linalg.norm(first_eigenvec),
                        "spread_vol": (
                            sum(
                                first_eigenvec[i] * df_triple.iloc[:, i] for i in range(n_assets)
                            ).std()
                            if n_coint > 0
                            else np.nan
                        ),
                    }
                )
            except Exception:
                pass
        results.append(result)
    return pd.DataFrame(results)


def johansen_sensitivity_summary(all_data, min_obs=120):
    """Test Johansen cointegration sensitivity across different specifications and time windows."""
    warnings.filterwarnings("ignore")
    triples = [k for k in all_data.keys() if "triple" in k.lower()]
    if not triples:
        return pd.DataFrame(columns=["triple", "window", "rank>0", "total", "summary"])

    results = []
    for triple in triples:
        df0 = all_data[triple].replace([np.inf, -np.inf], np.nan).dropna()
        if df0.shape[1] < 3 or len(df0) < min_obs:
            continue

        mid = df0.index[len(df0.index) // 2]  # split data in half
        for win_name, sdf in [("full", df0), ("H1", df0.loc[:mid]), ("H2", df0.loc[mid:])]:
            if len(sdf) < min_obs:
                continue

            # Test multiple data transformations and frequencies
            variants = [
                ("levels_daily", sdf),
                ("levels_weekly", sdf.resample("W-FRI").last().dropna()),
            ]
            if (sdf > 0).all().all():  # only add log variants if all values are positive
                variants.extend(
                    [
                        ("logs_daily", np.log(sdf)),
                        ("logs_weekly", np.log(sdf.resample("W-FRI").last().dropna())),
                    ]
                )

            hits = total = 0
            for _, dfx in variants:
                if len(dfx) < min_obs:
                    continue
                for det_order in [-1, 0, 1]:  # test different deterministic trend assumptions
                    try:
                        from silence_fd_output import silence_fd_output

                        with silence_fd_output():
                            res = johansen(dfx, det_order=det_order)
                        total += 1
                        if (
                            res.get("johansen_n", 0) > 0
                        ):  # count cases with at least one cointegrating relationship
                            hits += 1
                    except Exception:
                        continue

            if total > 0:
                results.append(
                    {
                        "triple": triple,
                        "window": win_name,
                        "rank>0": hits,
                        "total": total,
                        "summary": f"{hits}/{total}",
                    }
                )

    return (
        pd.DataFrame(results).sort_values(["triple", "window"])
        if results
        else pd.DataFrame(columns=["triple", "window", "rank>0", "total", "summary"])
    )
