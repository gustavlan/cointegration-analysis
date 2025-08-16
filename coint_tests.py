import warnings
import os
import contextlib
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, InterpolationWarning, zivot_andrews
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.var_model import VAR
from pykalman import KalmanFilter

warnings.filterwarnings(
    'ignore',
    message='The test statistic is outside of the range of p-values available',
    category=InterpolationWarning
)

@contextlib.contextmanager
def silence_fd_output():
    """Silence low-level stdout/stderr (FD 1/2) temporarily (tiny helper)."""
    try:
        saved_out, saved_err = os.dup(1), os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        os.close(devnull)
        try:
            yield
        finally:
            os.dup2(saved_out, 1)
            os.dup2(saved_err, 2)
            os.close(saved_out)
            os.close(saved_err)
    except Exception:
        # If anything goes wrong, do nothing but still proceed
        yield


def matrix_ols_regression(y, X):
    """Performs OLS regression using matrix algebra with numpy."""
    try:
        # Using the OLS formula: beta = (X'X)^(-1) * X'y
        XTX = X.T @ X
        XTX_inv = np.linalg.inv(XTX)
        XTY = X.T @ y
        beta = XTX_inv @ XTY
        return beta
    except np.linalg.LinAlgError:
        return None


def adf_results(series, freq="B", verbose=False):
    """Returns ADF test outputs."""
    series = series.asfreq(freq)
    stat, pval, _, _, crit, _ = adfuller(series.dropna(), autolag='AIC')
    # Micro-decision print for ADF
    if verbose:
        print(f"ADF(p={pval:.3f}) → {'stationary' if pval < 0.05 else 'non-stationary'}")
    return {
        'stat': stat,
        'pvalue': pval,
        **{f'crit_{k}': v for k, v in crit.items()}
    }


def kpss_results(series, freq="B", verbose=False):
    """Returns KPSS test outputs."""
    series = series.asfreq(freq)
    stat, pval, _, crit = kpss(series.dropna(), regression='c', nlags='auto')
    if verbose:
        print(f"KPSS(p={pval:.3f}) → {'non-stationary' if pval < 0.05 else 'stationary'}")
    return {
        'stat': stat,
        'pvalue': pval,
        **{f'crit_{k}': v for k, v in crit.items()}
    }

def engle_granger(df, y, x, maxlag=1, freq="B", verbose=False):
    """Returns hedge ratio and ADF p-value on residuals, plus spread if cointegrated."""
    # Ensure frequency is set
    df = df.asfreq(freq)
    x0 = sm.add_constant(df[x])
    model = sm.OLS(df[y], x0).fit()
    beta = model.params[x]
    alpha = model.params['const']
    spread = model.resid  # Use model residuals to include intercept
    pval = adfuller(spread.dropna(), maxlag=maxlag, autolag=None)[1]
    if verbose:
        print(f"ADF(p={pval:.3f}) → {'stationary' if pval < 0.05 else 'non-stationary'}")
    return {'beta': beta, 'alpha': alpha, 'eg_pvalue': pval, 'spread': spread if pval <= .05 else None, 'maxlag': maxlag}


def engle_granger_bidirectional(df, a, b, maxlag=1, freq="B", verbose=False):
    """
    Run EG in both directions; keep the residual with the lower p-value.
    Returns results w.r.t. 'a' as dependent variable.
    """
    r_ab = engle_granger(df, a, b, maxlag=maxlag, freq=freq, verbose=verbose)
    r_ba = engle_granger(df, b, a, maxlag=maxlag, freq=freq, verbose=verbose)
    
    if r_ab['eg_pvalue'] <= r_ba['eg_pvalue']:
        return r_ab
    
    # Convert back to a spread defined as a - beta*b - alpha
    # Need to recompute spread with proper intercept
    df = df.asfreq(freq)
    x0 = sm.add_constant(df[a])
    model_ba = sm.OLS(df[b], x0).fit()
    beta_ba = model_ba.params[a]
    alpha_ba = model_ba.params['const']
    
    beta_ab = 1.0 / beta_ba if beta_ba != 0 else np.nan
    alpha_ab = -alpha_ba / beta_ba if beta_ba != 0 else np.nan
    spread = df[a] - beta_ab * df[b] - alpha_ab if np.isfinite(beta_ab) and np.isfinite(alpha_ab) else None
    
    return {
        'beta': float(beta_ab), 
        'alpha': float(alpha_ab),
        'eg_pvalue': float(r_ba['eg_pvalue']), 
        'spread': spread, 
        'maxlag': int(maxlag)
    }


def analyze_error_correction_model(y, x, spread, freq="B"):
    """
    Error-Correction Model (ECM) returns the coefficient and p-value of the error term.
    """
    y, x, spread = y.asfreq(freq), x.asfreq(freq), spread.asfreq(freq)
    ec_term = spread.shift(1).dropna() # lag spread by 1 period
    delta_y, delta_x = y.diff().dropna(), x.diff().dropna() # difference to original
    aligned_data = pd.concat([delta_y, delta_x, ec_term], axis=1).dropna() # Align all series to the same index
    aligned_data.columns = ['delta_y', 'delta_x', 'ec_term']
    X_ecm = sm.add_constant(aligned_data[['delta_x', 'ec_term']]) # regress on error term
    y_ecm = aligned_data['delta_y']
    model = sm.OLS(y_ecm, X_ecm).fit()
    ec_coeff = model.params['ec_term']
    ec_pvalue = model.pvalues['ec_term']
    
    return {'ecm_coeff': ec_coeff, 'ecm_pvalue': ec_pvalue}

def ou_params(spread, freq="B"):
    """Returns OU mu, theta, and half-life."""
    spread = spread.asfreq(freq)
    dS = spread.diff().dropna()
    S1 = spread.shift(1).dropna()
    df = pd.concat([dS, S1], axis=1).dropna()
    df.columns = ['dS', 'S1']
    model = sm.OLS(df['dS'], sm.add_constant(df['S1'])).fit()
    theta = -model.params['S1']
    mu = model.params['const'] / theta
    hl = np.log(2) / theta
    sigma_eq = spread.std()
    return {'ou_mu': mu, 'ou_theta': theta, 'OU_HalfLife': hl, 'ou_sigma': sigma_eq}


def johansen(df, freq="B", det_order=0):
    """
    Run Johansen cointegration test with adaptive lag selection.
    """
    df = df.asfreq(freq).dropna()
    try:
        var_table, best_aic, best_bic, best_hqic = select_var_order(df)
        p = int(np.median([best_aic, best_bic, best_hqic]))
        k_ar_diff = max(p - 1, 1)
    except Exception:
        k_ar_diff = 1
    res = coint_johansen(df.values, det_order=det_order, k_ar_diff=k_ar_diff)
    n = np.sum(res.lr1 > res.cvt[:, 1])
    vec = res.evec[:, 0]
    out = {'johansen_n': int(n)}  # trace test at 5%
    # Leading eigenvector
    if res.evec is not None and res.evec.shape[1] > 0:
        v = res.evec[:, 0]
        for i, w in enumerate(v):
            out[f'eig_{i}'] = float(w)
    out['k_ar_diff_used'] = int(k_ar_diff)
    out['det_order'] = int(det_order)
    return out

def kalman_hedge(df, y, x, freq="B"):
    """Returns dynamic beta and spread series."""
    df = df.asfreq(freq)
    yv, xv = df[y].values, df[x].values
    kf = KalmanFilter(
        n_dim_obs=1, n_dim_state=2,
        initial_state_mean=np.zeros(2),
        initial_state_covariance=np.eye(2),
        observation_matrices=np.vstack([xv, np.ones_like(xv)]).T[:, None, :],
        transition_matrices=np.eye(2),
        transition_covariance=1e-4 * np.eye(2)
    )
    state_means, _ = kf.filter(yv)
    beta = pd.Series(state_means[:, 0], index=df.index, name='kf_beta')
    alpha = pd.Series(state_means[:, 1], index=df.index, name='kf_alpha')
    spread = df[y] - beta * df[x] - alpha
    return {'kf_beta': beta, 'kf_spread': spread}


def select_var_order(df, maxlags=10, trend='c', freq="B"):
    """
    Multivariate time series df, fit VAR(p) for p=1..maxlags
    """
    # Ensure frequency is set
    df = df.asfreq(freq)
    records = []
    for p in range(1, maxlags+1):
        model = VAR(df)
        try:
            res = model.fit(p, trend=trend)
        except Exception:
            # Skip failing lag orders silently
            continue

        # companion‐matrix roots = VAR stability eigenvalues
        eigs = res.roots
        is_stable = all(abs(r) < 1 for r in eigs)

        records.append({
            'lag': p,
            'aic': res.aic,
            'bic': res.bic,
            'hqic': res.hqic,
            'stable': is_stable,
            'eigenvalues': eigs
        })

    results_df = pd.DataFrame(records)
    best_aic   = int(results_df.loc[results_df['aic'].idxmin(),  'lag'])
    best_bic   = int(results_df.loc[results_df['bic'].idxmin(),  'lag'])
    best_hqic  = int(results_df.loc[results_df['hqic'].idxmin(), 'lag'])

    return results_df, best_aic, best_bic, best_hqic


def subsample_cointegration(df, y, x, n_periods=4, min_obs=30):
    """
    Split df into n_periods equal length slices run Engle Granger + ECM on each
    """
    # compute period boundaries
    idx = df.index.sort_values()
    boundaries = pd.to_datetime(
        np.linspace(idx[0].value, idx[-1].value, n_periods + 1).astype('int64')
    )
    
    records = []
    for i in range(n_periods):
        start, end = boundaries[i], boundaries[i+1]
        slice_df = df.loc[start:end].dropna()
        if len(slice_df) < min_obs:
            continue
        
        # Engle Granger on this slice
        eg   = engle_granger(slice_df, y, x)
        beta = eg['beta']
        p_eg = eg['eg_pvalue']
        
        # If cointegrated, get ECM coeff; else NaNs
        if eg['spread'] is not None:
            ecm_res = analyze_error_correction_model(
                slice_df[y], slice_df[x], eg['spread']
            )
            coeff, p_ec = ecm_res['ecm_coeff'], ecm_res['ecm_pvalue']
        else:
            coeff, p_ec = np.nan, np.nan
        
        records.append({
            'period_start': start.date(),
            'period_end':   end.date(),
            'beta':         beta,
            'eg_pvalue':    p_eg,
            'ecm_coeff':    coeff,
            'ecm_pvalue':   p_ec
        })
    
    return pd.DataFrame(records)

def summarize_cointegration_tests(all_data: dict):
    """Perform cointegration tests more efficiently."""
    records = []
    
    for group, df in all_data.items():
        for asset in df.columns:
            series = df[asset]
            adf_res = adf_results(series)
            kpss_res = kpss_results(series)
            records.append({
                'group': group, 'asset': asset,
                **adf_res, **kpss_res
            })
        
        n_assets = len(df.columns)
        if n_assets == 2:
            y, x = df.columns
            # Combine EG test and related calculations
            eg = engle_granger(df, y, x)
            records.append({
                'group': group,
                'test': 'Engle-Granger',
                'beta': eg['beta'],
                'eg_pvalue': eg['eg_pvalue']
            })

            # Matrix OLS as an alternative hedge ratio
            X0 = sm.add_constant(df[x])
            mbeta = matrix_ols_regression(df[y].values, X0.values)
            if mbeta is not None:
                records.append({
                    'group': group,
                    'test': 'Matrix-OLS',
                    'const': mbeta[0],
                    'slope': mbeta[1]
                })

            # If cointegrated, estimate OU and ECM
            if eg.get('spread') is not None:
                ou = ou_params(eg['spread'])
                records.append({'group': group, 'test': 'OU', **ou})

                ecm = analyze_error_correction_model(df[y], df[x], eg['spread'])
                records.append({'group': group, 'test': 'ECM', **ecm})

            # Kalman filter summary for dynamic hedge ratio
            kf = kalman_hedge(df, y, x)
            records.append({
                'group': group,
                'test': 'Kalman',
                'kf_beta_mean': kf['kf_beta'].mean()
            })

        elif n_assets == 3:
            # Johansen test for triple cointegration
            jres = johansen(df)
            records.append({
                'group': group,
                'test': 'Johansen',
                **jres
            })

    return pd.DataFrame(records)


def run_pair_backtests(
    selected,
    all_data,
    z_list,
    train_months,
    test_months,
    step_months
):
    """
    Run nested cross-validation backtests on cointegrated pairs.
    """
    # Local import to avoid circular dependency at module import time
    from backtest import nested_cv
    results = []

    for pair in selected:
        df = all_data[pair]
        y, x = df.columns

        # Engle–Granger: obtain stationary spread
        eg = engle_granger(df, y, x)
        spread = eg['spread'].dropna()

        # Estimate OU parameters on spread
        ou = ou_params(spread)
        mu_e = ou['ou_mu']
        sigma_eq = ou['ou_sigma']
        df_pair = pd.DataFrame({'spread': spread})

        # Run nested cross‑validation
        cv_df = nested_cv(
            df_pair,
            spread_col='spread',
            mu_e=mu_e,
            sigma_eq=sigma_eq,
            z_list=z_list,
            train_months=train_months,
            test_months=test_months,
            step_months=step_months
        )

        # Add pair column and append to results
        cv_df['pair'] = pair
        results.append(cv_df)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def multi_subperiod_stability_check(df, y_col, x_col, n_periods=5, maxlag=1):
    """
    Split into n_periods windows and run full ECM per window.
    """
    def process_subperiod(sub, y_col, x_col, maxlag):
        """Helper function to process a single subperiod."""
        try:
            eg = engle_granger(sub, y_col, x_col, maxlag=maxlag, verbose=False)
            spread = eg.get('spread')
            if spread is None or spread.isna().all():
                return None
            ecm = analyze_error_correction_model(sub[y_col], sub[x_col], spread)
            return {
                'Period': f"{sub.index.min():%Y-%m-%d} to {sub.index.max():%Y-%m-%d}",
                'β_ec': ecm.get('ecm_coeff', np.nan),
                'p_value': ecm.get('ecm_pvalue', np.nan),
                'Cointegrated?': 'Yes' if (ecm.get('ecm_pvalue', 1.0) < 0.05 and ecm.get('ecm_coeff', 0.0) < 0.0) else 'No',
                'n_obs': len(sub)
            }
        except Exception:
            return None

    df_clean = df[[y_col, x_col]].dropna()
    n = len(df_clean)
    w = n // n_periods if n_periods > 0 else n
    
    rows = [
        process_subperiod(df_clean.iloc[i*w:(i+1)*w], y_col, x_col, maxlag)
        for i in range(n_periods)
        if len(df_clean.iloc[i*w:(i+1)*w]) >= max(60, w // 2)
    ]
    
    # Filter out None results and create DataFrame
    rows = [row for row in rows if row is not None]
    results_df = pd.DataFrame(rows)
    overall_stable = bool((results_df['Cointegrated?'] == 'Yes').all()) if not results_df.empty else False
    summary = {
        'overall_stable': overall_stable,
        'n_windows': int(len(results_df)),
        'n_yes': int((results_df['Cointegrated?'] == 'Yes').sum()) if not results_df.empty else 0
    }
    return results_df, summary


def za_test(series, trim=0.1, lags=None, model='trend'):
    """Zivot–Andrews test with a single break; returns index and timestamp."""
    s = series.dropna()
    regression = {'level': 'c', 'trend': 'ct'}[model]
    autolag = ('AIC', None)[int(lags is not None)]
    stat, pval, crit, bp, usedlag = zivot_andrews(
        s.values, trim=trim, maxlag=lags, regression=regression, autolag=autolag
    )
    bp = int(bp)
    bp_date = pd.Index(s.index)[bp]
    return pd.DataFrame([{
        'stat': float(stat),
        'pvalue': float(pval),
        'breakpoint': bp,
        'break_date': pd.Timestamp(bp_date),
        'model': model
    }])


def analyze_regression_var_summary(all_data):
    """
    Analyze regression R^2 and VAR model order selection for all data groups.
    """
    reg_var_summary = []
    for name, df in all_data.items():
        y = df.iloc[:,0]
        X = sm.add_constant(df.iloc[:,1:])
        beta = matrix_ols_regression(y.values, X.values)
        preds = X.values @ beta
        r2 = 1 - ((y.values - preds)**2).sum() / ((y.values - y.mean())**2).sum()

        var_df, best_aic, best_bic, best_hqic = select_var_order(df.dropna())
        eigvals = var_df.loc[var_df['lag']==best_aic, 'eigenvalues'].iloc[0]
        eigvals_magnitude = [np.abs(x) for x in eigvals]
        eigvals_str = ' '.join([f'{abs(x):.3f}' for x in eigvals_magnitude])

        reg_var_summary.append({
            'group': name,
            'r_squared': r2,
            'best_aic': best_aic,
            'best_bic': best_bic,
            'best_hqic': best_hqic,
            'eigenvalues': eigvals_str
        })
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
    return_details=False
):
    """
    ECM-by-periods analysis with per-slice cointegrating regression and optional direction selection.
    """
    df = pd.concat({'y': y.asfreq(freq), 'x': x.asfreq(freq)}, axis=1).dropna()
    if len(df) == 0:
        out = pd.DataFrame([[np.nan]*periods, [np.nan]*periods],
                           index=['ecm_coeff','ecm_pvalue'],
                           columns=[f'P{i+1}' for i in range(periods)])
        return (out, []) if return_details else out

    # Pre-split indices
    slices = np.array_split(np.arange(len(df)), periods)

    def ecm_direction(sub_df, dep, reg):
        """
        Compute ECM for dep|reg within a slice:
        """
        # Cointegrating regression
        Xc = sm.add_constant(sub_df[reg])
        ols = sm.OLS(sub_df[dep], Xc).fit()
        beta = float(ols.params[reg])
        alpha = float(ols.params['const'])
        spread_local = sub_df[dep] - (alpha + beta * sub_df[reg])

        d_dep = sub_df[dep].diff()
        d_reg = sub_df[reg].diff()
        u_lag = spread_local.shift(1)

        z = pd.concat({'d_dep': d_dep, 'd_reg': d_reg, 'u_lag': u_lag}, axis=1).dropna()
        if len(z) < 3:
            return {'coeff': np.nan, 'pvalue': np.nan, 'tvalue': np.nan,
                    'beta': beta, 'alpha': alpha, 'n': int(len(z))}

        X_cols = []
        if include_dX:
            X_cols.append('d_reg')
        X_cols.append('u_lag')

        X = sm.add_constant(z[X_cols])
        res = sm.OLS(z['d_dep'], X).fit()

        coeff = float(res.params.get('u_lag', np.nan))
        pval  = float(res.pvalues.get('u_lag', np.nan))
        tval  = float(res.tvalues.get('u_lag', np.nan))
        return {'coeff': coeff, 'pvalue': pval, 'tvalue': tval,
                'beta': beta, 'alpha': alpha, 'n': int(len(z))}

    results = []
    details = []
    for idx in slices:
        if len(idx) == 0:
            results.append((np.nan, np.nan))
            details.append({'direction': None})
            continue

        sub = df.iloc[idx].dropna()
        if len(sub) < min_obs:
            results.append((np.nan, np.nan))
            details.append({'direction': None, 'n_slice': int(len(sub))})
            continue

        # Direction 1: y | x
        if reestimate_beta_per_slice:
            d1 = ecm_direction(sub, dep='y', reg='x')
        else:
            # Optional path: use provided full-sample spread for y|x
            if spread is None:
                d1 = ecm_direction(sub, dep='y', reg='x')
            else:
                sp = pd.Series(spread, index=df.index).iloc[idx]
                d_y = sub['y'].diff()
                d_x = sub['x'].diff()
                u_lag = pd.Series(sp, index=sub.index).shift(1)
                z = pd.concat({'d_dep': d_y, 'd_reg': d_x, 'u_lag': u_lag}, axis=1).dropna()
                if len(z) >= 3:
                    X_cols = ['d_reg'] if include_dX else []
                    X_cols.append('u_lag')
                    X = sm.add_constant(z[X_cols])
                    res = sm.OLS(z['d_dep'], X).fit()
                    d1 = {
                        'coeff': float(res.params.get('u_lag', np.nan)),
                        'pvalue': float(res.pvalues.get('u_lag', np.nan)),
                        'tvalue': float(res.tvalues.get('u_lag', np.nan)),
                        'beta': np.nan, 'alpha': np.nan, 'n': int(len(z))
                    }
                else:
                    d1 = {'coeff': np.nan, 'pvalue': np.nan, 'tvalue': np.nan,
                          'beta': np.nan, 'alpha': np.nan, 'n': int(len(z))}

        # Direction 2: x | y (flip)
        d2 = ecm_direction(sub, dep='x', reg='y') if pick_direction else None

        # Select direction
        chosen = d1
        chosen_dir = 'y_on_x'
        if pick_direction and d2 is not None:
            cands = []
            if np.isfinite(d1['coeff']) and np.isfinite(d1['pvalue']):
                cands.append(('y_on_x', d1))
            if np.isfinite(d2['coeff']) and np.isfinite(d2['pvalue']):
                cands.append(('x_on_y', d2))

            if cands:
                negs = [(name, d) for name, d in cands if d['coeff'] < 0]
                if negs:
                    # Prefer smaller p-value among negative coeffs; tiebreak by more negative coeff
                    name, d = sorted(negs, key=lambda nd: (nd[1]['pvalue'], nd[1]['coeff']))[0]
                else:
                    # If none negative, pick lower p-value
                    name, d = sorted(cands, key=lambda nd: nd[1]['pvalue'])[0]
                chosen, chosen_dir = d, name

        results.append((chosen['coeff'], chosen['pvalue']))
        details.append({
            'direction': chosen_dir,
            'y_on_x': d1,
            'x_on_y': d2,
            'n_slice': int(len(sub))
        })

    cols = [f'P{i+1}' for i in range(len(results))]
    summary = pd.DataFrame(results, index=cols, columns=['ecm_coeff', 'ecm_pvalue']).T

    return (summary, details) if return_details else summary


def analyze_johansen_triples(all_data):
    """
    Perform Johansen multivariate cointegration test for triple groups.
    """
    triple_groups = [k for k in all_data.keys() if 'triple' in k.lower()]

    if triple_groups:
        johansen_summary = []
        
        for triple_name in triple_groups:
            df_triple = all_data[triple_name].dropna()
            
            if len(df_triple.columns) >= 3 and len(df_triple) > 100:
                try:
                    johansen_result = johansen(df_triple)
                    n_coint = johansen_result['johansen_n']
                    first_eigenvec = [johansen_result.get(f'eig_{i}', 0) for i in range(len(df_triple.columns))]
                    
                    # Create spread using first eigenvector
                    if n_coint > 0:
                        spread_triple = sum(first_eigenvec[i] * df_triple.iloc[:, i] for i in range(len(first_eigenvec)))
                        spread_vol = spread_triple.std()
                    else:
                        spread_vol = np.nan
                    
                    johansen_summary.append({
                        'triple': triple_name,
                        'n_assets': len(df_triple.columns),
                        'n_coint_relations': n_coint,
                        'first_eigenvec_norm': np.linalg.norm(first_eigenvec),
                        'spread_vol': spread_vol,
                        'data_points': len(df_triple)
                    })
                    
                except Exception:
                    johansen_summary.append({
                        'triple': triple_name,
                        'n_assets': len(df_triple.columns),
                        'n_coint_relations': None,
                        'first_eigenvec_norm': None,
                        'spread_vol': None,
                        'data_points': len(df_triple)
                    })
        
        return pd.DataFrame(johansen_summary)
    else:
        return pd.DataFrame()  # Empty DataFrame if no triples


def johansen_sensitivity_summary(all_data, min_obs=120):
    """
    Minimal Johansen sensitivity summary.
    Counts how many simple specifications yield rank>0 per window (full/H1/H2).
    Uses existing johansen() and suppresses verbose solver output locally.
    Returns a DataFrame with columns: triple, window, rank>0, total, summary.
    """
    # Suppress noisy warnings during internal VAR/Johansen fits
    warnings.filterwarnings('ignore', message='.*SVD did not converge.*')
    warnings.filterwarnings('ignore', message='.*Critical values are only available.*')

    rows = []
    try:
        triples = [k for k in all_data.keys() if 'triple' in k.lower()]
    except Exception:
        return pd.DataFrame(columns=['triple', 'window', 'rank>0', 'total', 'summary'])

    for triple in triples:
        try:
            df0 = all_data[triple].replace([np.inf, -np.inf], np.nan).dropna().astype(float)
        except Exception:
            continue

        if df0.shape[1] < 3 or len(df0) < min_obs:
            continue

        idx = df0.index
        mid = idx[len(idx)//2]
        windows = {
            'full': df0,
            'H1': df0.loc[:mid],
            'H2': df0.loc[mid:]
        }

        for win, sdf in windows.items():
            sdf = sdf.dropna()
            if len(sdf) < min_obs:
                continue

            weekly = sdf.resample('W-FRI').last().dropna()
            variants = [('levels_daily', sdf), ('levels_weekly', weekly)]
            if (sdf > 0).all().all():
                variants += [('logs_daily', np.log(sdf)), ('logs_weekly', np.log(weekly))]

            det_orders = [-1, 0, 1]
            hits = 0
            tot = 0

            for _, dfx in variants:
                if len(dfx) < min_obs:
                    continue
                for det in det_orders:
                    try:
                        # Silence low-level solver noise printed to stdout/stderr
                        with silence_fd_output():
                            res = johansen(dfx, det_order=det)
                        r = res.get('johansen_n', 0) or 0
                        tot += 1
                        if isinstance(r, (int, np.integer)) and r > 0:
                            hits += 1
                    except Exception:
                        # Skip failing specs (ill-conditioned windows, singularities, etc.)
                        continue

            if tot > 0:
                rows.append({
                    'triple': triple,
                    'window': win,
                    'rank>0': hits,
                    'total': tot,
                    'summary': f"{hits}/{tot}"
                })

    summary = pd.DataFrame(rows).sort_values(['triple', 'window']) if rows else pd.DataFrame(columns=['triple','window','rank>0','total','summary'])
    return summary
