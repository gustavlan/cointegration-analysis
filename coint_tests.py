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

warnings.filterwarnings('ignore', category=InterpolationWarning)

@contextlib.contextmanager
def silence_fd_output():
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
    series = series.asfreq(freq)
    stat, pval, _, _, crit, _ = adfuller(series.dropna(), autolag='AIC')
    if verbose:
        print(f"ADF(p={pval:.3f}) → {'stationary' if pval < 0.05 else 'non-stationary'}")
    return {'stat': stat, 'pvalue': pval, **{f'crit_{k}': v for k, v in crit.items()}}

def kpss_results(series, freq="B", verbose=False):
    series = series.asfreq(freq)
    stat, pval, _, crit = kpss(series.dropna(), regression='c', nlags='auto')
    if verbose:
        print(f"KPSS(p={pval:.3f}) → {'non-stationary' if pval < 0.05 else 'stationary'}")
    return {'stat': stat, 'pvalue': pval, **{f'crit_{k}': v for k, v in crit.items()}}

def engle_granger(df, y, x, maxlag=1, freq="B", verbose=False):
    df = df.asfreq(freq)
    x0 = sm.add_constant(df[x])
    model = sm.OLS(df[y], x0).fit()
    beta, alpha = model.params[x], model.params['const']
    spread = model.resid
    pval = adfuller(spread.dropna(), maxlag=maxlag, autolag=None)[1]
    if verbose:
        print(f"ADF(p={pval:.3f}) → {'stationary' if pval < 0.05 else 'non-stationary'}")
    return {'beta': beta, 'alpha': alpha, 'eg_pvalue': pval, 'spread': spread if pval <= .05 else None, 'maxlag': maxlag}


def engle_granger_bidirectional(df, a, b, maxlag=1, freq="B", verbose=False):
    r_ab = engle_granger(df, a, b, maxlag=maxlag, freq=freq, verbose=verbose)
    r_ba = engle_granger(df, b, a, maxlag=maxlag, freq=freq, verbose=verbose)
    
    if r_ab['eg_pvalue'] <= r_ba['eg_pvalue']:
        return r_ab
    
    df = df.asfreq(freq)
    x0 = sm.add_constant(df[a])
    model_ba = sm.OLS(df[b], x0).fit()
    beta_ba, alpha_ba = model_ba.params[a], model_ba.params['const']
    
    beta_ab = 1.0 / beta_ba if beta_ba != 0 else np.nan
    alpha_ab = -alpha_ba / beta_ba if beta_ba != 0 else np.nan
    spread = df[a] - beta_ab * df[b] - alpha_ab if np.isfinite(beta_ab) and np.isfinite(alpha_ab) else None
    
    return {'beta': float(beta_ab), 'alpha': float(alpha_ab), 'eg_pvalue': float(r_ba['eg_pvalue']), 
            'spread': spread, 'maxlag': int(maxlag)}


def analyze_error_correction_model(y, x, spread, freq="B"):
    y, x, spread = y.asfreq(freq), x.asfreq(freq), spread.asfreq(freq)
    ec_term = spread.shift(1).dropna()
    delta_y, delta_x = y.diff().dropna(), x.diff().dropna()
    aligned_data = pd.concat([delta_y, delta_x, ec_term], axis=1).dropna()
    aligned_data.columns = ['delta_y', 'delta_x', 'ec_term']
    X_ecm = sm.add_constant(aligned_data[['delta_x', 'ec_term']])
    model = sm.OLS(aligned_data['delta_y'], X_ecm).fit()
    return {'ecm_coeff': model.params['ec_term'], 'ecm_pvalue': model.pvalues['ec_term']}

def ou_params(spread, freq="B"):
    spread = spread.asfreq(freq)
    dS = spread.diff().dropna()
    S1 = spread.shift(1).dropna()
    df = pd.concat([dS, S1], axis=1).dropna()
    df.columns = ['dS', 'S1']
    model = sm.OLS(df['dS'], sm.add_constant(df['S1'])).fit()
    theta = -model.params['S1']
    mu = model.params['const'] / theta
    hl = np.log(2) / theta
    return {'ou_mu': mu, 'ou_theta': theta, 'OU_HalfLife': hl, 'ou_sigma': spread.std()}


def select_var_order(df, maxlags=10, trend='c', freq="B"):
    """Select optimal VAR order using multiple information criteria."""
    try:
        df = df.asfreq(freq).dropna()
        model = VAR(df)
        results = model.select_order(maxlags=maxlags)
        order_table = pd.DataFrame({
            'AIC': [results.aic[lag] for lag in range(len(results.aic))],
            'BIC': [results.bic[lag] for lag in range(len(results.bic))],
            'HQIC': [results.hqic[lag] for lag in range(len(results.hqic))]
        })
        return (
            order_table,
            results.selected_orders['aic'],
            results.selected_orders['bic'],
            results.selected_orders['hqic']
        )
    except Exception:
        return pd.DataFrame(), 1, 1, 1


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
    df = df.asfreq(freq)
    records = []
    for p in range(1, maxlags+1):
        model = VAR(df)
        try:
            res = model.fit(p, trend=trend)
            eigs = res.roots
            records.append({
                'lag': p, 'aic': res.aic, 'bic': res.bic, 'hqic': res.hqic,
                'stable': all(abs(r) < 1 for r in eigs), 'eigenvalues': eigs
            })
        except Exception:
            continue
    
    results_df = pd.DataFrame(records)
    best_aic = int(results_df.loc[results_df['aic'].idxmin(), 'lag'])
    best_bic = int(results_df.loc[results_df['bic'].idxmin(), 'lag'])
    best_hqic = int(results_df.loc[results_df['hqic'].idxmin(), 'lag'])
    return results_df, best_aic, best_bic, best_hqic


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
    reg_var_summary = []
    for name, df in all_data.items():
        y = df.iloc[:, 0]
        X = sm.add_constant(df.iloc[:, 1:])
        beta = matrix_ols_regression(y.values, X.values)
        preds = X.values @ beta
        r2 = 1 - ((y.values - preds)**2).sum() / ((y.values - y.mean())**2).sum()

        var_df, best_aic, best_bic, best_hqic = select_var_order(df.dropna())
        eigvals = var_df.loc[var_df['lag'] == best_aic, 'eigenvalues'].iloc[0]
        eigvals_str = ' '.join([f'{abs(x):.3f}' for x in eigvals])

        reg_var_summary.append({
            'group': name, 'r_squared': r2, 'best_aic': best_aic, 'best_bic': best_bic,
            'best_hqic': best_hqic, 'eigenvalues': eigvals_str
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
    """Perform Johansen multivariate cointegration test for triple groups."""
    triple_groups = [k for k in all_data.keys() if 'triple' in k.lower()]
    if not triple_groups:
        return pd.DataFrame()
    
    results = []
    for triple_name in triple_groups:
        df_triple = all_data[triple_name].dropna()
        n_assets, data_points = len(df_triple.columns), len(df_triple)
        
        # Default values for failed cases
        result = {
            'triple': triple_name, 'n_assets': n_assets, 'data_points': data_points,
            'n_coint_relations': None, 'first_eigenvec_norm': None, 'spread_vol': None
        }
        
        if n_assets >= 3 and data_points > 100:
            try:
                johansen_result = johansen(df_triple)
                n_coint = johansen_result['johansen_n']
                first_eigenvec = [johansen_result.get(f'eig_{i}', 0) for i in range(n_assets)]
                
                result.update({
                    'n_coint_relations': n_coint,
                    'first_eigenvec_norm': np.linalg.norm(first_eigenvec),
                    'spread_vol': sum(first_eigenvec[i] * df_triple.iloc[:, i] for i in range(n_assets)).std() if n_coint > 0 else np.nan
                })
            except Exception:
                pass  # Keep default None values
        
        results.append(result)
    
    return pd.DataFrame(results)


def johansen_sensitivity_summary(all_data, min_obs=120):
    """Minimal Johansen sensitivity summary."""
    warnings.filterwarnings('ignore', message='.*SVD did not converge.*')
    warnings.filterwarnings('ignore', message='.*Critical values are only available.*')

    triples = [k for k in all_data.keys() if 'triple' in k.lower()]
    if not triples:
        return pd.DataFrame(columns=['triple', 'window', 'rank>0', 'total', 'summary'])

    results = []
    for triple in triples:
        df0 = all_data[triple].replace([np.inf, -np.inf], np.nan).dropna().astype(float, errors='ignore')
        if df0.shape[1] < 3 or len(df0) < min_obs:
            continue

        # Split into windows
        mid = df0.index[len(df0.index)//2]
        windows = {'full': df0, 'H1': df0.loc[:mid], 'H2': df0.loc[mid:]}

        for win_name, sdf in windows.items():
            sdf = sdf.dropna()
            if len(sdf) < min_obs:
                continue

            # Create variants (daily/weekly, levels/logs)
            weekly = sdf.resample('W-FRI').last().dropna()
            variants = [('levels_daily', sdf), ('levels_weekly', weekly)]
            if (sdf > 0).all().all():
                variants.extend([('logs_daily', np.log(sdf)), ('logs_weekly', np.log(weekly))])

            # Test all combinations
            hits = total = 0
            for _, dfx in variants:
                if len(dfx) < min_obs:
                    continue
                for det_order in [-1, 0, 1]:
                    try:
                        with silence_fd_output():
                            res = johansen(dfx, det_order=det_order)
                        total += 1
                        if res.get('johansen_n', 0) > 0:
                            hits += 1
                    except Exception:
                        continue

            if total > 0:
                results.append({
                    'triple': triple, 'window': win_name, 'rank>0': hits, 
                    'total': total, 'summary': f"{hits}/{total}"
                })

    return pd.DataFrame(results).sort_values(['triple', 'window']) if results else pd.DataFrame(columns=['triple','window','rank>0','total','summary'])
