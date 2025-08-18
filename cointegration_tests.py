import warnings
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.var_model import VAR

warnings.filterwarnings('ignore')

def matrix_ols_regression(y, X):
    """Compute OLS regression coefficients using matrix operations."""
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
    """Run ADF test on time series to check for stationarity."""
    series = series.asfreq(freq)
    stat, pval, _, _, crit, _ = adfuller(series.dropna(), autolag='AIC')
    if verbose:
        print(f"ADF(p={pval:.3f}) → {'stationary' if pval < 0.05 else 'non-stationary'}")
    return {'stat': stat, 'pvalue': pval, **{f'crit_{k}': v for k, v in crit.items()}}

def kpss_results(series, freq="B", verbose=False):
    """Run KPSS test on time series to check for stationarity (opposite null hypothesis to ADF)."""
    series = series.asfreq(freq)
    stat, pval, _, crit = kpss(series.dropna(), regression='c', nlags='auto')
    if verbose:
        print(f"KPSS(p={pval:.3f}) → {'non-stationary' if pval < 0.05 else 'stationary'}")
    return {'stat': stat, 'pvalue': pval, **{f'crit_{k}': v for k, v in crit.items()}}

def engle_granger(df, y, x, maxlag=1, freq="B", verbose=False):
    """Perform Engle-Granger cointegration test between two price series."""
    df = df.asfreq(freq)
    x0 = sm.add_constant(df[x])
    model = sm.OLS(df[y], x0).fit()
    beta, alpha = model.params[x], model.params['const']
    spread = model.resid
    pval = adfuller(spread.dropna(), maxlag=maxlag, autolag=None)[1]  # test spread stationarity
    if verbose:
        print(f"ADF(p={pval:.3f}) → {'stationary' if pval < 0.05 else 'non-stationary'}")
    return {'beta': beta, 'alpha': alpha, 'eg_pvalue': pval, 'spread': spread if pval <= .05 else None, 'maxlag': maxlag}

def engle_granger_bidirectional(df, a, b, maxlag=1, freq="B", verbose=False):
    """Run Engle-Granger test in both directions and return the better result."""
    r_ab = engle_granger(df, a, b, maxlag=maxlag, freq=freq, verbose=verbose)
    r_ba = engle_granger(df, b, a, maxlag=maxlag, freq=freq, verbose=verbose)
    
    if r_ab['eg_pvalue'] <= r_ba['eg_pvalue']:
        return r_ab
    
    # Convert b->a result to a->b format for consistency
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
    """Estimate error correction model coefficients from cointegrated series."""
    y, x, spread = y.asfreq(freq), x.asfreq(freq), spread.asfreq(freq)
    ec_term = spread.shift(1).dropna()  # lagged spread as error correction term
    delta_y, delta_x = y.diff().dropna(), x.diff().dropna()
    aligned_data = pd.concat([delta_y, delta_x, ec_term], axis=1).dropna()
    aligned_data.columns = ['delta_y', 'delta_x', 'ec_term']
    X_ecm = sm.add_constant(aligned_data[['delta_x', 'ec_term']])
    model = sm.OLS(aligned_data['delta_y'], X_ecm).fit()
    return {'ecm_coeff': model.params['ec_term'], 'ecm_pvalue': model.pvalues['ec_term']}

def ou_params(spread, freq="B"):
    """Estimate Ornstein-Uhlenbeck process parameters from spread series."""
    spread = spread.asfreq(freq)
    dS = spread.diff().dropna()  # first difference 
    S1 = spread.shift(1).dropna()  # lagged level
    df = pd.concat([dS, S1], axis=1).dropna()
    df.columns = ['dS', 'S1']
    model = sm.OLS(df['dS'], sm.add_constant(df['S1'])).fit()
    theta = -model.params['S1']  # mean reversion speed
    mu = model.params['const'] / theta  # long-run mean
    hl = np.log(2) / theta  # half-life calculation
    return {'ou_mu': mu, 'ou_theta': theta, 'OU_HalfLife': hl, 'ou_sigma': spread.std()}

def select_var_order(df, maxlags=10, trend='c', freq="B"):
    """Select optimal VAR model order using information criteria."""
    df = df.asfreq(freq)
    records = []
    for p in range(1, maxlags+1):
        try:
            res = VAR(df).fit(p, trend=trend)
            records.append({
                'lag': p, 'aic': res.aic, 'bic': res.bic, 'hqic': res.hqic,
                'stable': all(abs(r) < 1 for r in res.roots), 'eigenvalues': res.roots  # check stability condition
            })
        except Exception:
            continue
    
    if not records:
        return pd.DataFrame(), 1, 1, 1
    results_df = pd.DataFrame(records)
    return (results_df, 
            int(results_df.loc[results_df['aic'].idxmin(), 'lag']),
            int(results_df.loc[results_df['bic'].idxmin(), 'lag']),
            int(results_df.loc[results_df['hqic'].idxmin(), 'lag']))

def johansen(df, freq="B", det_order=0):
    """Perform Johansen cointegration test to find number of cointegrating relationships."""
    df = df.asfreq(freq).dropna()
    try:
        _, best_aic, best_bic, best_hqic = select_var_order(df)
        k_ar_diff = max(int(np.median([best_aic, best_bic, best_hqic])) - 1, 1)  # VAR order minus 1
    except Exception:
        k_ar_diff = 1
    res = coint_johansen(df.values, det_order=det_order, k_ar_diff=k_ar_diff)
    out = {'johansen_n': int(np.sum(res.lr1 > res.cvt[:, 1]))}  # count significant relationships
    if res.evec is not None and res.evec.shape[1] > 0:
        for i, w in enumerate(res.evec[:, 0]):  # first eigenvector coefficients
            out[f'eig_{i}'] = float(w)
    out.update({'k_ar_diff_used': int(k_ar_diff), 'det_order': int(det_order)})
    return out


def za_test(series, trim=0.1, lags=None, model='trend'):
    """Perform Zivot-Andrews structural break test for unit root with endogenous breakpoint."""
    s = series.dropna()
    regression = 'ct' if model == 'trend' else 'c'  # constant + trend or constant only
    autolag = 'AIC' if lags is None else None
    stat, pval, crit, bp, usedlag = zivot_andrews(s.values, trim=trim, maxlag=lags, regression=regression, autolag=autolag)
    return pd.DataFrame([{
        'stat': float(stat), 'pvalue': float(pval), 'breakpoint': int(bp),
        'break_date': pd.Timestamp(s.index[int(bp)]), 'model': model
    }])

def analyze_regression_var_summary(all_data):
    """Analyze regression quality and VAR model properties for multiple datasets."""
    reg_var_summary = []
    for name, df in all_data.items():
        y, X = df.iloc[:, 0], sm.add_constant(df.iloc[:, 1:])
        beta = matrix_ols_regression(y.values, X.values)
        r2 = 1 - ((y.values - X.values @ beta)**2).sum() / ((y.values - y.mean())**2).sum()  # R-squared calculation
        var_df, best_aic, best_bic, best_hqic = select_var_order(df.dropna())
        eigvals_str = ' '.join([f'{abs(x):.3f}' for x in var_df.loc[var_df['lag'] == best_aic, 'eigenvalues'].iloc[0]]) if len(var_df) > 0 else ''
        reg_var_summary.append({
            'group': name, 'r_squared': r2, 'best_aic': best_aic, 'best_bic': best_bic,
            'best_hqic': best_hqic, 'eigenvalues': eigvals_str
        })
    return pd.DataFrame(reg_var_summary)


def analyze_ecm_timeslices(y, x, spread=None, periods=5, include_dX=True, reestimate_beta_per_slice=True, 
                         pick_direction=True, freq="B", min_obs=30, return_details=False):
    """Analyze error correction model stability across time periods."""
    df = pd.concat({'y': y.asfreq(freq), 'x': x.asfreq(freq)}, axis=1).dropna()
    if len(df) < min_obs:
        return pd.DataFrame([[np.nan]*periods, [np.nan]*periods], 
                           index=['ecm_coeff','ecm_pvalue'], columns=[f'P{i+1}' for i in range(periods)])
    
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
        x0 = sm.add_constant(sub['x'])
        model = sm.OLS(sub['y'], x0).fit()
        beta, alpha = model.params['x'], model.params['const']
        spread_local = sub['y'] - alpha - beta * sub['x']  # calculate spread
        
        # ECM regression: delta_y = gamma * error_correction_term + noise
        d_y = sub['y'].diff().dropna()
        u_lag = spread_local.shift(1).dropna()  # lagged error correction term
        aligned = pd.concat([d_y, u_lag], axis=1).dropna()
        
        if len(aligned) < 3:
            results.append((np.nan, np.nan))
            continue
            
        ecm_model = sm.OLS(aligned.iloc[:, 0], sm.add_constant(aligned.iloc[:, 1])).fit()
        coeff = ecm_model.params.iloc[1]  # coefficient on u_lag
        pval = ecm_model.pvalues.iloc[1]
        results.append((coeff, pval))
    
    return pd.DataFrame(results, index=[f'P{i+1}' for i in range(len(results))], 
                       columns=['ecm_coeff', 'ecm_pvalue']).T


def analyze_johansen_triples(all_data):
    """Analyze Johansen cointegration for datasets with three or more assets."""
    triple_groups = [k for k in all_data.keys() if 'triple' in k.lower()]
    if not triple_groups:
        return pd.DataFrame()
    
    results = []
    for triple_name in triple_groups:
        df_triple = all_data[triple_name].dropna()
        n_assets, data_points = len(df_triple.columns), len(df_triple)
        result = {'triple': triple_name, 'n_assets': n_assets, 'data_points': data_points,
                 'n_coint_relations': None, 'first_eigenvec_norm': None, 'spread_vol': None}
        
        if n_assets >= 3 and data_points > 100:  # minimum requirements for Johansen test
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
                pass
        results.append(result)
    return pd.DataFrame(results)

def johansen_sensitivity_summary(all_data, min_obs=120):
    """Test Johansen cointegration sensitivity across different specifications and time windows."""
    warnings.filterwarnings('ignore')
    triples = [k for k in all_data.keys() if 'triple' in k.lower()]
    if not triples:
        return pd.DataFrame(columns=['triple', 'window', 'rank>0', 'total', 'summary'])

    results = []
    for triple in triples:
        df0 = all_data[triple].replace([np.inf, -np.inf], np.nan).dropna()
        if df0.shape[1] < 3 or len(df0) < min_obs:
            continue

        mid = df0.index[len(df0.index)//2]  # split data in half
        for win_name, sdf in [('full', df0), ('H1', df0.loc[:mid]), ('H2', df0.loc[mid:])]:
            if len(sdf) < min_obs:
                continue

            # Test multiple data transformations and frequencies
            variants = [('levels_daily', sdf), ('levels_weekly', sdf.resample('W-FRI').last().dropna())]
            if (sdf > 0).all().all():  # only add log variants if all values are positive
                variants.extend([('logs_daily', np.log(sdf)), ('logs_weekly', np.log(sdf.resample('W-FRI').last().dropna()))])

            hits = total = 0
            for _, dfx in variants:
                if len(dfx) < min_obs:
                    continue
                for det_order in [-1, 0, 1]:  # test different deterministic trend assumptions
                    try:
                        with silence_fd_output():
                            res = johansen(dfx, det_order=det_order)
                        total += 1
                        if res.get('johansen_n', 0) > 0:  # count cases with at least one cointegrating relationship
                            hits += 1
                    except Exception:
                        continue

            if total > 0:
                results.append({'triple': triple, 'window': win_name, 'rank>0': hits, 
                               'total': total, 'summary': f"{hits}/{total}"})

    return pd.DataFrame(results).sort_values(['triple', 'window']) if results else pd.DataFrame(columns=['triple','window','rank>0','total','summary'])
