import warnings
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, InterpolationWarning
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.var_model import VAR
from pykalman import KalmanFilter
from backtest import nested_cv

warnings.filterwarnings(
    'ignore',
    message='The test statistic is outside of the range of p-values available',
    category=InterpolationWarning
)


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
        except Exception as e:
            # singularity, too many parameters etc.
            print(f"p={p} failed: {e}")
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


def za_test(series, trim=0.1, lags=None):
    """Zivot-Andrews unit-root test with structural break."""
    try:
        from arch.unitroot import ZivotAndrews
        s = series.dropna()
        if len(s) < 50:
            return {'stat': None, 'pvalue': None, 'breakpoint': None}
        
        if lags is None:
            res = ZivotAndrews(s, trim=trim, trend='ct')
        else:
            res = ZivotAndrews(s, trim=trim, lags=lags, trend='ct')
            
        breakpoint = getattr(res, 'breakpoint', None)
        return {'stat': res.stat, 'pvalue': res.pvalue, 'breakpoint': breakpoint}
    except (ImportError, Exception):
        return {'stat': None, 'pvalue': None, 'breakpoint': None}


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


def multi_subperiod_stability_check_fixed(df, y_col, x_col, n_periods=5):
    """
    Split data into exactly n_periods equal length periods and test each.
    Always calculates β_ec regardless of significance.
    """
    results = []
    
    # Calculate period length
    total_obs = len(df)
    period_length = total_obs // n_periods
    
    for i in range(n_periods):
        start_idx = i * period_length
        if i == n_periods - 1:  # Last period gets any remaining observations
            end_idx = total_obs
        else:
            end_idx = (i + 1) * period_length
            
        period_data = df.iloc[start_idx:end_idx]
        
        if len(period_data) < 30:  # Minimum observations check
            # Still add a row to maintain period count
            start_date = period_data.index[0].strftime('%Y-%m-%d') if len(period_data) > 0 else 'N/A'
            end_date = period_data.index[-1].strftime('%Y-%m-%d') if len(period_data) > 0 else 'N/A'
            
            results.append({
                'Period': f"{start_date} to {end_date}",
                'β_ec': np.nan,
                'p_value': np.nan,
                'Cointegrated?': 'No',
                'n_obs': len(period_data)
            })
            continue
            
        try:
            # Run Engle-Granger test
            eg_result = engle_granger(period_data, y_col, x_col)
            
            # Run ECM to get β_ec - ALWAYS calculate this
            ecm_result = analyze_error_correction_model(
                period_data[y_col], 
                period_data[x_col], 
                eg_result['spread']
            )
            
            start_date = period_data.index[0].strftime('%Y-%m-%d')
            end_date = period_data.index[-1].strftime('%Y-%m-%d')
            
            # Always include β_ec, mark significance separately
            results.append({
                'Period': f"{start_date} to {end_date}",
                'β_ec': ecm_result['ecm_coeff'],  # Always include
                'p_value': ecm_result['ecm_pvalue'],
                'Cointegrated?': 'Yes' if ecm_result['ecm_pvalue'] <= 0.05 else 'No',
                'n_obs': len(period_data)
            })
            
        except Exception as e:
            # Even for failed periods, maintain structure
            start_date = period_data.index[0].strftime('%Y-%m-%d') if len(period_data) > 0 else 'N/A'
            end_date = period_data.index[-1].strftime('%Y-%m-%d') if len(period_data) > 0 else 'N/A'
            
            results.append({
                'Period': f"{start_date} to {end_date}",
                'β_ec': np.nan,
                'p_value': np.nan,
                'Cointegrated?': 'No',
                'n_obs': len(period_data)
            })
    
    return pd.DataFrame(results)

def analyze_stability_across_classes_fixed(all_data, reg_var_summary):
    """
    Run stability check for pairs organized by asset class.
    Returns a summary DataFrame with exactly 5 periods for each pair.
    Always includes β_ec values regardless of significance.
    """
    asset_classes = {
        'Commodities': ['oil_pair', 'agri_pair'],
        'Fixed Income & Currency': ['yield_pair', 'currency_pair'], 
        'Volatility': ['volatility_pair'],
        'Country Indices': ['eu_index_pair_1', 'eu_index_pair_2'],
        'Equities': ['fr_banking_pair', 'fast_fashion_pair', 'investor_ab_pair', 'vw_porsche_pair', 'semiconductor_pair'],
        'ETFs': ['sector_etf_pair']
    }

    stability_summary_data = []
    
    # Flatten nested loops
    pair_analysis_data = [
        (asset_class, pair_name, all_data[pair_name])
        for asset_class, pair_list in asset_classes.items()
        for pair_name in pair_list
        if pair_name in all_data and pair_name.endswith('_pair') and len(all_data[pair_name].columns) >= 2
    ]
    
    for asset_class, pair_name, df in pair_analysis_data:
        y_col, x_col = df.columns[0], df.columns[1]
        try:
            # Force exactly 5 equal periods
            results_df = multi_subperiod_stability_check_fixed(df, y_col, x_col, n_periods=5)
            
            if not results_df.empty:
                # Calculate stability metrics (only count significant cointegration for this metric)
                n_cointegrated = (results_df['Cointegrated?'] == 'Yes').sum()
                stability_rate = (n_cointegrated / 5 * 100)  # Always 5 periods
                
                # Create row data starting with basic info
                row_data = {
                    'Pair': pair_name,
                    'Cointegrated': n_cointegrated,
                    'Stability_Rate_%': stability_rate,
                    'Overall_Stable': n_cointegrated >= 4  # 4 out of 5 periods
                }
                
                # Add individual period β_ec values - ALWAYS include from results_df
                for i in range(5):
                    if i < len(results_df):
                        # Always use the β_ec from results, regardless of significance
                        beta_ec = results_df.iloc[i]['β_ec']
                        row_data[f'Period_{i+1}_β_ec'] = beta_ec
                    else:
                        row_data[f'Period_{i+1}_β_ec'] = np.nan
                
                stability_summary_data.append(row_data)
            else:
                # Handle empty results - still create 5 period columns
                row_data = {
                    'Pair': pair_name,
                    'Cointegrated': 0,
                    'Stability_Rate_%': 0,
                    'Overall_Stable': False
                }
                for i in range(5):
                    row_data[f'Period_{i+1}_β_ec'] = np.nan
                
                stability_summary_data.append(row_data)
                
        except Exception:
            # Handle failed analysis - still create 5 period columns
            row_data = {
                'Pair': pair_name,
                'Cointegrated': 0,
                'Stability_Rate_%': 0,
                'Overall_Stable': False
            }
            for i in range(5):
                row_data[f'Period_{i+1}_β_ec'] = np.nan
                
            stability_summary_data.append(row_data)

    # Create summary DataFrame
    stability_df = pd.DataFrame(stability_summary_data)
    if not stability_df.empty:
        stability_df.set_index('Pair', inplace=True)
    
    return stability_df

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
