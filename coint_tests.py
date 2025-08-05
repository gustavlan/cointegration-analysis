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


def adf_results(series, freq="B"):
    """Returns ADF test outputs."""
    series = series.asfreq(freq)
    stat, pval, _, _, crit, _ = adfuller(series.dropna(), autolag='AIC')
    return {
        'stat': stat,
        'pvalue': pval,
        **{f'crit_{k}': v for k, v in crit.items()}
    }


def kpss_results(series, freq="B"):
    """Returns KPSS test outputs."""
    series = series.asfreq(freq)
    stat, pval, _, crit = kpss(series.dropna(), regression='c', nlags='auto')
    return {
        'stat': stat,
        'pvalue': pval,
        **{f'crit_{k}': v for k, v in crit.items()}
    }

def engle_granger(df, y, x, freq="B"):
    """Returns hedge ratio and ADF p-value on residuals, plus spread if cointegrated."""
    # Ensure frequency is set
    df = df.asfreq(freq)
    x0 = sm.add_constant(df[x])
    model = sm.OLS(df[y], x0).fit()
    beta = model.params[x]
    spread = df[y] - beta * df[x]
    pval = adfuller(spread.dropna(), maxlag=1, autolag=None)[1]
    return {'beta': beta, 'eg_pvalue': pval, 'spread': spread if pval <= .05 else None}


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
    ec_coeff = model.params['ec_term'] # extract results
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
    return {'ou_mu': mu, 'ou_theta': theta, 'ou_halflife': hl, 'ou_sigma': sigma_eq}


def johansen(df, freq="B"):
    """Returns number of coint relationships and eigenvector for first."""
    df = df.asfreq(freq)
    res = coint_johansen(df.dropna(), det_order=0, k_ar_diff=1)
    n = np.sum(res.lr1 > res.cvt[:, 1])
    vec = res.evec[:, 0]
    return {'johansen_n': int(n), **{f'eig_{i}': v for i, v in enumerate(vec)}}

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
    Multivariate time series df, fit VAR(p) for p=1..maxlags,
    record AIC, BIC, HQIC, plus companion‐matrix eigenvalues.
    freq: Frequency of the time series ('B' for business days, 'D' for calendar days)
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
        # Univariate tests: ADF and KPSS for each asset
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
    results = {}

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

        results[pair] = cv_df

    return results
