import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.var_model import VAR
from pykalman import KalmanFilter

# --- 1. Getting the data ---

end_date = datetime.now()
start_date = end_date - timedelta(days=5*365) # five years of data

# A dictionary to organize all the asset groups and their tickers
asset_groups = {
    # Commodities
    "precious_metals_triple": ["GC=F", "SI=F", "PL=F"], # Gold, Silver, Platinum Futures
    "oil_pair": ["CL=F", "BZ=F"],                     # WTI, Brent Crude Futures
    "agri_pair": ["ZC=F", "ZS=F"],                    # Corn, Soybean Futures

    # Fixed Income & Currency
    "yield_pair": ["^TNX", "IGLT.L"],                 # US 10Y Yield, iShares UK Gilts ETF
    "currency_pair": ["AUDUSD=X", "CADUSD=X"],        # AUD/USD, CAD/USD

    # Volatility
    "volatility_pair": ["^VIX", "VIXY"],            # US VIX Index vs. Short-Term VIX Futures ETF*

    # Country Indices
    "eu_index_pair_1": ["^FCHI", "^GDAXI"],           # CAC 40, DAX
    "eu_index_pair_2": ["^IBEX", "FTSEMIB.MI"],      # IBEX 35, FTSE MIB

    # Equities
    "fr_banking_pair": ["BNP.PA", "GLE.PA"],          # BNP Paribas, Societe Generale
    "fast_fashion_pair": ["ITX.MC", "HM-B.ST"],       # Inditex, H&M
    "german_auto_triple": ["VOW3.DE", "MBG.DE", "BMW.DE"], # VW, Mercedes, BMW
    "investor_ab_pair": ["INVE-A.ST", "INVE-B.ST"],    # Investor A, Investor B
    "vw_porsche_pair": ["VOW3.DE", "P911.DE"],        # VW, Porsche AG
    "semiconductor_pair": ["ASML.AS", "IFX.DE"],      # ASML, Infineon

    # ETFs
    "sector_etf_pair": ["XLRE", "XLU"]                # Real Estate ETF, Utilities ETF
}

all_data = {}

print("Starting data download...")

for group_name, tickers in asset_groups.items():
    print(f"--> Downloading data for: {group_name}")
    try:
        # Download daily data for the specified tickers
        data = yf.download(tickers,
                           start=start_date.strftime('%Y-%m-%d'),
                           end=end_date.strftime('%Y-%m-%d'),
                           interval="1d",
                           auto_adjust=True,
                           group_by='ticker')

        # When a single ticker in a group fails, yfinance might return a DataFrame
        # with only the successful tickers. We need to handle this.
        if isinstance(data.columns, pd.MultiIndex):
            df_processed = data.stack(level=0, future_stack=True).rename_axis(['Date', 'Ticker']).reset_index(level=1)
            price_data = df_processed.pivot(columns='Ticker', values='Close')
        else:
            # If only one ticker was successful, it won't have a multi-index
            price_data = data[['Close']]
            # Rename column to the correct ticker if there's only one
            if len(tickers) == 1:
                price_data.columns = tickers

        price_data = price_data.ffill().dropna()

        if not price_data.empty:
            all_data[group_name] = price_data
        else:
            print(f"    No data for {group_name} after processing.")

    except Exception as e:
        print(f"    An error occurred while downloading {group_name}: {e}")

print("\nData download complete.")

# --- 3. Verification ---

print("\n--- Verification ---")
print(f"Successfully downloaded data for {len(all_data)} groups.")
print("The following data groups are now available:")
for name in sorted(all_data.keys()):
    print(f"- {name}")

# --- 4. Statistical Tests ---

def matrix_ols_regression(y, X):
    """
    Performs OLS regression using matrix algebra with numpy.
    """
    try:
        # Using the OLS formula: beta = (X'X)^(-1) * X'y
        XTX = X.T @ X
        XTX_inv = np.linalg.inv(XTX)
        XTY = X.T @ y
        beta = XTX_inv @ XTY
        return beta
    except np.linalg.LinAlgError:
        # This can happen if the matrix is singular (perfect multicollinearity)
        return None


def adf_results(series):
    """Returns ADF test outputs."""
    stat, pval, _, _, crit, _ = adfuller(series.dropna(), autolag='AIC')
    return {
        'stat': stat,
        'pvalue': pval,
        **{f'crit_{k}': v for k, v in crit.items()}
    }

# DONT USE KPSS according to the lecture
def kpss_results(series):
    """Returns KPSS test outputs."""
    stat, pval, _, crit = kpss(series.dropna(), regression='c', nlags='auto')
    return {
        'stat': stat,
        'pvalue': pval,
        **{f'crit_{k}': v for k, v in crit.items()}
    }

def engle_granger(df, y, x):
    """Returns hedge ratio and ADF p-value on residuals, plus spread if cointegrated."""
    x0 = sm.add_constant(df[x])
    model = sm.OLS(df[y], x0).fit()
    beta = model.params[x]
    spread = df[y] - beta * df[x]
    pval = adfuller(spread.dropna(), maxlag=1, autolag=None)[1]
    return {'beta': beta, 'eg_pvalue': pval, 'spread': spread if pval <= .05 else None}

def analyze_error_correction_model(y, x, spread):
    """
    Error-Correction Model (ECM) analysis.
    Returns the coefficient and p-value of the error correction term.
    """
    # Lag the spread to get the error-correction term e(t-1)
    ec_term = spread.shift(1).dropna()
    
    # Calculate the differences of the original series
    delta_y, delta_x = y.diff().dropna(), x.diff().dropna()
    
    # Align all series to the same index
    aligned_data = pd.concat([delta_y, delta_x, ec_term], axis=1).dropna()
    aligned_data.columns = ['delta_y', 'delta_x', 'ec_term']

    # Regress delta_y on delta_x and the error correction term
    X_ecm = sm.add_constant(aligned_data[['delta_x', 'ec_term']])
    y_ecm = aligned_data['delta_y']
    
    model = sm.OLS(y_ecm, X_ecm).fit()
    
    # Extract results for the error correction term
    ec_coeff = model.params['ec_term']
    ec_pvalue = model.pvalues['ec_term']
    
    return {'ecm_coeff': ec_coeff, 'ecm_pvalue': ec_pvalue}

def ou_params(spread):
    """Returns OU mu, theta, and half-life."""
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


def johansen(df):
    """Returns number of coint relationships and eigenvector for first."""
    res = coint_johansen(df.dropna(), det_order=0, k_ar_diff=1)
    n = np.sum(res.lr1 > res.cvt[:, 1])
    vec = res.evec[:, 0]
    return {'johansen_n': int(n), **{f'eig_{i}': v for i, v in enumerate(vec)}}

def kalman_hedge(df, y, x):
    """Returns dynamic beta and spread series."""
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


def select_var_order(df, maxlags=10, trend='c'):
    """
    Multivariate time series df, fit VAR(p) for p=1..maxlags,
    record AIC, BIC, HQIC, plus companion‐matrix eigenvalues,
    """
    records = []
    for p in range(1, maxlags+1):
        model = VAR(df)
        try:
            res = model.fit(p, trend=trend)
        except Exception as e:
            # singularity, too many parameters, etc.
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
    Split df into n_periods equal length slices,
    run Engle Granger + ECM on each, and return a summary DataFrame.

    """
    # 1. compute period boundaries
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
        
        # 2. Engle–Granger on this slice
        eg   = engle_granger(slice_df, y, x)
        beta = eg['beta']
        p_eg = eg['eg_pvalue']
        
        # 3. If cointegrated, get ECM coeff; else NaNs
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


# 3. Loop through groups, collect into a DataFrame

records = []
for group, df in all_data.items():
    # Univariate tests
    for col in df.columns:
        records.append({'group': group, 'asset': col, **adf_results(df[col]), **kpss_results(df[col])})

    # Pair vs. Triple logic
    n_assets = len(df.columns)
    if n_assets == 2:
        y, x = df.columns
        # Engle–Granger
        eg = engle_granger(df, y, x)
        records.append({'group': group, 'test': 'Engle-Granger', 'beta': eg['beta'], 'eg_pvalue': eg['eg_pvalue']})

        # Matrix OLS
        X0 = sm.add_constant(df[x])
        mbeta = matrix_ols_regression(df[y].values, X0.values)
        if mbeta is not None:
            records.append({
                'group': group,
                'test': 'Matrix-OLS',
                'const': mbeta[0],
                'slope': mbeta[1]
            })

        # If cointegrated, OU & ECM
        if eg['spread'] is not None:
            ou = ou_params(eg['spread'])
            records.append({'group': group, 'test': 'OU', **ou})

            ecm = analyze_error_correction_model(df[y], df[x], eg['spread'])
            records.append({'group': group, 'test': 'ECM', **ecm})

        # Kalman summary
        kf = kalman_hedge(df, y, x)
        records.append({
            'group': group,
            'test': 'Kalman',
            'kf_beta_mean': kf['kf_beta'].mean()
        })

    elif n_assets == 3:
        # Johansen for triples
        jres = johansen(df)
        records.append({'group': group, 'test': 'Johansen', **jres})

summary_df = pd.DataFrame(records)
