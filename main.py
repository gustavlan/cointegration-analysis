import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen
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
        return beta.values
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
    return {'ou_mu': mu, 'ou_theta': theta, 'ou_halflife': hl}

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

# 3. Loop through groups, collect into a DataFrame

records = []

for group, df in all_data.items():
    for col in df.columns:
        adf = adf_results(df[col])
        kps = kpss_results(df[col])
        rec = {'group': group, 'asset': col, **adf, **kps}
        records.append(rec)

    # pair or triple logic
    if len(df.columns) == 2:
        y, x = df.columns
        eg = engle_granger(df, y, x)
        rec = {'group': group, 'test': 'Engle-Granger', **{k: v for k, v in eg.items() if k!='spread'}}
        records.append(rec)
        if eg['spread'] is not None:
            ou = ou_params(eg['spread'])
            rec = {'group': group, 'test': 'OU', **ou}
            records.append(rec)
        kf = kalman_hedge(df, y, x)
        # store only summary stats for KF
        rec = {'group': group, 'test': 'Kalman', 'kf_beta_mean': kf['kf_beta'].mean()}
        records.append(rec)

    elif len(df.columns) == 3:
        jres = johansen(df)
        rec = {'group': group, 'test': 'Johansen', **jres}
        records.append(rec)

summary_df = pd.DataFrame(records)
