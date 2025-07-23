import numpy as np
import pandas as pd

from coint_tests import (
    engle_granger,
    ou_params,
    kalman_hedge,
    analyze_error_correction_model,
    johansen
)
from threshold_optimization import optimize_thresholds

def assemble_group_summary(all_data: dict[str, pd.DataFrame],
                           cost: float       = 0.002,
                           Z_min: float      = 0.5,
                           Z_max: float      = 3.0,
                           dZ:   float       = 0.1
                          ) -> pd.DataFrame:
    """
    For each 2-asset or 3-asset group in all_data, compute:
      - cointegration stats (Engle-Granger for pairs, Johansen for triples)
      - OU half-life + sigma of the spread
      - static spread Sharpe
      - Kalman β stability (for pairs only)
      - ECM p-value (for pairs only)
      - optimal Z*, N_trades and avg_PnL from threshold sweep
    """
    records = []

    for name, df in all_data.items():
        n = df.shape[1]

        # Compute the spread & coins stats
        if n == 2:
            y, x = df.columns
            eg  = engle_granger(df, y, x)
            spread = eg['spread']
            beta   = eg['beta']
            eg_pv  = eg['eg_pvalue']

            # ECM & Kalman for pairs
            if spread is not None:
                ecm_pv = analyze_error_correction_model(df[y], df[x], spread)['ecm_pvalue']
                kf_ts  = kalman_hedge(df, y, x)['kf_beta']
                beta_stab = kf_ts.std() / abs(kf_ts.mean()) if kf_ts.mean() != 0 else np.nan
            else:
                ecm_pv, beta_stab = np.nan, np.nan

        elif n == 3: # if more than 2 Johansen test
            joh = johansen(df)
            eg_pv     = np.nan
            ecm_pv    = np.nan
            beta_stab = np.nan
            vec = np.array([joh[f'eig_{i}'] for i in range(3)]) # build the spread as first eigenvector combination
            vec = vec / np.sum(np.abs(vec)) # normalize so sum(abs(vec))==1
            spread = df.dot(vec)
            beta   = None  # no beta for >2 assets

        else:
            continue

        # OU‐params & static Sharpe for the spread
        if spread is not None:
            ou   = ou_params(spread)
            hl   = ou['ou_halflife']
            sig  = ou['ou_sigma']
            sharpe = spread.mean() / spread.std() if spread.std() != 0 else np.nan
        else:
            hl, sig, sharpe = np.nan, np.nan, np.nan

        # Z‐sweep backtest
        if spread is not None:
            df_opt  = optimize_thresholds(
                          spread, spread.mean(), spread.std(), 
                          beta if beta is not None else 1.0,
                          y=df.iloc[:,0], x=df.iloc[:,1] if n==2 else df.iloc[:,2],
                          Z_min=Z_min, Z_max=Z_max, dZ=dZ, cost=cost
                      )
            best    = df_opt.loc[df_opt['cum_PnL'].idxmax()]
            Z_star  = best['Z']
            N_tr    = best['N_trades']
            avg_pnl = best['avg_PnL']
        else:
            Z_star, N_tr, avg_pnl = np.nan, np.nan, np.nan

        records.append({
            'group':             name,
            'n_assets':          n,
            'eg_pvalue':         eg_pv,
            'joh_n_relations':   joh.get('johansen_n') if n==3 else np.nan,
            'ou_halflife':       hl,
            'ou_sigma':          sig,
            'sharpe_spread':     sharpe,
            'beta_stability':    beta_stab,
            'ecm_pvalue':        ecm_pv,
            'Z_star':            Z_star,
            'N_trades_Zstar':    N_tr,
            'avg_PnL_Zstar':     avg_pnl
        })

    return pd.DataFrame(records)