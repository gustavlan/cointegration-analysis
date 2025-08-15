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
                           dZ:   float       = 0.1,
                           normalize: bool   = False
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

    # Vectorized processing using list comprehension
    def process_asset_group(name, df):
        n = df.shape[1]

        # Compute the spread & cointegration stats
        if n == 2:
            y, x = df.columns
            eg = engle_granger(df, y, x)
            spread, beta, eg_pv = eg['spread'], eg['beta'], eg['eg_pvalue']

            # ECM & Kalman for pairs
            if spread is not None:
                ecm_pv = analyze_error_correction_model(df[y], df[x], spread)['ecm_pvalue']
                kf_ts = kalman_hedge(df, y, x)['kf_beta']
                beta_stab = kf_ts.std() / abs(kf_ts.mean()) if kf_ts.mean() != 0 else np.nan
            else:
                ecm_pv, beta_stab = np.nan, np.nan

        elif n == 3:  # Johansen test for triples
            joh = johansen(df)
            eg_pv, ecm_pv, beta_stab = np.nan, np.nan, np.nan
            vec = np.array([joh[f'eig_{i}'] for i in range(3)])
            vec = vec / np.sum(np.abs(vec))  # normalize
            spread = df.dot(vec)
            beta = None

        else:
            return None

        # OU-params & static Sharpe for the spread
        if spread is not None:
            ou = ou_params(spread)
            hl, sig = ou['OU_HalfLife'], ou['ou_sigma']
            sharpe = spread.mean() / spread.std() if spread.std() != 0 else np.nan
        else:
            hl, sig, sharpe = np.nan, np.nan, np.nan

        # Z-sweep backtest
        if spread is not None:
            if n == 2:
                # For pairs, use standard pair PnL optimization
                df_opt = optimize_thresholds(
                    spread, spread.mean(), spread.std(), 
                    beta if beta is not None else 1.0,
                    y=df.iloc[:,0], x=df.iloc[:,1],
                    Z_min=Z_min, Z_max=Z_max, dZ=dZ, cost=cost,
                    normalize=normalize
                )
            else:
                # For triples/more, use spread returns directly 
                spread_returns = spread.diff().fillna(0)
                from threshold_optimization import backtest_spread
                df_opt_records = []
                for Z in np.arange(Z_min, Z_max + dZ, dZ):
                    stats = backtest_spread(spread, spread.mean(), spread.std(), 1.0, spread, spread, Z, cost)
                    stats['Z'] = Z
                    df_opt_records.append(stats)
                df_opt = pd.DataFrame(df_opt_records)

            # Find optimal Z threshold
            best_row = df_opt.loc[df_opt['Sharpe'].idxmax()] if 'Sharpe' in df_opt.columns else df_opt.iloc[0]
            z_star, n_trades, avg_pnl = best_row['Z'], best_row['N_trades'], best_row['avg_PnL']
        else:
            z_star, n_trades, avg_pnl = np.nan, np.nan, np.nan

        return {
            'pair': name,
            'n_assets': n,
            'eg_pvalue': eg_pv,
            'ecm_pvalue': ecm_pv,
            'beta_stab': beta_stab,
            'halflife': hl,
            'sigma': sig,
            'spread_sharpe': sharpe,
            'best_Z': z_star,
            'N_trades_Zstar': n_trades,
            'avg_PnL_Zstar': avg_pnl
        }

    # Process all asset groups and filter out None results
    records = [
        result for name, df in all_data.items() 
        if (result := process_asset_group(name, df)) is not None
    ]

    return pd.DataFrame(records)