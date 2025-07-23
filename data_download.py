import yfinance as yf
import pandas as pd
from datetime import datetime

def fetch_asset_data(asset_groups: dict,
                     start_date: datetime,
                     end_date:   datetime
                    ):
    all_data = {}

    for group_name, tickers in asset_groups.items():
        try:
            raw = yf.download(
                tickers,
                start=start_date.strftime('%Y-%m-%d'),
                end=  end_date.  strftime('%Y-%m-%d'),
                interval="1d",
                auto_adjust=True,
                group_by='ticker'
            )

            # MultiTicker vs single-ticker handling
            if isinstance(raw.columns, pd.MultiIndex):
                stacked = (
                    raw
                    .stack(level=0, future_stack=True)
                    .rename_axis(['Date','Ticker'])
                    .reset_index(level='Ticker')
                )
                price = stacked.pivot(columns='Ticker', values='Close')
            else:
                price = raw[['Close']].copy()
                if len(tickers) == 1:
                    price.columns = tickers

            price = price.ffill().dropna()
            if not price.empty:
                all_data[group_name] = price

        except Exception as e:
            continue

    return all_data