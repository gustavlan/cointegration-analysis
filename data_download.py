import yfinance as yf
import pandas as pd
from datetime import datetime

def fetch_asset_data(asset_groups: dict,
                     start_date: datetime,
                     end_date:   datetime,
                     freq: str = "B"  # Default to business days
                    ):
    all_data = {}

    # Vectorized processing using dictionary comprehension
    def fetch_group_data(group_name, tickers):
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
            price = price.asfreq(freq, method='ffill')
            
            return price if not price.empty else None
        except Exception:
            return None

    # Process all groups and filter out None results  
    all_data = {
        group_name: data 
        for group_name, tickers in asset_groups.items() 
        if (data := fetch_group_data(group_name, tickers)) is not None
    }

    return all_data