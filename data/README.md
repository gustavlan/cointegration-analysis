# CQF Final Exam - Asset Data CSV Files

This directory contains pre-downloaded market data for the pairs trading analysis notebook.

## Files

The following CSV files contain historical price data for various asset groups:

- `agri_pair_data.csv` - Corn and Soybean futures
- `currency_pair_data.csv` - AUD/USD and CAD/USD currency pairs
- `eu_index_pair_1_data.csv` - CAC 40 and DAX indices
- `eu_index_pair_2_data.csv` - IBEX 35 and FTSE MIB indices
- `fast_fashion_pair_data.csv` - Inditex and H&M stocks
- `fr_banking_pair_data.csv` - BNP Paribas and Societe Generale stocks
- `german_auto_triple_data.csv` - VW, Mercedes, and BMW stocks
- `investor_ab_pair_data.csv` - Investor A and B shares
- `oil_pair_data.csv` - WTI and Brent crude oil futures
- `precious_metals_triple_data.csv` - Gold, Silver, and Platinum futures
- `sector_etf_pair_data.csv` - Real Estate and Utilities ETFs
- `semiconductor_pair_data.csv` - ASML and ASM International stocks
- `volatility_pair_data.csv` - VIX index and VIXY ETF
- `vw_porsche_pair_data.csv` - VW and Porsche stocks
- `yield_pair_data.csv` - US 10Y Treasury yield and UK Gilts ETF

## Data Format

Each CSV file contains:
- Index: Date (YYYY-MM-DD format)
- Columns: Asset prices (adjusted closing prices from Yahoo Finance)
- Frequency: Business days (weekdays only)
- Period: Approximately 5 years from the download date

## Usage

The notebook is configured to load data from these CSV files by default. To use fresh data instead:

1. Uncomment the line: `all_data = fetch_asset_data(asset_groups, start_date, end_date)`
2. Comment out the CSV loading section
3. Re-run the save-to-CSV cell to update the files

## Benefits

Using CSV files instead of live downloads:
- ✅ Faster notebook execution
- ✅ Reproducible results
- ✅ No dependency on internet connection
- ✅ No risk of API rate limits
- ✅ Easy sharing with collaborators
