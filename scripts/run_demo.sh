#!/bin/bash
set -e
# demo end-to-end minimal
TICKER=${1:-AAPL}
START=${2:-2015-01-01}
END=${3:-2024-01-01}

python src/data/download_data.py --ticker $TICKER --start $START --end $END
python src/preprocess.py --input data/raw/${TICKER}.csv --output data/processed/${TICKER}_parsed.csv
python src/evaluation/backtest.py --input data/processed/${TICKER}_parsed.csv
echo "Demo finished. Check backtest_results.csv and models/"