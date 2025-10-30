#!/usr/bin/env python3
"""
Simple preprocessing: parse dates, resample daily, fill missing, save processed csv.
Usage:
  python src/preprocess.py --input data/raw/AAPL.csv --output data/processed/AAPL_parsed.csv
"""
import os
import argparse
import pandas as pd

def preprocess_csv(in_path, out_path, freq="D"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = pd.read_csv(in_path, parse_dates=True, index_col=0)
    df = df.sort_index()
    # keep Close price as target
    if "Close" not in df.columns:
        raise RuntimeError("Expected 'Close' column")
    df = df[["Close"]].rename(columns={"Close":"y"})
    # resample (daily) and forward-fill
    df = df.asfreq(freq)
    df["y"] = df["y"].ffill()
    df.to_csv(out_path)
    print(f"Processed saved to {out_path}")
    return out_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    preprocess_csv(args.input, args.output)