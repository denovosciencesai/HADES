# run_hades_csv.py
import argparse
import pandas as pd
from hades_calculator import HADES

def main():
    parser = argparse.ArgumentParser(description="Score SMILES in a CSV with HADES")
    parser.add_argument("input_csv", help="Path to input CSV")
    parser.add_argument("output_csv", help="Path to write output CSV")
    parser.add_argument("--smiles-col", default="SMILES", help="Name of the SMILES column in the input CSV")
    args = parser.parse_args()

    df_in = pd.read_csv(args.input_csv)
    smiles_list = df_in[args.smiles_col].tolist()

    hades = HADES()
    result = hades.score(smiles_list)  # returns DataFrame: SMILES, HADES_Score

    # merge back onto original columns, in case the input had more than just SMILES
    df_out = df_in.merge(result, left_on=args.smiles_col, right_on="SMILES", how="left")
    df_out.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(df_out)} scored rows to {args.output_csv}")

if __name__ == "__main__":
    main()