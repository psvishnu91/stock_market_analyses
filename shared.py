import json
import pandas as pd


def df_from_csv(path):
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    return df.astype(float)



def load_json(path):
    with open(path) as infile:
        return json.load(infile)
