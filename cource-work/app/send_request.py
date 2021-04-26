import requests
import pandas as pd
import numpy as np

test_df = pd.read_csv('../data/test.csv')
url = 'http://127.0.0.1:8080/predict'


def test_one():
    data = test_df.iloc[10, :].to_json()
    print(data)
    response = requests.post(url, json=data).json()
    print(response)

if __name__ == "__main__":
    test_one()
