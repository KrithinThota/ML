import pandas as pd

data = pd.read_csv("Data/stock.csv")

data['Date'] = pd.to_datetime(data['Date'])

day_mapping = {
    'Mon': 1,
    'Tue': 2,
    'Wed': 3,
    'Thu': 4,
    'Fri': 5,
    'Sat': 6,
    'Sun': 7
}

# Convert day names to integers
data['Day'] = data['Day'].map(day_mapping)

data.to_csv("Data/stock_re.csv", index=False)
