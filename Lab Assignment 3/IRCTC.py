import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_path, sheet_name):
    return pd.read_excel(file_path, sheet_name=sheet_name)

def calculate_mean_and_variance(data):
    price_mean = data['Price'].mean()
    price_variance = data['Price'].var()
    return price_mean, price_variance

def calculate_sample_mean(data, condition_column, condition_value, target_column):
    subset = data[data[condition_column] == condition_value][target_column]
    if not subset.empty:
        return subset.mean()
    return None

def calculate_loss_probability(data, target_column):
    return (data[target_column] < 0).mean()

def calculate_profit_probability(data, condition_column, condition_value, target_column):
    subset = data[(data[condition_column] == condition_value) & (data[target_column] > 0)]
    if not subset.empty:
        return subset.shape[0] / len(subset)
    return None

def scatter_plot(data, x_column, y_column):
    plt.scatter(data[x_column], data[y_column])
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(f'Scatter plot of {y_column} data against {x_column}')
    plt.xticks(rotation=45)
    plt.show()

file_path = 'Lab Assignment 3\lab.xlsx'
sheet_name = 'IRCTC Stock Price'
df = load_data(file_path, sheet_name)

price_mean, price_variance = calculate_mean_and_variance(df)
print("Mean of Price data:", price_mean)
print("Variance of Price data:", price_variance)

wednesday_mean = calculate_sample_mean(df, 'Day', 'Wed', 'Price')
if wednesday_mean is not None:
    print("\nSample mean of Wednesday price data:", wednesday_mean)

april_mean = calculate_sample_mean(df, 'Month', 'Apr', 'Price')
if april_mean is not None:
    print("\nSample mean of April price data:", april_mean)

loss_probability = calculate_loss_probability(df, 'Chg%')
print("\nProbability of making a loss over the stock:", loss_probability)

wednesday_profit_probability = calculate_profit_probability(df, 'Day', 'Wed', 'Chg%')
if wednesday_profit_probability is not None:
    print("Probability of making a profit on Wednesday:", wednesday_profit_probability)

conditional_profit_probability = calculate_profit_probability(df, 'Day', 'Wed', 'Chg%')
if conditional_profit_probability is not None:
    print("Conditional probability of making profit, given that today is Wednesday:", conditional_profit_probability)

scatter_plot(df, 'Day', 'Chg%')
