import numpy as np
from collections import Counter
import pandas as pd

class DecisionTree:
    def __init__(self):
        pass

    def calculate_entropy(self, labels):
        counter = Counter(labels)
        probabilities = [count / len(labels) for count in counter.values()]
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def calculate_information_gain(self, data, labels, feature):
        total_entropy = self.calculate_entropy(labels)
        feature_values = set(data[:, feature])
        weighted_entropy = 0

        for value in feature_values:
            subset_indices = np.where(data[:, feature] == value)
            subset_labels = labels[subset_indices]
            subset_entropy = self.calculate_entropy(subset_labels)
            weighted_entropy += (len(subset_labels) / len(labels)) * subset_entropy

        information_gain = total_entropy - weighted_entropy
        return information_gain

    def find_root_node(self, data, labels, binning_type=None, num_bins=None):
        if binning_type is not None and num_bins is not None:
            for i in range(data.shape[1]):  # Iterate over each feature
                data[:, i] = self.perform_binning(data[:, i], num_bins, binning_type)

        num_features = data.shape[1]
        information_gains = []

        for feature in range(num_features):
            information_gain = self.calculate_information_gain(data, labels, feature)
            information_gains.append(information_gain)

        root_node_index = np.argmax(information_gains)
        root_node_info = {
            "index": root_node_index,
            "information_gain": information_gains[root_node_index],
            "feature_values": set(data[:, root_node_index]),
            "num_instances": len(data),
            "num_features": num_features
        }
        return root_node_info

    def perform_binning(self, data, num_bins=None, binning_type='equal-width'):
        if binning_type == 'equal-width':
            return self.equal_width_binning(data, num_bins)
        elif binning_type == 'frequency':
            return self.frequency_binning(data, num_bins)
        else:
            raise ValueError("Invalid binning type. Choose 'equal-width' or 'frequency'.")

    def equal_width_binning(self, data, num_bins):
        min_value = np.min(data)
        max_value = np.max(data)
        bin_width = (max_value - min_value) / num_bins
        bins = [min_value + i * bin_width for i in range(num_bins)]
        binned_data = np.digitize(data, bins)
        return binned_data

    def frequency_binning(self, data, num_bins):
        _, bins = np.histogram(data, bins=num_bins)
        binned_data = np.digitize(data, bins)
        return binned_data

data = pd.read_csv('Data/parkinsson_data.csv')
X = data.drop(['status', 'name'], axis=1).values  # Features
y = data['status'].values  # Target

dt = DecisionTree()

binning_type = 'equal-width'  # Choose binning type ('equal-width' or 'frequency')
num_bins = 5  
root_node_info = dt.find_root_node(X, y, binning_type, num_bins)
print("Root node information:")
print("Index:", root_node_info["index"])
print("Information Gain:", root_node_info["information_gain"])
print("Feature Values:", root_node_info["feature_values"])
print("Number of Instances:", root_node_info["num_instances"])
print("Number of Features:", root_node_info["num_features"])
