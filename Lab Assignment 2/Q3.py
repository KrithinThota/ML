#Write a function to convert categorical variables to numeric using label encoding. Don’t use any existing functionalities.
def label_encode(categories):
    category_to_numeric = {}
    for i, category in enumerate(categories):
        if category not in category_to_numeric:
            category_to_numeric[category] = i
    return category_to_numeric

def apply_label_encoding(data, category_to_numeric):
    encoded_data = [category_to_numeric[category] for category in data]
    return encoded_data

categories = ['apple', 'banana', 'orange', 'apple', 'orange', 'banana']
category_to_numeric = label_encode(categories)
encoded_data = apply_label_encoding(categories, category_to_numeric)
print("Encoded Data:", encoded_data)
