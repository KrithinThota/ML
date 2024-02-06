def label_encode(categories):
    category_to_numeric = {}
    for i, category in enumerate(categories):
        if category not in category_to_numeric:
            category_to_numeric[category] = len(category_to_numeric)
    return category_to_numeric

categories = ['apple', 'banana', 'orange', 'apple', 'orange', 'banana']
category_to_numeric = label_encode(categories)
print("Category to Numeric Mapping:\n", category_to_numeric)
