#Write a function to calculate the Euclidean distance and Manhattan distance between two vectors. The vectors dimension is variable. Please don’t use any distance calculation functions available in Python.
from typing import List

def euclidean_distance(vec1: List[float], vec2: List[float]) -> float:
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same dimension")
    
    squared_diff_sum = sum((x - y) ** 2 for x, y in zip(vec1, vec2))
    
    return squared_diff_sum ** 0.5

def manhattan_distance(vec1: List[float], vec2: List[float]) -> float:
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same dimension")
    
    return sum(abs(x - y) for x, y in zip(vec1, vec2))

vector1 = [1, 2, 3]
vector2 = [4, 5, 6]

euclidean_dist = euclidean_distance(vector1, vector2)
manhattan_dist = manhattan_distance(vector1, vector2)

print("Euclidean distance:", euclidean_dist)
print("Manhattan distance:", manhattan_dist)
