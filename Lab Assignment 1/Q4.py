#Write a program that accepts a matrix as input and returns its transpose.
def mat_transpose(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    transpose = [[0] * rows for _ in range(cols)]
    for i in range(rows):
        for j in range(cols):
            transpose[j][i] = matrix[i][j]
    return transpose

def input_matrix():
    matrix = []
    rows = int(input("Enter the number of rows: "))
    cols = int(input("Enter the number of columns: "))
    for i in range(rows):
        row = []
        for j in range(cols):
            element = int(input(f"Enter element at [{i}][{j}]: "))
            row.append(element)
        matrix.append(row)
    return matrix

def main():
    matrix = input_matrix()
    if matrix:
        transpose = mat_transpose(matrix)
        return transpose
    else:
        return None

if __name__ == "__main__":
    transpose = main()
    if transpose:
        print("Transpose of the matrix:")
        for row in transpose:
            print(row)
