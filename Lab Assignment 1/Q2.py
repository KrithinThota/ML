#Write a program that accepts two matrices A and Bas input and returns their product AB. Check if A & B aremultipliable; if not, return error message.
def check(A, B):
    if len(A[0]) != len(B):
        return False
    return True

def multiply(A, B):
    result = []
    for i in range(len(A)):
        row = []
        for j in range(len(B[0])):
            total = 0
            for k in range(len(B)):
                total += A[i][k] * B[k][j]
            row.append(total)
        result.append(row)
    return result

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
    print("Enter matrix A:")
    A = input_matrix()
    print("Enter matrix B:")
    B = input_matrix()
    if check(A, B):
        product = multiply(A, B)
        return product
    else:
        return "Error: Matrices A and B cannot be multiplied."

if __name__ == "__main__":
    result = main()
    print("Matrix product AB:")
    for row in result:
        print(row)
