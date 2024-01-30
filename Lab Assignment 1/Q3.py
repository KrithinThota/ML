#Write a program to find the number of common elements between two lists. The lists contain integers.
def common_elements(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    common = set1.intersection(set2)
    return len(common)

def get_list_input():
    num_elements = int(input("Enter the number of elements: "))
    print("Enter the elements separated by spaces:")
    elements = list(map(int, input().split()))
    return elements

def main():
    print("Enter elements for list 1:")
    list1 = get_list_input()

    print("Enter elements for list 2:")
    list2 = get_list_input()

    common_count = common_elements(list1, list2)
    return common_count

if __name__ == "__main__":
    common_count = main()
    print("Number of common elements:", common_count)
