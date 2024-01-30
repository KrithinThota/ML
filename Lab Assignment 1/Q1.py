#Write a program to count the numberof vowels and consonants present in an input string.
def check(string):
    vowels_count = 0
    consonants_count = 0
    string=string.lower()
    vowels = "aeiou"
    for char in input_string:
        if char.isalpha():
            if char in vowels:
                vowels_count += 1
            else:
                consonants_count += 1
    return vowels_count,consonants_count

if __name__ == "__main__":
    input_string=input("Enter Input string")
    print(check(input_string))
