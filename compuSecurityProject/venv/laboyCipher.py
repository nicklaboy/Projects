from random import random, randint
def encrypt(message):
    hold = ''
    encrypted= []
    key1 = randint(0,10000)
    print("Key 1: ",key1)
    #Shift cipher
    for i in range(len(message)):
        char = message[i]
        if char == " ":
            hold = hold + char
            continue
        if char.isupper():
            hold += chr((ord(char) + key1 -65) % 26 +65)
            continue
        else:
            hold += chr((ord(char) + key1-97) % 26 + 97)
    print("caesar cipher: " + hold)
    key2 = randint(0,50)
    #ASCI shift
    print("Key 2: " ,key2)
    for i in hold:
        x = ord(i) - key2
        encrypted.append(x)
    print("Encrypted message: ")
    for i in encrypted:
        print(i, end='')

    print()
    decrypt(message, encrypted, key1, key2)

def decrypt(message, encrypted, key1, key2):
    decrypted = ""
    for i in encrypted:
        x = int(i)
        x = x + key2
        x = chr(x)
        decrypted += x
    print("Reverted casear:" ,decrypted)
    hold = ''
    for i in range(len(decrypted)):
        char = decrypted[i]
        if char == " ":
            hold += char
            continue
        if char.isupper():
            hold += chr((ord(char) - key1 -65) % 26 + 65)
            continue
        else:
            hold += chr((ord(char) - key1 -97) % 26 + 97)
    decrypted = hold
    # if decrypted != message:
    #     decrypted = message
    print("Decrpyted message:", decrypted)
def main():

    message = input("Input message: ")
    encrypt(message)

if __name__ == '__main__':
    main()