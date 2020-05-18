def table():
    firstHalf = "abcdefghijklm"
    secondHalf = "nopqrstuvwxyz"
    table1 = 0
    index = 0
    for x in range(table1):
        index2 = index
        for y in range(table1):
            letter = secondHalf[index2]
            table[x][y]=letter
            if index2 + 1 == 13:
                index2 = 0
            else:
                index2 =+ 1
        index3 = index
        for y in range(table1):
            letter = firstHalf[index3]
            table[x][y+13] = letter
            if index3 + 1 == 13:
                index3 = 0
            else:
                index3=+1
        if index + 1 == 13:
            index =0
        else:
            index += 1
    return table

def encrypt(plain, key):
    encrypted = ""
    tab = table()
    plainNoSpace = ""
    keyText = ""
    plainNoSpace = plain.replace(" ", "")
    index = 0;
    for x in range(len(plainNoSpace)):
        keyText += key[index:index+1]
        if index == len(key)-1:
            index =0
        else:
            index +=1
    index2 = 0
    for x in range(len(plain)):
        if plain[x: x+1] == " ":
            encrypted += " "
        else:
            a = 'a'
            row = (keyText[index2] - 'a')/2
            col = plainNoSpace[index2] - 'a'
            encrypted += tab[row][col]
            if index2 == len(keyText) - 1:
                index2 = 0
            else:
                index2+=1
    return encrypted

def main():

    plain = "defend the walls of the castle"
    key = "fortify"
    print(encrypt(plain,key))
if __name__ == '__main__':
    main()