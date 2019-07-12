# This program will format sentences for training a naive bayes classifier
import json

# Number of sentences this session
# repeat = int(input("# of sentences: "))
print("Press 0 to terminate")

# Append to file for easy copy-paste
f = open("sentences.txt", "a")

sentence = ""

while(1):
    sentence = input("Sentence: ")
    if(sentence == "0"):
        break
    list = sentence.lower().split(" ")
    # json.dumps outputs list with double quotes instead of single
    f.write(json.dumps(list) + ",\n")

f.close()