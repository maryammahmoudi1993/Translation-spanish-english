import random

# Data and Preprocessing 
# Load dataset
path = "D:\Python_Codes\Deep_Learning\Term 2\Translation spanish english\spa.txt"
print(path)
def load_doc(path): # load dataset
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")[-1] # read data --> split them every line --> don't read the last line because it's empty
    return lines
lines = load_doc(path=path)

# Esp-Eng pairs
def creat_pairs(lines): 
    text_pairs = []
    for line in lines:
        english, spanish = line.split("\t") # creating english and spanish words by camma
        spanish = "[start]" + spanish + "[end]" # Add [start] and [end] to spanish words
        text_pairs.append((english, spanish))
        return text_pairs
text_pairs = creat_pairs(lines)

# Split data to train-test-validation
train_per = 0.7
test_per = 0.15
val_per = 0.15
def split_data(text_pairs):
    random.shuffle(text_pairs) # shuffle text data
    num_train_data = train_per * (len(text_pairs)) # number of train data
    #num_test_data = test_per * (len(text_pairs)) # number of test data
    num_val_data = val_per * (len(text_pairs)) # number of validation data
    train_pairs = text_pairs[:num_train_data] # train data from 0 to the end of train data numbers
    val_pairs = text_pairs[num_train_data:num_train_data + num_val_data] # validation data from train numbers to the sum of train and validation numbers
    test_pairs = text_pairs[num_train_data + num_val_data:] # test data from sum of train and validation numbers to the end
    return train_pairs, val_pairs, test_pairs
train_pairs, val_pairs, test_pairs = split_data(text_pairs)
