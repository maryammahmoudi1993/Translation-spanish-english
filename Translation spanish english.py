import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import random
import string
import tensorflow as tf
import re

#  ----------------------------------Data and Preprocessing-------------------------------------- 
# variables
vocab_size = 15000 # length of all data(vocabularies)
sequence_length = 20 # length of each sentence
# Load dataset
path = "D:\Python_Codes\Deep_Learning\Term 2\Translation spanish english\spa.txt"
print(path)
def load_doc(path): # load dataset
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")[:-1] # read data --> split them every line --> don't read the last line because it's empty
    return lines
lines = load_doc(path)

# Esp-Eng pairs
def creat_pairs(lines): 
    text_pairs = []
    for line in lines:
        english, spanish, other = line.split("\t") # creating english and spanish words by camma
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
    num_train_data = int(train_per * (len(text_pairs))) # number of train data
    #num_test_data = test_per * (len(text_pairs)) # number of test data
    num_val_data = int(val_per * (len(text_pairs))) # number of validation data
    train_pairs = text_pairs[:num_train_data] # train data from 0 to the end of train data numbers
    val_pairs = text_pairs[num_train_data:num_train_data + num_val_data] # validation data from train numbers to the sum of train and validation numbers
    test_pairs = text_pairs[num_train_data + num_val_data:] # test data from sum of train and validation numbers to the end
    return train_pairs, val_pairs, test_pairs
train_pairs, val_pairs, test_pairs = split_data(text_pairs)

# Standardization
def standardization(input_string):
    strip_char = string.punctuation + "¿" + "¿" + "á" + "é" + "í" + "ó" + "ú" + "ñ" + "ü" # add accent letters in spanish
    strip_char = strip_char.replace("[", "") # change [ with nothing for [start] and [end]
    strip_char = strip_char.replace("]", "") # change ] with nothing for [start] and [end]
    lowercase = tf.strings.lower(input_string)

    return tf.strings.regex_replace(lowercase, f"[{re.escape(strip_char)}]", "") # remove any punctuations from input

# Tokenization
def tokenization(train_pairs):
    source_vectorization = tf.keras.layers.TextVectorization(max_tokens=vocab_size, output_mode='int', output_sequence_length=sequence_length) # English vectorization
    target_vectorization = tf.keras.layers.TextVectorization(max_tokens=vocab_size, output_mode='int', output_sequence_length=sequence_length+1,
                                                            standardize = standardization) # Spanish vectorization
    

    train_english_text = [pair[0] for pair in test_pairs] # all english sentences: [eng1, eng2, ...]
    train_spanish_text = [pair[1] for pair in test_pairs] # all spanish sentences: [esp1, esp2, ...]

    source_vectorization.adapt(train_english_text) # vectorization english tokens
    target_vectorization.adapt(train_spanish_text) # vectorization spanish tokens
    print("[INFO] data tokenized and converted to int numbers")

    return source_vectorization, target_vectorization
source_vectorization, target_vectorization = tokenization(train_pairs) 


#  ----------------------------------Define Network And Training--------------------------------------
def format_dataset(eng, esp): # prepareing input and output 
    eng = source_vectorization(eng)
    esp = target_vectorization(esp)
    return ({"english": eng, "spanish": esp[:, :-1]}, esp[:, 1:]) # label

