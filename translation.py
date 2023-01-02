

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