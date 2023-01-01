

path = "D:\Python_Codes\Deep_Learning\Term 2\Translation\spa.txt"
def load_doc(path): # load dataset
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")[-1] # read data --> split them every line --> don't read the last line because it's empty
    return lines