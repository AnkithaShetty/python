fpath ="D:\PythonProjects\SentimentAnalysis\input\pos.txt"


def file_len(fpath):
    with open(fpath) as f :
        for i,l in enumerate(f):
            pass
        return i+1

def word_count(fpath):
    for line in open(fpath):
        tokens = line.split()
        return tokens.count("The")


#linecount = file_len(fpath)
#print (linecount)
#wordcount = word_count(fpath)
#print (wordcount)
