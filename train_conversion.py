import csv

from nltk.corpus import stopwords
words = stopwords.words('english')

filename="task_a_distant.tsv"
limit=50000
# limit=200000
text=[]
label=[]
with open(filename,encoding='utf-8') as f:
    f_csv = csv.reader(f, delimiter='\t')
    # headers = next(f_csv)
    cnt=0
    for row in f_csv:
        if cnt:
            # print(row)
            # print(row[1:])
            # text.append(row[1])
            text.append(" ".join(i for i in row[1].split() if i not in words))

            label.append(float(row[2]))
            # if float(row[2])>=0.5:
            #     print(row[1])
        if cnt>=limit:
            break
        cnt += 1

# for i in text:
#     print(i)

f = open('first50000_NoStopWord.tsv','w+',encoding='utf-8')
f.write("text")
f.write("\t")
f.write("label")
f.write("\n")
for i in range(len(text)):
    f.write(text[i])
    f.write("\t")
    f.write("0" if label[i]<=0.5 else "1")
    f.write("\n")

# print(label)
#
# f = open('X_train.txt','w+',encoding='utf-8')
# for i in text:
#     f.write(i)
#     f.write("\n")
#
# f = open('y_train_float.txt','w+',encoding='utf-8')
# for i in label:
#     f.write(str(i))
#     f.write("\n")
#
# f = open('y_train.txt','w+',encoding='utf-8')
# for i in label:
#     f.write("1" if i>=0.5 else "0")
#     f.write("\n")
