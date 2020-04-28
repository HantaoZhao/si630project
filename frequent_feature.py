import csv
from collections import Counter

from nltk.corpus import stopwords
words = stopwords.words('english')

filename="task_a_distant.tsv"
limit=50000
# limit+=10000
text=[]
label=[]
with open(filename,encoding='utf-8') as f:
    f_csv = csv.reader(f, delimiter='\t')
    cnt=0
    for row in f_csv:
        if 0<cnt:
            text.append(row[1])
            label.append(float(row[2]))
        if cnt>=limit:
            break
        cnt += 1

temp=[]
for i in text:
    for j in i.split():
        temp.append(j)

ans=Counter(temp)
ans=sorted(ans,key=lambda i:-ans[i])
for i in ans:
    if i not in words:
        print(i)
# print(ans)