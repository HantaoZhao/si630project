import csv

filename="task_a_distant.tsv"
limit=50000
limit+=10000
text=[]
label=[]
with open(filename,encoding='utf-8') as f:
    f_csv = csv.reader(f, delimiter='\t')
    # headers = next(f_csv)
    cnt=0
    for row in f_csv:
        if 50000<cnt:
            # print(row[1:])
            text.append(row[1])
            label.append(float(row[2]))
            # if float(row[2])>=0.5:
            #     print(row[1])
        if cnt>=limit:
            break
        cnt += 1

print(text)
print(label)

f = open('X_test.txt','w+',encoding='utf-8')
for i in text:
    f.write(i)
    f.write("\n")

f = open('y_test_float.txt','w+',encoding='utf-8')
for i in label:
    f.write(str(i))
    f.write("\n")

f = open('y_test.txt','w+',encoding='utf-8')
for i in label:
    f.write("1" if i>=0.5 else "0")
    f.write("\n")
