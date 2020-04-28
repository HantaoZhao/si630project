f = open('y_train.txt','r',encoding='utf-8')
ans=[]
for i in f:
    i=i.strip()
    if i!="\n":
        if i=="0":
            ans.append(0)
        if i=="1":
            ans.append(1)
        # print(i)
        # print(type(i))

print(sum(ans)/len(ans))