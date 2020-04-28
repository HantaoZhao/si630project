import csv
import re
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
words = stopwords.words('english')

d={}
f = open("y_train.txt",encoding='utf-8')
label=[int(_) for _ in f]
pos_label_cnt=sum(label)
neg_label_cnt=len(label)-pos_label_cnt
p_pos=pos_label_cnt/(pos_label_cnt+neg_label_cnt)
p_neg=1.0-p_pos

def tokenize(s):
    # return s.strip().split(' ')
    # return [i for i in s.replace("@USER", "").strip().replace("’", "'").split(' ') if i not in words]
    return [i for i in s.strip().replace("’","'").split(' ') if i not in words]

def better_tokenize(s):
    # Very lower F1-score if use this
    # s = re.sub(r'[^\w]', ' ', s.lower())

    s = s.lower()
    s = re.sub(r'\!{2,}', '!', s)
    s = re.sub(r'\?{2,}', '?', s)
    s = re.sub(r'\*{2,}', '*', s)
    s = re.sub(r'\.{2,}', '.', s)
    s = re.sub(r'\s{2,}', ' ', s)
    return s.strip().split(' ')

def train(smoothing_alpha=0):
    for i in d.values():
        i[2]=((i[0]+smoothing_alpha)/(pos_word_cnt+smoothing_alpha*total_words))
        i[3]=((i[1]+smoothing_alpha)/(neg_word_cnt+smoothing_alpha*total_words))

def classify(s):
    ratio=p_pos/p_neg
    for i in s:
        if i not in d:
            continue
        if d[i][3]==0:
            if d[i][2]!=0:
                ratio *= 2**d[i][0]
            continue
        if d[i][2]==0:
            if d[i][3] != 0:
                ratio /= 2**d[i][1]
            continue
        ratio*=(d[i][2]/d[i][3])
    return ratio>=1

f = open("X_train.txt",encoding='utf-8')
cnt, pos_word_cnt, neg_word_cnt=0,0,0
for line in f:
    for word in tokenize(line):
        if label[cnt]:
            pos_word_cnt+=1
        else:
            neg_word_cnt+=1
        # pos_cnt, neg_cnt, pos_probability(tbd), neg_probability(tbd)
        d[word]=[d.get(word,[0,0])[0]+label[cnt],d.get(word,[0,0])[1]+1^label[cnt],0,0]
    cnt+=1

# print(d)
f = open("y_test.txt",encoding='utf-8')
label_dev=[int(_) for _ in f]
pos_dev=sum(label_dev)
neg_dev=len(label_dev)-pos_dev
total_words=len(d)

def plot():
    f1 = []
    p_list,r_list=[],[]
    for i in range(100):
        smooth=i/1000

        f = open("X_test.txt",encoding='utf-8')
        train(smooth)
        train_dev = [int(classify(tokenize(l))) for l in f]


        pos,neg=0,0
        for i in range(len(label_dev)):
            if train_dev[i]==label_dev[i]==1:
                pos+=1
            elif train_dev[i]==label_dev[i]==0:
                neg+=1
        p,r=pos/pos_dev,neg/neg_dev
        p_list.append(p)
        r_list.append(r)
        f1.append(2*p*r/(p+r))
        print("Precision:",p, "\tRecall:",r,"\tSmooth:", smooth, '\tNaive Bayes F1 Score:',2*p*r/(p+r))
    plt.figure('Alpha from 0 to 0.1, 100 steps')
    # plt.scatter([i / 100 for i in range(100)], f1)
    plt.plot([i / 1000 for i in range(100)], f1, label="F1 Score")
    plt.plot([i / 1000 for i in range(100)], p_list, label="Precision")
    plt.plot([i / 1000 for i in range(100)], r_list, label="Recall")

    plt.xlabel('Alpha')
    plt.legend()
    plt.show()
    f.close()

def token_plot():
    f1 = []
    f = open("X_train.txt",encoding='utf-8')
    d={}
    cnt, pos_word_cnt, neg_word_cnt = 0, 0, 0
    for line in f:
        for word in better_tokenize(line):
            if label[cnt]:
                pos_word_cnt += 1
            else:
                neg_word_cnt += 1
            # pos_cnt, neg_cnt, pos_probability(tbd), neg_probability(tbd)
            d[word] = [d.get(word, [0, 0])[0] + label[cnt], d.get(word, [0, 0])[1] + 1 ^ label[cnt], 0, 0]
        cnt += 1

    for i in range(100):
        smooth=i/1000

        f = open("X_test.txt",encoding='utf-8')
        train(smooth)
        train_dev = [int(classify(better_tokenize(l))) for l in f]

        pos,neg=0,0
        for i in range(len(label_dev)):
            if train_dev[i]==label_dev[i]==1:
                pos+=1
            elif train_dev[i]==label_dev[i]==0:
                neg+=1
        p,r=pos/pos_dev,neg/neg_dev
        f1.append(2*p*r/(p+r))
        print("Smooth:", smooth, ' Naive Bayes F1 Score:',2*p*r/(p+r))
    plt.figure('Alpha from 0 to 0.1, 100 steps')
    plt.scatter([i/100 for i in range(100)],f1)
    plt.show()
    f.close()
    print(d)


# smooth=0.0001
# token_plot()  #在训练集上测试

plot()

smooth=1/10000000


# f = open("X_test.txt",encoding='utf-8')
# train(smooth)
# train_dev = [int(classify(tokenize(l))) for l in f]
#
# csvFile = open("result.csv", "w+", newline='')
# writer = csv.writer(csvFile)
# writer.writerow(['Id', 'Category'])
# for i in range(len(train_dev)):
#     writer.writerow([i, train_dev[i]])
#
# f.close()