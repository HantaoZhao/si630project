import csv
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.sparse import *

from nltk.corpus import stopwords
words = stopwords.words('english')

def tokenize(s):
    return s.strip().split(' ')

d={}
f = open("y_train.txt",encoding='utf-8')
label=[int(_) for _ in f]
pos_label_cnt=sum(label)
neg_label_cnt=len(label)-pos_label_cnt

f = open("X_train.txt",encoding='utf-8')
s = set()
for line in f:
    for t in tokenize(line):
        s.add(t)

s=list(s)
cnt=0
for i in range(len(s)):
    d[s[i]]=cnt
    cnt+=1

print(d)
print(len(d))

cnt=0
f = open("X_train.txt",encoding='utf-8')
X = dok_matrix((50000, len(s)+1), dtype=np.int8)
beta=np.zeros(len(s)+1)

for line in f:
    for t in tokenize(line):
        X[cnt,d[t]]+=1
    X[cnt,len(s)]=1
    cnt+=1

X=X.tocsr()
# print(X)

Y=np.array(label)
# print(Y)
# print('Y sum:',np.sum(Y))

# X_dev starts here
cnt = 0
f = open("X_test.txt",encoding='utf-8')
X_dev = dok_matrix((10000, len(s) + 1), dtype=np.int8)
keys = d.keys()
for line in f:
    for t in tokenize(line):
        if t in keys:
            X_dev[cnt, d[t]] += 1
    X_dev[cnt, len(s)] = 1
    cnt += 1
X_dev = X_dev.tocsr()

f = open("y_test.txt")
label_dev = [int(_) for _ in f]
pos_dev=sum(label_dev)
neg_dev=len(label_dev)-pos_dev

def sigmoid(vec):
    # Broadcast sigmoid function to all elements in vec.
    return expit(vec)

def log_likelihood(beta,X,Y):
    # Calculates the Log Likelihood

    dot_prod=X.dot(beta)

    # dot_prod = X.dot(beta)
    # dot_prod=np.transpose(dot_prod)
    # print('dot_prod:',dot_prod)
    # print('dot_prod.shape:',dot_prod.shape)
    # print("np.sum(dot_prod):",np.sum(dot_prod))
    # print('ll:',Y.dot(dot_prod)+np.log(sigmoid(dot_prod)))

    temp=np.log(1+np.exp(dot_prod))
    # if np.sum(temp)<=np.sum(dot_prod):
    #     raise RuntimeError
    # print('temp.shape:', temp.shape)
    # print('np.sum(temp):',np.sum(temp))

    return np.sum(Y.dot(dot_prod))-np.sum(temp)
    # return Y.dot(dot_prod)-np.log(1+np.exp(dot_prod))

def compute_gradient(beta,X,Y):
    step1=X.dot(beta)
    step2=Y-sigmoid(step1)
    step3=np.transpose(X).dot(step2)
    return step3

def logistic_regression(X,Y,learning_rate,num_step):
    beta = np.zeros(len(s) + 1)
    print('len(s):',len(s))
    ll=[]
    for i in range(num_step):
        step1 = X.dot(beta)
        step2 = Y - sigmoid(step1)
        step3 = np.transpose(X).dot(step2)
        beta = beta + learning_rate * step3

        ll.append(log_likelihood(beta, X, Y))

    pred = X_dev.dot(beta)

    ans = []
    for i in pred:
        if i >= 0:
            ans.append(1)
        else:
            ans.append(0)
    print('pred.shape:', pred.shape)
    print('pred:', pred)
    print('ans:', ans)

    predict(ans)

    return ll

def predict(ans):
    pos, neg = 0, 0
    for i in range(len(label_dev)):
        if ans[i] == label_dev[i] == 1:
            pos += 1
        elif ans[i] == label_dev[i] == 0:
            neg += 1
    p, r = pos / pos_dev, neg / neg_dev
    # f1.append(2 * p * r / (p + r))
    print("Precision:",p, "\tRecall:",r, '\tLogistic Regression F1 Score:', 2 * p * r / (p + r))

# ll=logistic_regression(X,Y,5e-5,1000)
# ll=logistic_regression(X,Y,5e-5,1000)

ll=logistic_regression(X,Y,5e-5,1000)
ll_=logistic_regression(X,Y,5e-4,1000)
ll__=logistic_regression(X,Y,5e-6,1000)


# print(ll)
# print('len(ll):',len(ll))
# plt.scatter(list(range(len(ll))),ll)
# plt.scatter(list(range(len(ll_))),ll_)
# plt.scatter(list(range(len(ll__))),ll__)
# plt.show()

# Some Kaggle results

# alpha=1e-6
# N=50000
# 82.0%

# alpha=8e-6
# N=100000
# 83.7%

# alpha=5e-6
# N=100000
# 82.0%

# alpha=1e-5
# N=100000
# 84.1%

# alpha=2e-5
# N=100000
# 85.5%

# alpha=2e-5
# N=100000
# 85.9%

# alpha=5e-5
# N=100000
# 86.1%

# X_test starts here
#
# cnt=0
# f = open("X_test.txt")
# X_test = dok_matrix((8001, len(s)+1), dtype=np.int8)
#
# for line in f:
#     for t in tokenize(line):
#         if t in keys:
#             X_test[cnt,d[t]]+=1
#     X_test[cnt,len(s)]=1
#     cnt+=1
#
# X_test=X_test.tocsr()
#
# pred=X_test.dot(beta)
# ans=[]
# for i in pred:
#     if i>=0:
#         ans.append(1)
#     else:
#         ans.append(0)
# print('pred.shape:',pred.shape)
# print('pred:',pred)
# print('ans:',ans)
#
# csvFile = open("result_lr_2.csv", "w+", newline='')
# writer = csv.writer(csvFile)
# writer.writerow(['Id', 'Category'])
# for i in range(len(ans)):
#     writer.writerow([i, ans[i]])