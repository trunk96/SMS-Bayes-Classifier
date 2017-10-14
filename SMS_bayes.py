from k_fold import k_fold_text
from textClassifyNaiveBayes import textClassifyNaiveBayes
import operator

dataset={}
vocabulary={}
with open("SMSSpamCollection", "r") as data_file:
  for line in data_file:
    string=line.split('\t', 1)

    value=string[0]  #take the ham/spam form each line
    key=string[1].strip()
    dataset[key]=value #dictionary with the string as key and the f(x) as value
    words=key.split(' ')
    for word in words:
      if word not in vocabulary:
        vocabulary[word]=1
      else:
        vocabulary[word]+=1

#starting K-FOLD cross validation
k=10
result=k_fold_text(k, dataset, ["spam", "ham"], vocabulary, True)
print("K-FOLD method with k=", k, " \nEstimated accuracy: ",result)


#here we can fun with the classifier :)
b=textClassifyNaiveBayes(dataset, ["spam", "ham"], vocabulary)
while(True):
  res=b.classify(input())
  classification=max(res.items(), key=operator.itemgetter(1))[0]
  print(classification)
