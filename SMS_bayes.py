from k_fold import k_fold_text
from textClassifyNaiveBayes import textClassifyNaiveBayes
import operator
import os

dataset={}
vocabulary={}
path=os.path.join("collection","SMSSpamCollection")
with open(path, "r") as data_file:
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
k=100
result=k_fold_text(k, dataset, ["spam", "ham"], False)
print("K-FOLD method with k=", k, " \nEstimated accuracy: ",result)


<<<<<<< HEAD
=======
#here we can fun with the classifier :)
print("\n\nNow you can classify any SMS...")
b=textClassifyNaiveBayes(dataset, ["spam", "ham"], vocabulary)
text=""
while(text!="end"):
  text=input()
  if text!="end":
    res=b.classify(text)
    classification=max(res.items(), key=operator.itemgetter(1))[0]
    print(classification)
>>>>>>> c14a23df945a63eef41821228a85aa6f0a92dd9f
