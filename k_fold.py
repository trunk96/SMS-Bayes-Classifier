from textClassifyNaiveBayes import textClassifyNaiveBayes
import random
import operator

def k_fold_text(n, dataset, target_values, dictionary, verbose):
    subsets={}
    counters={}
    D=len(dataset)
    li=[]
    secure_random = random.SystemRandom()
    print("Starting K-FOLD cross validation...")
    for e in dataset:
        li.append(e)

    for i in range(0, n):   #create n subset for the n-fold cross validation
        subsets[i]=[]
        for j in range (0, D//n):
            selected=secure_random.choice(li)
            subsets[i].append(selected)
            li.remove(selected)
    if verbose: print(n, "subsets successfully created. Cardinality of each subset is", D//n, "\n\n")
    for i in range(0, n):
        if verbose: print("K-FOLD iteration ", i+1, "> ")
        counters[i]=0
        training={}
        for e in subsets[i]:
            training[e]=dataset[e]
        if verbose: print("\t\tTraining set successfully generated")
        test=[]
        for elem in subsets[(i+1)%n]:
            test.append(elem)
        for j in range (0, n):
            if j!=i:
                for e in subsets[j]:
                    test.append(e)
        if verbose:
            print("\t\tTest set successfully generated. Cardinality of test set is", len(test))
            print("\t\tStarting classifying element in test set...")
        b=textClassifyNaiveBayes(training, target_values, dictionary)
        for elem in test:
            res=b.classify(elem)
            classification=max(res.items(), key=operator.itemgetter(1))[0]
            if classification!=dataset[elem]:
                counters[i]+=1
        counters[i]/=len(test)
        if verbose: print("\t\tEstimated accuracy:", 1-counters[i], "\n\n")
    result=0
    for elem in counters:
        result+=counters[elem]
    result/=n
    return 1-result
