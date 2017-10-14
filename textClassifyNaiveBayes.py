class textClassifyNaiveBayes:
    docs={}
    docs_words={}
    Pcj={}
    Pwi={}
    

    def __init__(self, dataset, target_values, dictionary):
        D=len(dataset)
        V=len(dictionary)
        self.target_values=target_values
        self.dictionary=dictionary
        for value in target_values:
            self.docs[value]=[]
            self.docs_words[value]=0
            self.Pwi[value]={}
            for data in dataset: #calculating probability of cj
                if dataset[data]==value:
                    self.docs[value].append(data)
                    self.docs_words[value]+=len(data.split(' '))
            self.Pcj[value]=len(self.docs[value])/D

            cjdictionary={} #creating dictionary of words from a certain value cj
            for data in self.docs[value]:
                text=data.strip().split(' ')
                for word in text:
                    if word in cjdictionary:
                        cjdictionary[word]+=1
                    else:
                        cjdictionary[word]=1
                       
            for word in self.dictionary: #calculating conditional probability of each word given cj
                if word in cjdictionary:
                    (self.Pwi[value])[word]=((cjdictionary[word]+1)/(self.docs_words[value]+V))
                else:
                    (self.Pwi[value])[word]=(1/(self.docs_words[value]+V))

                    
    def classify(self, new_data):
        vnb={}
        for value in self.target_values:
            vnb[value]=self.Pcj[value]
            words=new_data.strip().split(' ');
            for word in words:
                if word in self.dictionary:
                    vnb[value]*=(self.Pwi[value])[word]
        return vnb
    
            
            
        
                
