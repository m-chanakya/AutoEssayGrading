from numpy import zeros
from scipy.linalg import svd, inv
from math import log
from numpy import asarray, sum
from nltk import stopwords

#stopwords = ['and','edition','for','in','little','of','the','to']
ignorechars = ''',:'!'''
stopwords = stopwords.words('english')

class LSA(object):
    def __init__(self, stopwords, ignorechars):
        self.stopwords = stopwords
        self.ignorechars = ignorechars
        self.wdict = {}
        self.dcount = 0        
    def parse(self, doc):
        words = doc.split();
        for w in words:
            w = w.lower().translate(None, self.ignorechars)
            if w in self.stopwords:
                continue
            elif w in self.wdict:
                self.wdict[w].append(self.dcount)
            else:
                self.wdict[w] = [self.dcount]
        self.dcount += 1      
    def build(self):
        self.keys = [k for k in self.wdict.keys() if len(self.wdict[k]) > 1]
        self.keys.sort()
        self.A = zeros([len(self.keys), self.dcount])
        for i, k in enumerate(self.keys):
            for d in self.wdict[k]:
                self.A[i,d] += 1
    def calc(self):
        self.U, self.S, self.Vt = svd(self.A)
    def TFIDF(self):
        WordsPerDoc = sum(self.A, axis=0)        
        DocsPerWord = sum(asarray(self.A > 0, 'i'), axis=1)
        rows, cols = self.A.shape
        for i in range(rows):
            for j in range(cols):
                self.A[i,j] = (self.A[i,j] / WordsPerDoc[j]) * log(float(cols) / DocsPerWord[i])
    #def printA(self):
    #    print 'Here is the count matrix'
    #    print self.A

    #def printSVD(self):
    #    print 'Here are the singular values'
    #    print self.S
    #    print 'Here are the first 3 columns of the U matrix'
    #    print -1*self.U[:, 0:3]
    #    print 'Here are the first 3 rows of the Vt matrix'
    #    print -1*self.Vt[0:3, :]

mylsa = LSA(stopwords, ignorechars)
for t in titles:
    mylsa.parse(t)
mylsa.build()
#mylsa.printA()
#mylsa.calc()
mylsa.printSVD()
