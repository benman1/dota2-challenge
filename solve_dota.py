# -*- coding: utf-8 -*-
import numpy
from sklearn import svm
from sklearn.svm import SVC


# first step: create a hash map/dictionary of all characters
encounternr=0
chardict={}
f = open('/tmp/guest-Zx1l9V/Desktop/Dota2/training.txt', 'r')
for line in f:
    chars=line.split(',')
    for chnr in xrange(0,size(chars)-1):
        ch=chars[chnr]
        # if already known ignore, otherwise put in dictionary
        if not chardict.has_key(ch):
            chardict[ch] = encounternr
            encounternr+=1


f.close()


# second step: read data and write them to feature vectors
fvecsize=len(chardict.keys())
X = numpy.zeros(shape=([14926,fvecsize*2]))
y = numpy.zeros(shape=([14926,1]))
linenr=0
f = open('/tmp/guest-Zx1l9V/Desktop/Dota2/training.txt', 'r')
for line in f:
    chars=line.split(',')
    for chnr in xrange(0,size(chars)-1):
        # now distinguish: <=4 or higher
        ch=chars[chnr]
        X[linenr,chardict[ch]+((chnr<=4)*fvecsize)]=1

    y[linenr]=int(chars[-1]) # result: int(chars[-1])
    linenr+=1


y2=numpy.ravel(y-1)

# third step: classifier
clf = svm.SVC()
clf.fit(X, y2)

clf.score(X,y2)
# 0.61771405600964757


# last step: test data
X_test = numpy.zeros(shape=([74,fvecsize*2]))
linenr=0
f = open('/tmp/guest-Zx1l9V/Desktop/Dota2/test.txt', 'r')
for line in f:
    chars=line.split(',')
    for chnr in xrange(0,size(chars)-1):
        # now distinguish: <=4 or higher
        ch=chars[chnr]
        X_test[linenr,chardict[ch]+((chnr<=4)*fvecsize)]=1

    linenr+=1


y_test_spec=clf.predict(X_test)


