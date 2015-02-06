# -*- coding: utf-8 -*-
"""
Solution for the Dota2 challenge. See readme.
"""

import numpy
from sklearn import ensemble
from sklearn import cross_validation
import os


# first step: create a hash map/dictionary of all characters
encounternr=0
chardict={}
f = open(os.path.join(os.getcwd(), 'trainingdata.txt'), 'r')
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
X = numpy.zeros(shape=([14926,fvecsize*2])) # do wc -l trainingdata.txt
y = numpy.zeros(shape=([14926,1]))
linenr=0
f = open(os.path.join(os.getcwd(), 'trainingdata.txt'), 'r')
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
clf = ensemble.RandomForestClassifier(n_estimators=100,max_features=None,min_samples_leaf=2)
clf.fit(X, y2)
clf.score(X,y2)
# 0.98433333333333328



# last step: test data
X_test = numpy.zeros(shape=([74,fvecsize*2]))
y_test = numpy.zeros(shape=([74,1]))
linenr=0
f = open(os.path.join(os.getcwd(), 'testdata.txt'), 'r')
for line in f:
    chars=line.split(',')
    for chnr in xrange(0,size(chars)-1):
        # now distinguish: <=4 or higher
        ch=chars[chnr]
        X_test[linenr,chardict[ch]+((chnr<=4)*fvecsize)]=1
        
    y_test[linenr]=int(chars[-1]) # result: int(chars[-1])
    linenr+=1


y_test=numpy.ravel(y_test-1)

y_test_spec=clf.predict(X_test)
print(sum(y_test_spec==y_test)/size(y_test))
#0.51351351351351349



