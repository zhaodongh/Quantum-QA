import ConfigParser 
import numpy as np
import random 
from sklearn.utils import shuffle
import pandas as pd 

# test_1 = cf.options('sec_test')

a = [2,4,5,6,7]
b = [1,2,3,4,5]

import theano
import theano.tensor as T 

# m = [np.array(a), np.array(b)]

# index = T.lscalar('index')
# mat = T.vector('mat')

# f = theano.function([index, mat], mat[index])

# for i in range(5):
# 	print f(i, a)

# print m
# for i, _ in enumerate(m):
# 	m[i] = shuffle(m[i], random_state = 121)

# print m


# a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

# print a
# print np.trace(a, axis1=1, axis2=2)

# for b in a:
# 	print np.trace(b)

a = np.array([[1,2,3],[4,5,6],[7,8,9]])
b = a.copy()
c = np.array([1,2,3])
d = c.copy()

A = T.matrix()
B = T.matrix()
C = T.vector()

D = theano.shared(d)
# f = theano.function([A, B], A*B)
# print f(a, b)

# pro, _= theano.scan(fn = lambda x, y:x * y, sequences=[A, C])
# f = theano.function([A, C], pro)
# print f(a, c)
# E = D.repeat(3)
# print D.get_value()
# f = theano.function([], E)
# print f()

# a = np.ones(10)
# print a.sum() * a / len(a)
# print 10 * a

# df = pd.read_csv('test.result.csv', header=None,sep="\t",names=['qid','aid',"question","answer","flag",'score'],quoting =3)
# df1 = df.sort(['qid'])
# print df1

# a = np.ones(10, dtype=np.float32)
# b = np.array(a, dtype=np.float32)
# print a
# print type(a[0])
# print b
# print type(b[0])
# c = np.float(0.01)
# print type(c)


# a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 0], [0, 0]]], dtype=theano.config.floatX)
# print a.shape
# print len(a)

# b = np.array([[1, 2], [3, 4]], dtype=theano.config.floatX )
# print b.shape
# print b

# A = T.tensor3('A')
# B = T.matrix('B')

# f = theano.function([A, B], T.dot(A, B))

# print f(a, b).shape
# print f(a,b)

# a = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
# print a
# for index in a:
# 	a[index] -= 1
# print a

a = np.array([[1.,2.,3.],[4.,5., 6.]])
# b = np.ones(5)

# print a

# for index, _ in enumerate(a):
# 	a[index] -= 1
# print a
# print a
# print a.shape
# print type(a)

# b = np.array(a, dtype=np.float32)
# print b
# print b.shape
# print type(b)
# print type(a[0])
# a = a.astype(np.float32)
# print a.shape
# print type(a[0][0])
# print a

import pandas as pd

train_fname = open('data/trec/data_set/test')
df_train= pd.read_csv(train_fname, header=None,sep="\t",names=['qid','aid',"question","answer","flag"],quoting =3)

df_train.to_csv('test.result.csv', header=None,sep="\t",index=False, columns=['qid','aid',"question","answer","flag"],quoting =3)
# print df_train.sort_values('qid')
# print help(df_train.to_csv)
