import time
start_time = time.time()
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.mllib import *
from pyspark.mllib.linalg import *
from pyspark.mllib.linalg import Vectors, DenseMatrix
from pyspark.mllib.linalg.distributed import IndexedRowMatrix
from pyspark.mllib.linalg import Matrix, Matrices
from pyspark.mllib.linalg.distributed import *
from scipy.sparse import *


from scipy import *
import scipy.io as sio
import numpy as np
from scipy import linalg
from numpy import *

from decimal import *




def revised_simplex(h, f, b, m, n, basis, nonbasis):
    k = True
    B = f[:, basis]
    B = linalg.inv(B)
    Pxbb = B.dot(b).flatten()


    counter = 0
    while k:


        counter = counter + 1
        print counter

        cD = h[nonbasis]
        cB = h[basis]
        # print f
        # print basis

        B = f[:, basis]
        #print 'Basis of A with transpose'
        #print B.transpose()
        B = linalg.inv(B)
        #Pxbb = B.dot(b).flatten()
        #print 'inverse of B'
        #print B
        D = f[:, nonbasis]
        bs = Matrices.dense(m, 1, b.flatten().tolist())
        blocks0 = sc.parallelize([((0, 0), bs)])
        mat0 = BlockMatrix(blocks0, m, 1)

        dm1 = Matrices.dense(m, m, f[:, basis].flatten().tolist())  # A matrix basis indices chosen
        blocks1 = sc.parallelize([((0, 0), dm1)])
        mat1 = BlockMatrix(blocks1, m, m)
        mat1 = mat1.transpose()
        mat1.toLocalMatrix()
        # print mat1.toLocalMatrix()

        mat2 = IndexedRowMatrix(sc.parallelize(enumerate(f[:, nonbasis]))).toBlockMatrix(rowsPerBlock=m, colsPerBlock=n)
        # print (mat2.toLocalMatrix())

        G = mat1.toLocalMatrix()  # G is basis stored
        K = mat2.toLocalMatrix()

        # print (G)  # It will display Basis Matrix
        # print (K)

        dm2 = Matrices.dense(m, m, B.flatten().tolist())  # Inverse stored in dm2
        blocks2 = sc.parallelize([((0, 0), dm2)])  # Inverse B converted to blocks
        mat3 = BlockMatrix(blocks2, m, m)

        mat3 = mat3.transpose()
        L = mat3.toLocalMatrix()
        dm3 = Matrices.dense(1, m, h[basis].tolist())  # Cost vector C, basis stored in dm3
        blocks4 = sc.parallelize([((0, 0), dm3)])  # 'c' basis is stored in blocks4
        mat4 = BlockMatrix(blocks4, 1, m)  # 'c' stored as BlockMatrix
        S = mat4.toLocalMatrix()
        # print (S)

        dm4 = Matrices.dense(1, n, h[nonbasis].tolist())  # Cost vector C, non-basis stored in dm5
        blocks5 = sc.parallelize([((0, 0), dm4)])  # 'c' non-basis is stored in blocks5
        mat6 = BlockMatrix(blocks5, 1, n)  # 'c' stored as BlockMatrix
        R = mat6.toLocalMatrix()

        # print (R)

        La = mat4.multiply(mat3).toLocalMatrix()  # c is basis matrix, multiply by matrix B inverse. In main program it is "l = cB.dot(B)"
        # print (La)

        blocks6 = sc.parallelize([((0, 0),
                                   La)])  # this step is done to store La in mat variable so that it would be easy to use it for further multiplication
        mat7 = BlockMatrix(blocks6, 1,
                           m)  # from main program "l = cB.dot(B)" is stored in "mat 7" for future multiplication
        Sa = mat7.toLocalMatrix()
        # print (Sa)

        ga = mat7.multiply(
            mat2).toLocalMatrix()  # multiply "l = cB.dot(B)" by 'D' where 'D' is Matrix A's non basis. Here 'mat3'

        # print (ga)

        blocks7 = sc.parallelize([((0, 0), ga)])  # this step is done to store 'ga' in mat8
        mat8 = BlockMatrix(blocks7, 1, n)

        Cd = mat6.subtract(mat8).toLocalMatrix()

        #print 'mat6='
        #print mat6.toLocalMatrix()

        #print 'mat7'
        #print mat7.toLocalMatrix()

        #print 'mat2'
        #print mat2.transpose().toLocalMatrix()

        #print 'mat4'
        #print mat4.toLocalMatrix()

        #print 'mat3'
        #print mat3.toLocalMatrix()

        #print 'mat8='
        #print mat8.toLocalMatrix()

        ma = Cd.toArray()
        # maa = np.around(ma, decimals= 10)
        print 'ma ='
        print ma

        # print "printing Cd"
        minrD = np.argmin(ma)
        #print 'minimum index of maa is'

        print (minrD)

        do = minrD  # We get value 0

        Dxx = D[:, do]

        Dx = Matrices.dense(m, 1,
                            Dxx.tolist())  # the index of minimum of minrD is used to call matrix D's elements which we will parallelize
        blocks8 = sc.parallelize([((0, 0), Dx)])  # store Dx it in blocks8
        mat9 = BlockMatrix(blocks8, m, 1)  # Convert to blockmatrix and store in mat9
        Aa = mat9.toLocalMatrix()

        Pa = mat3.multiply(
            mat9).toLocalMatrix()  # Inverse of B multiply by Dx( where Dx = D[:, n] where D = A[:, nonbasis]
        Pxb = mat3.multiply(mat0).toLocalMatrix()

        #print (Pa)
        #print (Pxb)
        Paa = B.dot(Dxx)
        # Pxbb = B.dot(b)

        # Paaa = np.around(Paa, decimals= 16)
        # Pxbbb = np.around(Pxbb, decimals=16)


        print 'This is Paa'
        print Paa


        # abc = np.divide(Pxbb, Paa)
        # print (abc)
        # with np.errstate(divide='ignore'):

        abc = inf * np.ones(len(Pxbb))
        abcd = inf * np.ones(len(Pxbb))
        # print 'len(Paa) is'
        # print len(Paa) - 1
        for idx in range(0, len(Paa)):
            # print idx

           if Paa[idx] > 1e-12:
             abc[idx] = Pxbb[idx] / Paa[idx]

             print 'this is Pxbb before update '
             print Pxbb

        Qa = np.argmin(abc)
        #Qa = np.argmin(abc[np.nonzero(abc)])
        Pxbb = Pxbb - np.multiply(np.amin(abc), Paa).transpose()
        print np.multiply(np.amin(abc), Paa)


        Pxbb[Qa] = np.amin(abc)

        #for idx in range(0, len(Paa)):
            #if Paa[idx] > 0:
                #abcd[idx] = Pxbb[idx] / Paa[idx]

        print 'this is Paa after update'
        print Paa


        print 'this is Pxbb after updating'
        print Pxbb

        print 'abc with updated Pxbb'
        print abc



        #Qc = np.argmin(abcd[np.nonzero(abcd)])

        #print 'do = The leaving variable index'
        #print do

        #print 'np.argmin(abc) is the entering variable index'
        #print Qa

        #print 'printing nonbasis do'
        #print nonbasis[do]

        object = h[basis]

        print 'printing Qa='
        print Qa

        final = basis
        k = np.any(ma < -0.00000000001)
        if k == False:
            break

        temp = basis[Qa]
        basis[Qa] = nonbasis[do]
        nonbasis[do] = temp

        #print 'Cd ='
        #print (Cd)

        print 'nonbasis ='
        print nonbasis

        print 'basis ='
        print basis

        # print shape(basis)

        #ma = Cd.toArray()

        #print 'ma ='
        #print ma

        # print k
        # print 'Pxbb ='
        #print type(Pxbb)

    zzz = np.inner(h[basis], Pxbb)

    solution = [zzz, basis, Pxbb]
    return solution


# print revised_simplex(h, f, b, basis)
# print revised_simplex(h, f, b, basis)

sc = SparkContext()
#spark = SparkSession(sc)
spark = SparkSession.builder.appName("farm").getOrCreate()
# J = sio.loadmat('/Users/rahulkhanolkar/Desktop/farm.mat')
J = sio.loadmat('/Users/rahulkhanolkar/Downloads/p0201.mat')
K = J['Problem']

# print K.dtype

# print K
# print K[0][0][7]
# fs = np.shape(K[0][0][7])
A = K['A']
B = K['b']
# aux = K['aux']
# C = aux[0]
fs = np.shape(A)
# print fs
aux2 = K['aux']
# aux2 = aux['aux']
c = aux2[0][0]['c'][0][0].flatten()
#print 'c='
#print c

# a = csr_matrix(K[0][0][7], shape=(m, n)).toarray()

# print 'A[0]'

##print A[0][0]
# a = csc_matrix(A).toarray()
a = A[0][0]
a = a.toarray()

fs = np.shape(a)
# print 'fs'
# print fs

m = fs[0]
n = fs[1]
o = m + n
# print a
# print B
b = B[0][0]
#print 'b='
#print b
# RHS of AX+ BY = C

# print C
# c = C[0][0][0]
# print c

# c = csr_matrix(K[0][0][9][0][0][0], shape=(n, 1)).toarray()
d = c.transpose()
e = np.identity((m))
# print shape(a)
# print shape(e)
f = np.concatenate((a, e), axis=1)
# print f# actual matrix with joined identity matrix

g = np.zeros(m)
ha = np.zeros(n)
hb = np.ones(m)
h = np.append(ha, hb)
#print 'h = '
#print h

# np.append(d, g)                                              # Minimize condtion matrix


ii = np.arange(0, n)
jj = np.arange(n, o)

# nonbasis = ii
basis = jj
# nonbasis = ii
nonbasis = np.setdiff1d(np.arange(n), basis)
#B = f[:, basis]
#B = linalg.inv(B)

# print basis
# print shape(basis)

auxiliarySolution = revised_simplex(h, f, b, m, n, basis, nonbasis)
# print basis
n2 = n - m
# print n2


basis = auxiliarySolution[1]
print 'basis after auxillary solution 1'
print auxiliarySolution[1]

nonbasis = np.setdiff1d(np.arange(n), basis)

# print nonbasis
# print basis
initialBasis = basis
print nonbasis

print 'objective value phase 1 is'
print auxiliarySolution[0]
print initialBasis
print c
print a
print b
print m
print n2

solution = revised_simplex(c, a, b, m, n2, initialBasis, nonbasis)
basis = solution[1]

#print c

optimalvalue = solution[0]

print 'optimal value is'
print optimalvalue
print basis
print nonbasis
print(time.time() - start_time)
spark.stop()















