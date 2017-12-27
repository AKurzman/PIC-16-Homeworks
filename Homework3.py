##Homework 3

import itertools
import operator
from difflib import SequenceMatcher
import numpy as np
from numpy import linalg as LA

def corrcoeff(x,y):
    xnew=x    ##copies of x,y
    ynew=y
    for i in range(0,len(x)):
        xnew[i]-=(sum(x)/len(x))  ##plug in to correlation coeffecient formula given
        ynew[i]-=(sum(x)/len(x))
    return np.dot(xnew,ynew)/(LA.norm(xnew,2)*LA.norm(ynew,2))

def degree(n):
    count=0
    mat=np.matrix(n)
    finalvec=[]
    hightolow=[]
    high=0
    for j in range(0,len(mat)):
        for k in range(0,len(mat)):
            if mat[k,j]>0:   ##if n[k,j] has a value, aka an edge/connection, increase count
                count+=1
        finalvec.append(count)  ##add total amount in column to final vector
        count=0
    a=finalvec
    for x in range(0,len(finalvec)):   ##search through finalvec for maximum,when found it will then display the location and eliminate those from the search
        high = max(a)
        hightolow.append([i for i,x in enumerate(finalvec) if x == high])
        a = [x for x in a if x != high]
        if a==[]:
            break
    hightolow=list(itertools.chain.from_iterable(hightolow))
    return finalvec,hightolow

def strength(n):  ##same method as above but instead of a count we found the total
    finalvec=[]
    hightolow=[]
    mat=np.matrix(n)
    high=0
    total=0
    for j in range(0,len(mat)):
        for k in range(0,len(mat)):
            total+=mat[k,j]
        finalvec.append(total)
        total=0
    a=finalvec
    for x in range(0,len(finalvec)):
        high = max(a)
        hightolow.append([i for i,x in enumerate(finalvec) if x == high])
        a = [x for x in a if x != high]
        if a==[]:
            break
    hightolow=list(itertools.chain.from_iterable(hightolow))
    return finalvec,hightolow

def pagerank(n, p=0.15):
    rsum=0
    esum=0
    mat=np.matrix(n)
    vec1={}
    vec2={}
    np1=mat.astype(float)
    for i in range(0,len(np1)):
        for j in range(0,len(np1)):
            np1[i,j]+=p  ##add standard constant p,0.15 as specified by online, to all values of network
            rsum+=np1[i,j]
        vec1[i]=rsum  ##append vec1 with row total, aka how many total links to said node
        rsum=0
    length=len(np1)
    nt=np.hsplit(np1,length)  ##to find out nt, we split it up in to columns, normalize each one then recombine
    for i in range(0,len(nt)):
        nt[i]=nt[i]/LA.norm(nt[i],1)
    nt=np.concatenate(nt,1)
    eigvals,eigvecs=LA.eig(nt)
    
    length2=len(eigvecs)
    prm=np.hsplit(eigvecs,length2)  ##reapeat same process again to find our page rank matrix
    for i in range(0,len(prm)):
        prm[i]=prm[i]/LA.norm(prm[i],1)
    prm=np.concatenate(prm,1)
    for i in range(0,len(prm)):
        for j in range(0,len(prm)):
            if prm[i,j]<0:
                prm[i,j]*=-1  ##get all values positive
            esum+=prm[i,j]
        vec2[i]=esum  ##same logic as before, add up values in row to see all possible acces to one node
        esum=0
    sorted_vec2 = sorted(vec2.items(), key=operator.itemgetter(1))
    sorted_vec2.reverse()  ##sort vec2 in ascending order
    vec2=sorted_vec2
    return vec1,vec2

def similarity(t):
    s=0
    similar=np.zeros((len(t),len(t)),dtype='float')
    for i in range(len(t)):
        for j in range(len(t)):
            s=SequenceMatcher(lambda x: x==" ",t[i],t[j])  ##determine the similarity between a tweet vs all the others excluding spaces
            similar[j,i]=s.ratio()
            similar[j,i]=round(similar[j,i],4)  ##create a percent ratio rounded to two decimal places
    similar*=100
    print('Numbers represent percent similarity between tweets!')
    print('Over 60 is considered very similar!')
    return similar
       


"""
Answers:
    Challenge 1
IN: a=[1,6,2,9]
    b=[8,2,7,1]
    corrcoeff(a,b)

OUT: -0.40292653175554555

    Challenge 2:
IN:  degree(zn)
OUT: ([17, 10, 11, 7, 4, 5, 5, 5, 6, 3, 4, 2, 3, 6, 3, 3, 3, 3, 3, 4, 3, 3, 3, 6, 4, 4, 3, 5, 4, 5, 5, 7, 13, 18],
 [33, 0, 32, 2, 1, 3, 31, 8, 13, 23, 5, 6, 7, 27, 29, 30, 4, 10, 19, 24, 25, 28, 9, 12, 14, 15, 16, 17, 18, 20, 21, 22, 26, 11])

IN: strength(zn)
OUT: ([17, 10, 11, 7, 4, 5, 5, 5, 6, 3, 4, 2, 3, 6, 3, 3, 3, 3, 3, 4, 3, 3, 3, 6, 4, 4, 3, 5, 4, 5, 5, 7, 13, 18],
 [33, 0, 32, 2, 1, 3, 31, 8, 13, 23, 5, 6, 7, 27, 29, 30, 4, 10, 19, 24, 25, 28, 9, 12, 14, 15, 16, 17, 18, 20, 21, 22, 26, 11])

    Challenge 3:
IN:  pagerank(zn, 0.1)

OUT: ({0: 20.400000000000016,
     1: 13.39999999999999,
     2: 14.39999999999999,
     3: 10.39999999999999,
     4: 7.3999999999999932,
     5: 8.3999999999999932,
     6: 8.3999999999999915,
     7: 8.3999999999999897,
     8: 9.3999999999999932,
     9: 6.3999999999999968,
     10: 7.3999999999999932,
     11: 5.3999999999999968,
     12: 6.3999999999999932,
     13: 9.3999999999999897,
     14: 6.4000000000000004,
     15: 6.4000000000000004,
     16: 6.399999999999995,
     17: 6.3999999999999959,
     18: 6.4000000000000004,
     19: 7.3999999999999968,
     20: 6.4000000000000004,
     21: 6.3999999999999968,
     22: 6.4000000000000004,
     23: 9.3999999999999986,
     24: 7.3999999999999986,
     25: 7.3999999999999986,
     26: 6.4000000000000004,
     27: 8.3999999999999986,
     28: 7.4000000000000004,
     29: 8.3999999999999986,
     30: 8.3999999999999968,
     31: 10.399999999999999,
     32: 16.399999999999995,
     33: 21.400000000000002},
    [(14, (1.5512397743802551+0j)),
    (33, (1.3841409279294896+0j)),
    (32, (1.2830145529277914+0j)),
    (0, (1.222560381013116+0j)),
    (21, (1.175775880600946+0j)),
    (17, (1.1757758806009329+0j)),
    (2, (1.1285590101833844+0j)),
    (15, (1.1223504818327199+0j)),
    (31, (1.0740684800364617+0j)),
    (1, (1.0524451298629653+0j)),
    (4, (1.0375444885108169+0j)),
    (10, (1.0375444885108027+0j)),
    (5, (1.0365630513281407+0j)),
    (6, (1.0365630513281237+0j)),
    (23, (1.028211232600438+0j)),
    (7, (1.0120999391450083+0j)),
    (27, (0.98834901923138363+0j)),
    (18, (0.98827833546497224+0j)),
    (8, (0.97658801623178071+0j)),
    (28, (0.94599135876615825+0j)),
    (13, (0.94344551762289564+0j)),
    (19, (0.92138483032627316+0j)),
    (30, (0.90652302832016707+0j)),
    (25, (0.90051527965013556+0j)),
    (20, (0.89669660634986381+0j)),
    (3, (0.85250038731873767+0j)),
    (29, (0.82886929144253119+0j)),
    (12, (0.81451715701870708+0j)),
    (9, (0.80860886351127759+0j)),
    (26, (0.80761751866875142+0j)),
    (24, (0.79622604554900378+0j)),
    (16, (0.67348929554475601+0j)),
    (11, (0.63985094800476905+0j)),
    (22, (0.58725825380193364+0j))])
 
    challenge 5:
IN: L=['dog dog', 'dog cat', 'cat cat']
    similarity(L)
    
OUT: 'Numbers represent percent similarity between tweets!
      Over 60 is considered very similar!'
      array([[ 100.  ,   57.14,    0.  ],
             [  57.14,  100.  ,   42.86],
             [   0.  ,   42.86,  100.  ]])
    
    """
   
