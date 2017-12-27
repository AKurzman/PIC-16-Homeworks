# PIC-16-Homeworks
Compilation of all python homework assignments completed during the duration of the course 

##HOMEWORK 1

import math
import random as rand

def largerindex(l):
    k = [0] *len(l)    #create list of 0s with length of l
    for i in range(0, len(l)):
        """determine if each entry is less than, equal to, or greater than its position
        and assign the correct values accordingly"""
        if l[i] > i:   
            k[i] = 1
        if l[i] == i:
            k[i] = 0
        if l[i] < i:
            k[i] = -1
            
    return k

def squaresupto(n):
    """sets x equal to the int of the squareroot of given n than appends
    list l accordingly with squares up to x"""
    x = int(math.sqrt(n))
    l = [i**2 for i in range(0, x + 1)]
    return l

def weekday(M,D,Y):  ##Finds day of week any day in history falls on
    days = {0: 'Sunday', 1:'Monday', 2:'Tuesday', 3:'Wednesday', 4:'Thursday', 5:'Friday', 6:'Saturday'}
    offset = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    leapoffset = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    cent1 = range(1800,1900)
    cent2 = range(1900,2000)
    cent3 = range(2000,2100)
    cent4 = range(2100,2200)
    anchorcents = range(1800,2200)
    anchors = {cent1: 5, cent2: 3, cent3: 2, cent4: 0} #anchor days for corresponding centuries
    while Y not in anchorcents: #get year input in to corresponding anchor centuries
        if Y > 2199:
            Y -= 400
        else:
            Y += 400
    hold = Y
    """determine whice century the Y would fall in to, then follow the 5 step process
    to find the doomsday of the week for corresponding year"""
    if Y in cent1:
        hold %= 1800
        incent = 1
    if Y in cent2:
        hold %= 1900
        incent = 2
    if Y in cent3:
        hold %= 2000
        incent = 3
    if Y in cent4:
        hold %= 2100
        incent = 4
    step1 = hold // 12
    step2 = hold - (step1 * 12)
    step3 = step2 // 4
    if incent == 1:
        step4 = anchors[cent1]
    if incent == 2:
        step4 = anchors[cent2]
    if incent == 3:
        step4 = anchors[cent3]
    if incent == 4:
        step4 = anchors[cent4]
    step5 = step4 + step3 + step2 +step1
    step5 %=7
    """once the doomsday is known, we can find the difference between the doomsday and the given
    day of the year by summing the month +days (offset lists at top) and returning the result"""
    if Y%4 != 0:
        totaldays = offset[M-1] + D
        temp = totaldays
        if temp > 157:
            while totaldays > 157:
                totaldays -= 7
        else:
            while totaldays < 157:
                totaldays +=7
        difference = 157 - totaldays
    else:
        totaldays = leapoffset[M-1] + D
        temp = totaldays
        if temp > 158:
            while totaldays > 158:
                totaldays -= 7
        else:
            while totaldays < 158:
                totaldays +=7
        difference = 158 - totaldays
    if (step5 - difference) < 0:
        return days[((step5 - difference) +7) % 7]
    else:
        return days[(step5 - difference) % 7]
    
def longestpath(d):
    """in order to prevent infinite loops, I created a shallow copy for each iteration
    that then deletes the previous entry.  It will then count every step and return the
    greatest amount"""
    count = 0
    k = 0
    longest = 0
    for k in range(0, len(d)):
        copy = d.copy()
        while k in copy:
            hold = k
            k = d[k]
            count +=1
            del copy[hold]
        if count > longest:
            longest = count
        count = 0
    return longest

def flip1in3():
    def biasedcoin(): #function of biased coin that returns 1 or 0 (1/3 of the time 1)
        coin1 = 0
        coin2 = 0
        coin3 = 0
        a = rand.random()
        b = rand.random()
        if a > 0.5:
            coin1 = 1
        if b > 0.5:
            coin2 = 1
        cointotal = coin1+ coin2
        if (cointotal == 2):  #if both flips are "heads", return heads
            coin3 = 1
        if (cointotal == 0): # if both came up tails, runs function again
            coin3 = biasedcoin()
        return coin3 #returns 0 or "tails" for double coin flip coming up HT or TH
    sumofcoinflips = []
    for i in range(0, 10000): #runs it 10000 times and counts all 1s to show 1/3 probability
        sumofcoinflips.append(biasedcoin())
    return (sumofcoinflips.count(1) / 10000)
    
## HOMEWORK 2 -> Image manipulation

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import math

##For a salt/pepper noise addition
def salt_pepper(im,ps=.1,pp=.1):
    im1=mpimg.imread(im)
    imcopy=im1
    n,m,z=im1.shape
    for i in range(n):
        for j in range(m):
            b=np.random.uniform()
            if b<ps:
                imcopy[i,j]=[0,0,0]
            elif b>1-pp:
                imcopy[i,j]=[1,1,1]
    for i in range(n):
        for j in range(m):
            if imcopy[i,j]==[0,0,0]:
                im1[i,j]==[0,0,0]
            elif imcopy[i,j]==[1,1,1]:
                im1[i,j]==[1,1,1]
    plt.imshow(im1)
    return im1

##next two functions convert any image to fixed grayscale to make pixels easier to manipulate by
##all that follows

def fixdtype(im):
    if im.dtype == np.uint8:
        im=im.astype(np.float32)/255.0
        return im
    elif im.dtype==np.float32 or im.dtype==np.float64:
        return im


def convertgray(im):
    if len(im.shape)==2:
        return im
    elif len(im.shape)==3 and (im.shape[2]==3 or im.shape[2]==4):
        im=im[:,:,(0,1,2)].sum(2)/3.0
        return im


##Blur to remove noise, such as salt pepper above
def blurring(im, k, btype, sigma):
    if k%2==0:
        print('please enter an odd number for \'k\'')
        return
    img = mpimg.imread(im)
    xpixel, ypixel, z = img.shape
    img = fixdtype(img)
    img = convertgray(img)
    img2 = img
    if btype == 'uniform' or btype == 'Uniform':
        def avgcolor(i,j, img):
            avg = 0.0
            for z in range(0,k):
                for x in range(0,k):
                    color = img[(i-int(k/2)) + z,(j-int(k/2)) + x]
                    avg = avg + color
            color = img[i,j]
            avg = (avg - color)/((k*k)-1)
            return avg

        for i in range(int(k/2), int(xpixel)-int(k/2)):
            for j in range(int(k/2), int(ypixel)-int(k/2)):
                img2[i,j] = avgcolor(i,j, img)
        plt.title('blurring uniform')
        plt.imshow(img2, cmap="gray", norm=plt.Normalize(vmin=0.0, vmax=1.0))
        return img2
    
    elif btype == 'gaussian' or btype == 'Gaussian':
        filterg=np.zeros((k,k), dtype = 'float32')
        gsum=0
        for i in range(0,k):
            for j in range(0,k):
                filterg[i,j] = (1/(2*math.pi*sigma**2))*(math.e**-(((i-int(k/2))**2)+((j-int(k/2))**2))/(2*sigma**2))
                gsum += filterg[i,j]
        filterg /= gsum
            
                    
        for i in range(int(k/2), int(xpixel)-int(k/2)):
            for j in range(int(k/2), int(ypixel)-int(k/2)):
                img2[i,j] = np.sum(np.sum(img[i-int(k/2):i+int(k/2)+1, j-int(k/2):j+int(k/2)+1]*filterg)) 
        plt.title('blurring gaussian')
        plt.imshow(img2, cmap="gray", norm=plt.Normalize(vmin=0.0, vmax=1.0))
        return img2

##To detect any edges in an image and make them white while everything else goes black

def detect_edge(img,method):
    img=mpimg.imread(img)
    xpixel, ypixel, z = img.shape
    img = fixdtype(img)
    img = convertgray(img)
    img2 = img
    sh=0.
    sv=0.
    horfilter = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype = 'float32')
    vertfilter = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype= 'float32')
    hold=np.zeros((3,3), dtype='float32')
    
    if method=='horizontal' or method=='Horizontal':
        tempim=np.zeros((xpixel,ypixel), dtype='float32')
        for i in range(1,xpixel-1):
            for j in range(1,ypixel-1):
                hold=np.array(img2[i-1:i+2,j-1:j+2])
                hold*=horfilter
                sh = np.sum(hold)
                sh= (sh+4)/8
                tempim[i,j]=sh
        plt.title('horizontal edge detection')
        plt.imshow(tempim, cmap="gray", norm=plt.Normalize(vmin=0.0, vmax=1.0))
        return tempim
        
    elif method=='vertical' or method=='Vertical':
        tempim=np.zeros((xpixel,ypixel), dtype='float32')
        for i in range(1,xpixel-1):
            for j in range(1,ypixel-1):
                hold=np.array(img2[i-1:i+2,j-1:j+2])
                hold*=vertfilter
                sv = np.sum(hold)
                sv= (sv+4)/8
                tempim[i,j]=sv
        plt.title('vertical edge detection')
        plt.imshow(tempim, cmap="gray", norm=plt.Normalize(vmin=0.0, vmax=1.0))
        return tempim
        
    elif method=='both' or method=='Both':
        tempim=np.zeros((xpixel,ypixel), dtype='float32')
        hold2=np.zeros((3,3), dtype='float32')
        for i in range(1,xpixel-1):
            for j in range(1,ypixel-1):
                hold=np.array(img2[i-1:i+2,j-1:j+2])
                hold2=np.array(img2[i-1:i+2,j-1:j+2])
                hold*=vertfilter
                hold2*=horfilter
                sv = np.sum(hold)
                sh = np.sum(hold2)
                sb = (((sv+4)/8)**2+((sh+4)/8)**2)**(1/2)
                tempim[i,j]=sb
        plt.title('combination edge detection')
        plt.imshow(tempim, cmap="gray", norm=plt.Normalize(vmin=0.0, vmax=1.0))
        return tempim
        
    else:
        print('please enter a valid edge-detection method')
        return

## Creates an image threshold using otsus algorithm

def otsu_threshold(img):
    img2=mpimg.imread(img)
    xpixel, ypixel, z = img2.shape
    img2 = fixdtype(img2)
    img2 = convertgray(img2)
    varf=0.
    thresh=0.
    imgcopy=img2.reshape((1, xpixel*ypixel))[0]
    bag=np.sort(imgcopy)
    mean=np.mean(bag)
    for t in [float(j) / 500 for j in range(0, 500, 1)]:
        ind = np.argmax(bag > t)
        var=((ind/(xpixel*ypixel))*(np.mean(bag[:ind])-mean)**2)+(((xpixel*ypixel-ind)/(xpixel*ypixel))*(np.mean(bag[ind:])-mean)**2)
        if var > varf:
            varf=var
            thresh=t
    mask=np.zeros((xpixel,ypixel), dtype='bool')
    for i in range(0,xpixel):
        for j in range(0,ypixel):
            if img2[i,j] <= thresh:
                mask[i,j]=True
            else:
                mask[i,j]=False
    plt.title('otsu threshold')
    plt.imshow(mask, cmap="gray", norm=plt.Normalize(vmin=0.0, vmax=1.0))
    return mask

## Blurs only the background of an image

def blur_background(im):
    img=mpimg.imread(im)
    xpixel, ypixel, z = img.shape
    img = fixdtype(img)
    img = convertgray(img)
    imcopy=img
    imblurred = blurring(im,5,"uniform",0.6)
    fgbg=otsu_threshold(im)
    for i in range(0,xpixel):
        for j in range(0,ypixel):
            if fgbg[i,j] == 0:
                imcopy[i,j]=imblurred[i,j]
    plt.title('blur background')
    plt.imshow(imcopy, cmap="gray", norm=plt.Normalize(vmin=0.0, vmax=1.0))
    return imcopy
    
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
    
# HOMEWORK 4

import sympy as sp

v0=[0,1]
w0=[1,0]
u0=[1,1]

def all_satisfiable(s):
    df=[]
    var=list(s.atoms())  ##puts all variables in to a list
    total_var=len(var)  ##gets the total number of variables
    if total_var>3:      ##keeps the total number below three so the function still works properly
        print('Please enter a function with 3 or less variables')
        return
    d={}
    if total_var==3:
        for i in range(2):
            for j in range(2):   ##for three variables, three nested loops so every T/F is tested
                for k in range(2):
                    if s.subs({var[0]:i,var[1]:j,var[2]:k})!=False:   ##if the vars subbed in is not False
                        d={var[0]:bool(i),var[1]:bool(j),var[2]:bool(k)}  ##i.e. satisfiable, append df with dict entry
                        df.append(d)
    if total_var==2:   ##repeat process but for two variables
        for i in range(2):
            for j in range(2):
                if s.subs({var[0]:i,var[1]:j})!=False:
                    d={var[0]:bool(i), var[1]:bool(j)}
                    df.append(d)
    if total_var==1:   ##repeat process for one variable
        if s.subs({var[0]:0})!=0:
            d={var[0]:False}
            df.append(d)
        if s.subs({var[0]:1})!=0:
            d={var[0]:True}
            df.append(d)
    return df




def drange(start, stop, step):  ##create my own for loop so I can increment by non integers
    numelements = int((stop-start)/float(step))
    for i in range(numelements+1):
            yield start + i*step
            
def normalcurve(a,b):
    x=sp.symbols('x')
    e=2.71828
    pi=3.14159
    phi=(e**((-1/2)*x**2))/((2*pi)**(1/2))   ##normal curve function
    tsum=0
    for i in drange(a,b,0.01):  ##create a Left Riemmans sum to simulate integration with 0.01 as width
        xleft=i
        tsum+=0.01*phi.subs({x:xleft})
    xright=b   ##add final amount because for loop stops one short
    tsum+=0.01*phi.subs({x:xright})
    return tsum
        
        

##Calculate total time first for y-velocity to reach zero then time it takes to fall back down to ground
##Then plug in total time for both legs in to x-velocity equation to get point    

def ball(v0):
    t=0
    d2=sp.symbols('d2')
    tf=((2*d2)/9.81)**(1/2)
    t+=v0[1]/9.81
    d1=v0[1]*t-0.5*9.81*t**2
    t2=tf.subs({d2:d1})
    t+=t2
    xland=v0[0]*t
    point= [xland, 0]
    return point

##HOMEWORK 5 -> Sorting Algorithms

import numpy as np
N=np.array[[0,2,1,0],[2,0,1,2],[1,1,0,1],[0,1,2,0]]

def select_sort(n):
    for i in range(len(n)):
        minindex=i
        for j in range(i+1,len(n)):
            if n[j]<n[minindex]:
                minindex=j
        n[minindex],n[i] = n[i],n[minindex]
    return n


def bubble_sort(n):
    oneswap=True
    while oneswap:
        oneswap=False
        for i in range(len(n)-1):
            if n[i]>n[i+1]:
                n[i+1], n[i]=n[i],n[i+1]
                oneswap=True
    return n




def merge_sort(n):
    if len(n)<=1:
        return n
    else:
        middle=int(len(n)/2)
        final=[]
        first=n[:middle]
        last=n[middle:]
        sn1=merge_sort(first)
        sn2=merge_sort(last)
        while len(sn1)>0 and len(sn2)>0:
            if sn1[0]<sn2[0]:
                final.append(sn1.pop(0))
            else:
                final.append(sn2.pop(0))
        while len(sn1)>0:
            final.append(sn1.pop(0))
        while len(sn2)>0:
            final.append(sn2.pop(0))
        return final
    
    

def partition(n,low,high):
    p=low
    pivot=n[low]
    while (low<=high) and (n[low]<=pivot):
        low+=1
    while (n[high]>pivot):
        high-=1
    if (low<high):
        n[high],n[low]=n[low],n[high]
    while (low<high):
        while (low<=high) and (n[low]<=pivot):
            low+=1
        while (n[high]>pivot):
            high-=1
        if (low<high):
            n[high],n[low]=n[low],n[high]
    n[high],n[p]=n[p],n[high]
    p = high
    return(p) 

        
def quick_sort(n, first, last):
    if (last-first)>=1:
        pivotindex=partition(n,first,last)
        quick_sort(n,first,pivotindex-1)
        quick_sort(n,pivotindex+1,last)
    return n



def shortest(n,v,w):
    verts=[]
    infinite=max(max(n))+1
    for i in range(len(n)):
        verts.append(infinite)
    verts[v]=0
    while len(verts)>0:
        burnin=verts.index(min(verts))
        burnval=verts[burnin]
        verts.pop(burnin)
        for i in range(len(verts)):
            if n[i][burnin]!=0:
                weight=n[i][burnin]
                if verts[i]>(burnval+weight):
                    verts[i]=(burnval+weight)
            print(burnin)

        if burnin==w:
            return burnval
            
## HOMEWORK 6

from tkinter import *
import numpy as np
import sympy as sp
from sympy import *
from sympy.plotting import plot

class maxlist(object): ##Creates a max list class with add and delete functions 
    def __init__(self,val):
        self.val=val
    
    def delmax(self):
        maxl=0
        for i in range(len(self.val)):
            if self.val[i]>maxl:
                maxl=self.val[i]
        self.val=[x for x in self.val if x!=maxl]
        return self.val
    
    def addmax(self):
        maxl=0
        for i in range(len(self.val)):
            if self.val[i]>maxl:
                maxl=self.val[i]
        add = maxl+1
        self.val.append(add)
        return self.val
    
    
## Creates an interactive gui of knights tour playable through tkinter
def knights_tour(n):
    global currentx,currenty
    currentx,currenty=1,1
    tracker=np.zeros((n,n))
    def box_coord(x1,y1):
        a1=float(x1/(500/(n)))
        b1=float(y1/(500/(n)))
        for i in range(n):
            if a1>i and a1<i+1:
                xbox=i
            if b1>i and b1<i+1:
                ybox=i
        return xbox,ybox
    def is_valid(x1,y1,x2,y2):
        move=False
        xbox1,ybox1=box_coord(x1,y1)
        xbox2,ybox2=box_coord(x2,y2)
        if xbox2-xbox1==2 or xbox2-xbox1==-2:
            if ybox2-ybox1==1 or ybox2-ybox1==-1:
                move=True
        if xbox2-xbox1==1 or xbox2-xbox1==-1:
            if ybox2-ybox1==2 or ybox2-ybox1==-2:
                move=True
        return move
    master = Tk()
    w = Canvas(master, width=500, height=500)
    w.pack()
    w.create_rectangle(0,0,500,500)
    k=500/(n)
    for i in range(0,n-1):
        w.create_line(0,k,500,k)
        k+=500/(n)
    l=500/(n)
    for j in range(0,n-1):
        w.create_line(l,0,l,500)
        l+=500/(n)
    m=500/(n)
    w.create_rectangle(0,0,m,m,fill='red')
    def click(event):
        global currentx,currenty
        if is_valid(currentx,currenty,event.x,event.y):
            xprev,yprev=box_coord(currentx,currenty)
            w.create_rectangle((xprev*m),(yprev*m),(xprev+1)*m,(yprev+1)*m, fill='gray')
            x,y=box_coord(event.x,event.y)
            if tracker[x][y]==0:
                w.create_rectangle((x*m),(y*m),(x+1)*m,(y+1)*m, fill='red')
                currentx,currenty=event.x,event.y
                tracker[x][y]=1
    w.bind("<Button-1>", click)
    mainloop()

def lambdafunction(grades):  ##Calculates a students grade using a lambda function
    hw=list(map(lambda x: 2*x[1],grades))
    final=list(map(lambda x: .8*x[2],grades))
    letter=list(map(lambda x,y: x+y,hw,final))
    finallist=[]
    for i in range(len(letter)):
        if letter[i]<40:
            finallist.append(grades[i][0])
            finallist.append('D')
        if letter[i]<60:
            finallist.append(grades[i][0])
            finallist.append('C')
        if letter[i]<80:
            finallist.append(grades[i][0])
            finallist.append('B')
        else:
            finallist.append(grades[i][0])
            finallist.append('A')
            
    return finallist
    

    
    
    
    
    
    
    
    

     
    
    
        
    
    
  
    
