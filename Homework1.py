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
    
