
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
            
