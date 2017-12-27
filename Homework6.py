
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
