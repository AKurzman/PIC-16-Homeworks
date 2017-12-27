
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
