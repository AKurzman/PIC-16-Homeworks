
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
    

