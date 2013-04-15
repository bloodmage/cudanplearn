__doc__="""
CUDANPLEARN Python utility
Copyright (C) 2013 Zhouyisu <zhouyisu # gmail.com>

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""

import numpy
import cudanplearn.learn as _l
import os
_WHITE_COE=numpy.array([
 [ -0.004695638 , 0.008173633875 , -0.0086846625 , 0.00142440925 , -0.0086846625 , 0.008173633875 , -0.004695638 ,] ,
 [ 0.008173633875 , -0.01488577 , 0.01585275 , -0.005398815 , 0.01585275 , -0.01488577 , 0.008173633875 ,] ,
 [ -0.0086846625 , 0.01585275 , 0.01774525 , -0.091637825 , 0.01774525 , 0.01585275 , -0.0086846625 ,] ,
 [ 0.00142440925 , -0.005398815 , -0.091637825 , 0.273005 , -0.091637825 , -0.005398815 , 0.00142440925 ,] ,
 [ -0.0086846625 , 0.01585275 , 0.01774525 , -0.091637825 , 0.01774525 , 0.01585275 , -0.0086846625 ,] ,
 [ 0.008173633875 , -0.01488577 , 0.01585275 , -0.005398815 , 0.01585275 , -0.01488577 , 0.008173633875 ,] ,
 [ -0.004695638 , 0.008173633875 , -0.0086846625 , 0.00142440925 , -0.0086846625 , 0.008173633875 , -0.004695638 ,] ,
# [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
],'f').reshape((1,7,7,1))

_WHITE_CIELAB_COE=numpy.array([[[
        [0.0062203,0.0033909,-0.000312759],
        [-0.0871005,-0.0227324,-0.00785892],
        [0.0062203,0.0033909,-0.000312759]],
       [[-0.0871005,-0.0227324,-0.00785892],
        [0.330848,0.0777191,0.0351395],
        [-0.0871005,-0.0227324,-0.00785892]],
       [[0.0062203,0.0033909,-0.000312759],
        [-0.0871005,-0.0227324,-0.00785892],
        [0.0062203,0.0033909,-0.000312759]]],
      [[[0.00339405,0.0139338,0.00207464],
        [-0.0227371,-0.235708,-0.0150918],
        [0.00339405,0.0139338,0.00207464]],
       [[-0.0227371,-0.235708,-0.0150918],
        [0.0777191,0.946852,0.0536429],
        [-0.0227371,-0.235708,-0.0150918]],
       [[0.00339405,0.0139338,0.00207464],
        [-0.0227371,-0.235708,-0.0150918],
        [0.00339405,0.0139338,0.00207464]]],
      [[[-0.000309899,0.00207109,0.00355706],
        [-0.00785814,-0.0150962,-0.0568861],
        [-0.000309899,0.00207109,0.00355706]],
       [[-0.00785814,-0.0150962,-0.0568861],
        [0.0351395,0.0536429,0.23047],
        [-0.00785814,-0.0150962,-0.0568861]],
       [[-0.000309899,0.00207109,0.00355706],
        [-0.00785814,-0.0150962,-0.0568861],
        [-0.000309899,0.00207109,0.00355706]]]],'f')

def saveArrayToFile(data, fileName, tmpbak=True):
    if tmpbak==False:
        with open(fileName, 'wb') as file:
            file.write(data.tostring())
        return
    saveArrayToFile(data,'%s.tmp'%fileName,tmpbak=False)
    try: os.mkdir('%s.tmpdone'%fileName)
    except OSError: pass
    try: os.remove(fileName)
    except OSError: pass
    os.rename('%s.tmp'%fileName,fileName)
    os.rmdir('%s.tmpdone'%fileName)

def readArrayFromFile(fileName, shape, count=-1, tmpbak=True):
    if tmpbak==False:
        _featDesc = numpy.fromfile(fileName, 'f', count)
        _featDesc = _featDesc.reshape(shape)
        return _featDesc
    if os.path.exists('%s.tmpdone'%fileName):
        if os.path.exists('%s.tmp'%fileName): #Use tmp first
            return readArrayFromFile('%s.tmp'%fileName,shape,count,False)
    return readArrayFromFile(fileName,shape,count,False) #Fallback

def readArrayFromFileOrFunc(filename, shape, lambda_, tmpbak=True):
    if tmpbak==False:
        if os.path.exists(filename):
            return readArrayFromFile(filename, shape)
        else:
            return lambda_()
    if os.path.exists('%s.tmpdone'%filename):
        if os.path.exists('%s.tmp'%filename): #Use tmp first
            return readArrayFromFileOrFunc('%s.tmp'%filename,shape,lambda_,False)
    return readArrayFromFileOrFunc(filename,shape,lambda_,False) #Fallback


def saveNumberToFile(data, fileName):
    with open(fileName, 'w') as file:
        file.write("%s"%data)

def readNumberFromFile(fileName, default=0.0):
    if os.path.exists(fileName):
        return float(file(fileName,'r').read())
    else:
        return default

def whiteninggray(images,dc=True):
    """
    This function automatically whites gray images (Image only, data range 0-255, precalculated filter).
    Images is a 3d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, YSize, XSize
    Return 3d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, YSize-6, XSize-6
    """
    global _WHITE_COE
    _l.ASSERTTYPE(images)
    if len(images.shape)!=3: raise Exception, "3d-ndarray needed."

    #Remove DC Coefficient
    if dc:
        if images.shape[0]==1:
            tmp=images
        else:
            tmp=_l.empty(images.shape)
            _l.removeDC(images.reshape(images.shape[0],images.shape[1]*images.shape[2]),tmp.reshape(images.shape[0],images.shape[1]*images.shape[2]))
    else:
        tmp=images

    #Convolution to get result
    out=_l.empty((images.shape[0],1,images.shape[1]-6,images.shape[2]-6))
    _l.convolution4D(tmp.reshape(images.shape[0],images.shape[1],images.shape[2],1),_WHITE_COE,out)
    return out.reshape(images.shape[0],images.shape[1]-6,images.shape[2]-6)

def whiteningcielab(image):
    """
    This function does whitening on a CIEL*A*B* colorspace color image.
    It uses predefined filter to do convolution on image.
    
    Image is a 3d-ndarray with dtype=float32, with dimension ordering:
        YSize, XSize, [L,A,B]
    Returns 3d-ndarray with dtype=float32, with dimension ordering:
        YSize-2, XSize-2, [w_L,w_A,w_B]
    """
    global _WHITE_CIELAB_COE
    _l.ASSERTTYPE(image)
    out=_l.empty((1,3,image.shape[0]-2,image.shape[1]-2))
    ireshape=image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
    _l.convolution4D(ireshape,_WHITE_CIELAB_COE,out)
    return numpy.array(out.reshape((3,image.shape[0]-2,image.shape[1]-2)).transpose((1,2,0)),'f')

_xyz=numpy.array([[0.412453,0.212671,0.019334],[0.357580,0.715160,0.119193],[0.180423,0.072169,0.950227]],'f')

_lfunc='[0.33333] a pow [116] mul [-16] add [903.3] a mul a [0.008856] >= ?:'
_ffunc='[0.33333] a pow [7.787] a mul [0.137931] add a [0.008856] >= ?:'

def rgb2cielab(image):
    """
    This function translates a RGB image (0~255 for each pixel) to a CIEL*A*B* image (0~100 for L*, -127~127 for A*, -127~127 for B*)

    Image is a 3d-ndarray with dtype=uint8, with dimension ordering:
        YSize, XSize, [R,G,B]
    Returns 3d-ndarray with dtype=float32, with dimension ordering:
        YSize, XSize, [L,A,B]
    """
    global _xyz,_lfunc,_ffunc
    ishape=image.shape
    dmat=numpy.array(image.reshape((-1,3)),'f')
    dmat/=255
    dmat=dmat.dot(_xyz)

    x=numpy.array(dmat[:,0],'f')
    y=numpy.array(dmat[:,1],'f')
    z=numpy.array(dmat[:,2],'f')
    x/=0.950456
    z/=1.088754

    l=_l.empty((dmat.shape[0],))
    _l.transform(_lfunc,l,y)
    fx=_l.empty((dmat.shape[0],))
    fy=_l.empty((dmat.shape[0],))
    fz=_l.empty((dmat.shape[0],))
    _l.transform(_ffunc,fx,x)
    _l.transform(_ffunc,fy,y)
    _l.transform(_ffunc,fz,z)

    a=500*(fx-fy)
    b=500*(fy-fz)

    dmat[:,0]=l
    dmat[:,1]=a
    dmat[:,2]=b

    return dmat.reshape(ishape)

if __name__=="__main__":
    import Image
    data=Image.open('test.jpg')
    dbw=data.convert('L')
    print dbw.size

    npa=numpy.array(list(dbw.getdata()),'f').reshape(1,dbw.size[1],dbw.size[0])
    npw=whiteninggray(npa)
    print npw.shape
    saveArrayToFile(npw, 'test.out')
