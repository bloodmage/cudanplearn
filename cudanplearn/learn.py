__doc__="""
CUDANPLEARN Main learn functions wrapper
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

import platform as _p, sys as _sys
import ctypes as ct
import numpy as np
import threading
import time
if _p.system() == 'Windows':
    _dll = ct.cdll.LoadLibrary('cudanplearn.dll')
else:
    _dll = ct.cdll.LoadLibrary('cudanplearn.so')

_dll.get_ErrorString.restype=ct.c_char_p

def CASSERT(v):
    if v==0: return
    import traceback
    traceback.print_stack()
    if v==-1: print "CUDA_ERROR",str(_dll.get_ErrorString())
    elif v==-2: print "INSUFFICIENT_DEVICE_MEMORY"
    elif v==-3: print "INVALID_ARGUMENT"
    _sys.exit(-1)

def ASSERTTYPEG(v):
    if str(v.__class__)!="<class 'gnumpy.garray'>":
        raise Exception, "Argument type should be gnumpy garray"
    if v._base.mat.is_trans!=0:
        raise Exception, "Array must not be transposed"

def ASSERTTYPED(v):
    if v.dtype.name!='float64':
        raise Exception, "Argument type should be numpy ndarray with dtype float64/'f8'"
    #Check order of dimensions
    lstride=None
    for i in range(len(v.ctypes.strides)):
        if lstride==None:
            lstride=v.ctypes.strides[i]
        else:
            if lstride/v.ctypes.strides[i]!=v.shape[i]:
                raise Exception, "stride not follow C order"
            lstride=v.ctypes.strides[i]
    if lstride!=8:
        raise Exception, "Last stride must be 8 (double)"

def ASSERTTYPE(v):
    if v.dtype.name!='float32':
        raise Exception, "Argument type should be numpy ndarray with dtype float32/'f'"
    #Check order of dimensions
    lstride=None
    for i in range(len(v.ctypes.strides)):
        if lstride==None:
            lstride=v.ctypes.strides[i]
        else:
            if lstride/v.ctypes.strides[i]!=v.shape[i]:
                raise Exception, "Stride not follow C order"
            lstride=v.ctypes.strides[i]
    if lstride!=4:
        raise Exception, "Last stride must be 4 (float)"

def _transform_kern(d0,op,d1,d2,d3,out):
    if d0!=None: d0=d0.ctypes.data_as(ct.POINTER(ct.c_float))
    if d1!=None: d1=d1.ctypes.data_as(ct.POINTER(ct.c_float))
    if d2!=None: d2=d2.ctypes.data_as(ct.POINTER(ct.c_float))
    if d3!=None: d3=d3.ctypes.data_as(ct.POINTER(ct.c_float))
    CASSERT(_dll.transform(d0,op,len(op),out.shape[0],out.ctypes.data_as(ct.POINTER(ct.c_float)),d1,d2,d3))

def _transformD_kern(d0,op,d1,d2,d3,out):
    if d0!=None: d0=d0.ctypes.data_as(ct.POINTER(ct.c_double))
    if d1!=None: d1=d1.ctypes.data_as(ct.POINTER(ct.c_double))
    if d2!=None: d2=d2.ctypes.data_as(ct.POINTER(ct.c_double))
    if d3!=None: d3=d3.ctypes.data_as(ct.POINTER(ct.c_double))
    CASSERT(_dll.transformD(d0,op,len(op),out.shape[0],out.ctypes.data_as(ct.POINTER(ct.c_double)),d1,d2,d3))

def _transform2_kern(d0,op,d1,d2,d3,out,out2):
    if d0!=None: d0=d0.ctypes.data_as(ct.POINTER(ct.c_float))
    if d1!=None: d1=d1.ctypes.data_as(ct.POINTER(ct.c_float))
    if d2!=None: d2=d2.ctypes.data_as(ct.POINTER(ct.c_float))
    if d3!=None: d3=d3.ctypes.data_as(ct.POINTER(ct.c_float))
    CASSERT(_dll.transform2(d0,op,len(op),out.shape[0],out.ctypes.data_as(ct.POINTER(ct.c_float)),out2.ctypes.data_as(ct.POINTER(ct.c_float)),d1,d2,d3))

def _transform_kern_gnumpy(d0,op,d1,d2,d3,out):
    if d0!=None: d0=d0._base.mat.data_device
    if d1!=None: d1=d1._base.mat.data_device
    if d2!=None: d2=d2._base.mat.data_device
    if d3!=None: d3=d3._base.mat.data_device
    CASSERT(_dll.transformgpu(d0,op,len(op),out.shape[0],out._base.mat.data_device,d1,d2,d3))

def _transform2_kern_gnumpy(d0,op,d1,d2,d3,out,out2):
    if d0!=None: d0=d0._base.mat.data_device
    if d1!=None: d1=d1._base.mat.data_device
    if d2!=None: d2=d2._base.mat.data_device
    if d3!=None: d3=d3._base.mat.data_device
    CASSERT(_dll.transformgpu2(d0,op,len(op),out.shape[0],out._base.mat.data_device,out2._base.mat.data_device,d1,d2,d3))

def empty(shape):
    return np.empty(shape,'f')

def setMemory(newmemory):
    if not isinstance(newmemory,int): raise Exception, "New memory must be integer."
    if newmemory<=0: raise Exception, "Invalid memory value"
    _dll.set_CUDAMemory(newmemory)

def CAUTION_setAlign(newalign):
    "WARNING: change align will cause program to run very slowly."
    if not isinstance(newalign,int): raise Exception, "Alignment must be integer."
    if newalign<=0: raise Exception, "Invalid alignment"
    _dll.set_CUDAAligns(newalign)

def getCores():
    return _dll.get_CUDACores()

def setCore(coreid):
    if not isinstance(coreid,int): raise Exception, "coreid must be integer."
    if coreid<0 or coreid>getCores(): raise Exception, "Invalid core id (must in [0-%s))"%getCores()
    _dll.set_CUDACore(coreid)

multicore=None
def setMultiCore(coreids):
    "This function will make call to multiple CPU/GPU cores"
    global multicore
    l=[]
    maxid=getCores()
    #Core check
    for i in coreids:
        if not isinstance(i,int): raise Exception, "coreid must be integer."
        if i<0 or i>=maxid: raise Exception, "Invalid core id (must in [0-%s))"%maxid
        if i in l: raise Exception, "Duplicated coredid"
        l.append(i)
    if len(coreids)<1: raise Exception, "Needs at least one core"
    if len(coreids)==1:
        setCore(coreids[0])
        multicore=None
    else:
        #For calls cannot be splitted to multiple GPUs
        setCore(coreids[0])
        multicore=coreids

def multicall(*splitrule):
    def cmd(func):
        def wrappedfunc(*args):
            global multicore

            if multicore==None:
                return func(*args)

            #Get cores and collectors
            cores=len(multicore)
            valcollect=[None for i in range(cores)]
            exc=[False]
            def coreexec(myid):
                callarg=[]
                #Analysis rules
                for pi in range(len(splitrule)):
                    rule=splitrule[pi]

                    if isinstance(rule,list): #Optional args
                        if len(args)<=pi:
                            continue
                        rule=rule[0]
                    
                    if rule==None: #Plain rule
                        callarg.append(args[pi])
                    elif rule==0: #Split rule
                        splitadds=args[pi].shape[0]%cores
                        splitlen=args[pi].shape[0]/cores
                        if splitadds>myid:
                            splitlen+=1
                            splitstart=splitlen*myid
                        else:
                            splitstart=splitlen*myid+splitadds
                        callarg.append(args[pi][splitstart:splitstart+splitlen])
                    elif rule==-1: #Combine rule
                        v=empty(args[pi].shape)
                        valcollect[myid]=v
                        callarg.append(v)

                try:
                    #Run on each core
                    setCore(multicore[myid])
                    starttime=time.time()
                    func(*callarg)
                except:
                    import traceback
                    traceback.print_exc()
                    print "Core: %d, Kernel run time:%s"%(multicore[myid],time.time()-starttime)
                    exc[0]=True

            threads=[threading.Thread(target=coreexec,args=(i,)) for i in range(cores)]
            for i in threads: i.daemon=True
            for i in threads: i.start()
            for i in threads: i.join()
            if exc[0]: raise Exception,"Error happen"
            
            #If needs merge (add)
            if valcollect[0]!=None:
                for i in range(len(splitrule)):
                    if splitrule[i]==-1:
                        args[i][:]=reduce(np.add,valcollect)

        #Restore function definition in __doc__
        import inspect
        args=inspect.getargspec(func)
        defaults=list(args.defaults) if args.defaults is not None else []
        while len(defaults)<len(args.args):
            defaults.insert(0,args)
        arglist=', '.join([args.args[p]+'='+str(defaults[p]) if defaults[p] is not args else args.args[p] for p in range(len(defaults))])

        wrappedfunc.__doc__="%s(%s)\n"%(func.__name__,arglist)+func.__doc__+"\n(Multiple CUDA Core supported)"
        return wrappedfunc
    return cmd

@multicall(0,None,0)
def convkeep4D(data,filters,out):
    """
    [in] data is 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YSize, XSize
    [in] filters is a 3d-ndarray with dtype=float32, with dimension ordering:
        Layers, FilterY, FilterX
    [out] out is 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YSize, XSize

    This function applies a no-shrinking convolution on each layer.
    FilterX and FilterY must be odd number, The center of filter will move through each element of
    data in same layer to generate output.
    """
    ASSERTTYPE(data)
    ASSERTTYPE(filters)
    ASSERTTYPE(out)

    (ni,li,yi,xi)=data.shape
    (lf,yf,xf)=filters.shape
    (no,lo,yo,xo)=out.shape
    
    if ni!=no: raise Exception, "Data count mismatch"
    if li!=lf or li!=lo: raise Exception, "Layers mismatch"
    if (yf&1)!=1 or (xf&1)!=1: raise Exception, "Filter must be odd shape"
    if yi!=yo or xi!=xo: raise Exception, "Shape mismatch"

    data=data.ctypes.data_as(ct.POINTER(ct.c_float))
    filters=filters.ctypes.data_as(ct.POINTER(ct.c_float))
    out=out.ctypes.data_as(ct.POINTER(ct.c_float))

    CASSERT(_dll.convkeep4D(data,filters,ni,li,yi,xi,yf,xf,out))

@multicall(0,None,0)
def convolution4D(data,filters,out):
    """
    [in] data is 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, YSize, XSize, Layer_Of_Inputs
    [in] filters is a 4d-ndarray with dtype=float32, with dimension ordering:
        Layer_Of_Outputs, FilterY, FilterX, Layer_Of_Inputs
    [out] out is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layer_Of_Outputs, YOut(YSize-FilterY+1), XOut(XSize-FilterX+1)
    
    This function makes a convolution-like filter processing,
    It scans each filter in every spatial position of data, and record filtered result to out.
    """
    ASSERTTYPE(data)
    ASSERTTYPE(filters)
    ASSERTTYPE(out)
    
    (ni,yi,xi,li)=data.shape
    (lof,fy,fx,lif)=filters.shape
    (no,lo,yo,xo)=out.shape

    if ni!=no: raise Exception, "Data count mismatch"
    if li!=lif: raise Exception, "Layers of input mismatch"
    if lo!=lof: raise Exception, "Layers of output mismatch"
    if yi-fy+1!=yo: raise Exception, "Output Y size not match"
    if xi-fx+1!=xo: raise Exception, "Output X size not match"

    data=data.ctypes.data_as(ct.POINTER(ct.c_float))
    filters=filters.ctypes.data_as(ct.POINTER(ct.c_float))
    out=out.ctypes.data_as(ct.POINTER(ct.c_float))

    CASSERT(_dll.convolution4D(data,filters,ni,yi,xi,li,lof,fy,fx,out))

@multicall(0,0,-1)
def gradconvolution4D(data,grad,filtergrad):
    """
    [in] data is 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, YSize, XSize, Layer_Of_Inputs
    [in] grad is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layer_Of_Outputs, YOut, XOut
    [out] filtergrad is a 4d-ndarray with dtype=float32, with dimension ordering:
        Layer_Of_Outputs, FilterY(YSize-YOut+1), FilterX(XSize-XOut+1), Layer_Of_Inputs
    
    
    This function transfers grad of output layer into grad of filters,
    It stores the grad in filtergrad.
    """
    ASSERTTYPE(data)
    ASSERTTYPE(grad)
    ASSERTTYPE(filtergrad)

    (ni,yi,xi,li)=data.shape
    (no,lo,yo,xo)=grad.shape
    (lof,fy,fx,lif)=filtergrad.shape

    if ni!=no: raise Exception, "Data count mismatch"
    if li!=lif: raise Exception, "Layers of input mismatch"
    if lo!=lof: raise Exception, "Layers of output mismatch"
    if yi-fy+1!=yo: raise Exception, "Output Y size not match"
    if xi-fx+1!=xo: raise Exception, "Output X size not match"

    data=data.ctypes.data_as(ct.POINTER(ct.c_float))
    filtergrad=filtergrad.ctypes.data_as(ct.POINTER(ct.c_float))
    grad=grad.ctypes.data_as(ct.POINTER(ct.c_float))

    CASSERT(_dll.gradconvolution4D(data,grad,filtergrad,ni,yi,xi,li,lof,fy,fx))

@multicall(0,None,0)
def reverseconvolution4D_ro(grad,filters,datagrad):
    """
    [in] grad is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layer_Of_Outputs, YOut, XOut
    [in] filters is a 4d-ndarray with dtype=float32, with dimension ordering:
        Layer_Of_Outputs, FilterY, FilterX, Layer_Of_Inputs
    [out] datagrad is 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, YSize(YOut+FilterY-1), XSize(XOut+FilterX-1), Layer_Of_Inputs
    
    
    This function transfers grad of output layer into grad of data,
    It stores the grad in datagrad.
    Also, this function can be used to reconstruct data given conv result(as grad)
    """
    ASSERTTYPE(datagrad)
    ASSERTTYPE(grad)
    ASSERTTYPE(filters)

    (ni,yi,xi,li)=datagrad.shape
    (no,lo,yo,xo)=grad.shape
    (lof,fy,fx,lif)=filters.shape

    if ni!=no: raise Exception, "Data count mismatch"
    if li!=lif: raise Exception, "Layers of input mismatch"
    if lo!=lof: raise Exception, "Layers of output mismatch"
    if yi-fy+1!=yo: raise Exception, "Output Y size not match"
    if xi-fx+1!=xo: raise Exception, "Output X size not match"

    datagrad=datagrad.ctypes.data_as(ct.POINTER(ct.c_float))
    filters=filters.ctypes.data_as(ct.POINTER(ct.c_float))
    grad=grad.ctypes.data_as(ct.POINTER(ct.c_float))

    CASSERT(_dll.reverseconvolution4D(grad,filters,datagrad,ni,yi,xi,li,lof,fy,fx))

@multicall(0,None,0)
def reverseconvolution4D(grad,filters,datagrad):
    """
    [in] grad is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layer_Of_Outputs, YOut, XOut
    [in] filters is a 4d-ndarray with dtype=float32, with dimension ordering:
        Layer_Of_Outputs, FilterY, FilterX, Layer_Of_Inputs
    [out] datagrad is 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, YSize(YOut+FilterY-1), XSize(XOut+FilterX-1), Layer_Of_Inputs
    
    
    This function transfers grad of output layer into grad of data,
    It stores the grad in datagrad.
    Also, this function can be used to reconstruct data given conv result(as grad)
    """
    ASSERTTYPE(datagrad)
    ASSERTTYPE(grad)
    ASSERTTYPE(filters)

    (ni,yi,xi,li)=datagrad.shape
    (no,lo,yo,xo)=grad.shape
    (lof,fy,fx,lif)=filters.shape

    if ni!=no: raise Exception, "Data count mismatch"
    if li!=lif: raise Exception, "Layers of input mismatch"
    if lo!=lof: raise Exception, "Layers of output mismatch"
    if yi-fy+1!=yo: raise Exception, "Output Y size not match"
    if xi-fx+1!=xo: raise Exception, "Output X size not match"

    datagrad=datagrad.ctypes.data_as(ct.POINTER(ct.c_float))
    filters=filters.ctypes.data_as(ct.POINTER(ct.c_float))
    grad=grad.ctypes.data_as(ct.POINTER(ct.c_float))
    
    #if fx>=3 and fx<=28:
    #    CASSERT(_dll.reverseconvolution4D_smallfilter(grad,filters,datagrad,ni,yi,xi,li,lof,fy,fx))
    #else:
    CASSERT(_dll.reverseconvolution4D_outordered(grad,filters,datagrad,ni,yi,xi,li,lof,fy,fx))

@multicall(0,None,0)
def tohidden_noconv4D(data,weight,out):
    """
    [in] data is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, YSize, XSize, Layer_Of_Inputs
    [in] weight is a 6d-ndarray with dtype=float32, with dimension ordering:
        Layer_Of_Outputs, OutY, OutX, FilterY, FilterX, Layer_Of_Inputs
    [out] out is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layer_Of_Outputs, OutY, OutX

    To hidden layer (without convolution)
    Like convolution4D, but apply different filter on different places.
    Record filtered result in out.
    """
    ASSERTTYPE(data)
    ASSERTTYPE(weight)
    ASSERTTYPE(out)
    
    (ni,yi,xi,li)=data.shape
    (lof,oyf,oxf,fy,fx,lif)=weight.shape
    (no,lo,yo,xo)=out.shape

    if ni!=no: raise Exception, "Data count mismatch"
    if li!=lif: raise Exception, "Layers of input mismatch"
    if lo!=lof: raise Exception, "Layers of output mismatch"
    if yi-fy+1!=oyf: raise Exception, "Output Y size not match"
    if xi-fx+1!=oxf: raise Exception, "Output X size not match"
    if oyf!=yo or oxf!=xo: raise Exception, "Output not agree with Filter"

    data=data.ctypes.data_as(ct.POINTER(ct.c_float))
    weight=weight.ctypes.data_as(ct.POINTER(ct.c_float))
    out=out.ctypes.data_as(ct.POINTER(ct.c_float))
    
    CASSERT(_dll.tohidden_kern(data,weight,out,ni,yi,xi,li,lof,oyf,oxf,fy,fx))

@multicall(0,None,0)
def fromhidden_noconv4D(hidden,weight,dataout):
    """
    [in] hidden is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layer_Of_Outputs, OutY, OutX
    [in] weight is a 6d-ndarray with dtype=float32, with dimension ordering:
        Layer_Of_Outputs, OutY, OutX, FilterY, FilterX, Layer_Of_Inputs
    [out] dataout is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, YSize, XSize, Layer_Of_Inputs

    From hidden layer (without convolution)
    Like reverseconvolution4D, restore data from hidden, given weight,
    store result in dataout.
    Also, this function can be used to transfer grad from hidden to data layer, given weight.
    """
    ASSERTTYPE(hidden)
    ASSERTTYPE(weight)
    ASSERTTYPE(dataout)

    (ni,yi,xi,li)=dataout.shape
    (lof,oyf,oxf,fy,fx,lif)=weight.shape
    (no,lo,yo,xo)=hidden.shape

    if ni!=no: raise Exception, "Data count mismatch"
    if li!=lif: raise Exception, "Layers of input mismatch"
    if lo!=lof: raise Exception, "Layers of output mismatch"
    if yi-fy+1!=oyf: raise Exception, "Output Y size not match"
    if xi-fx+1!=oxf: raise Exception, "Output X size not match"
    if oyf!=yo or oxf!=xo: raise Exception, "Output not agree with Filter"

    dataout=dataout.ctypes.data_as(ct.POINTER(ct.c_float))
    weight=weight.ctypes.data_as(ct.POINTER(ct.c_float))
    hidden=hidden.ctypes.data_as(ct.POINTER(ct.c_float))
    
    CASSERT(_dll.fromhidden_kern(hidden,weight,dataout,ni,yi,xi,li,lof,oyf,oxf,fy,fx))

@multicall(0,0,-1)
def grad_noconv4D(data,gradhidden,gradweight):
    """
    [in] data is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, YSize, XSize, Layer_Of_Inputs
    [in] gradhidden is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layer_Of_Outputs, OutY, OutX
    [out] gradweight is a 6d-ndarray with dtype=float32, with dimension ordering:
        Layer_Of_Outputs, OutY, OutX, FilterY, FilterX, Layer_Of_Inputs

    Transfer grad from hidden layer to weights.
    """
    ASSERTTYPE(data)
    ASSERTTYPE(gradhidden)
    ASSERTTYPE(gradweight)

    (ni,yi,xi,li)=data.shape
    (lof,oyf,oxf,fy,fx,lif)=gradweight.shape
    (no,lo,yo,xo)=gradhidden.shape

    if ni!=no: raise Exception, "Data count mismatch"
    if li!=lif: raise Exception, "Layers of input mismatch"
    if lo!=lof: raise Exception, "Layers of output mismatch"
    if yi-fy+1!=oyf: raise Exception, "Output Y size not match"
    if xi-fx+1!=oxf: raise Exception, "Output X size not match"
    if oyf!=yo or oxf!=xo: raise Exception, "Output not agree with Filter"

    data=data.ctypes.data_as(ct.POINTER(ct.c_float))
    gradweight=gradweight.ctypes.data_as(ct.POINTER(ct.c_float))
    gradhidden=gradhidden.ctypes.data_as(ct.POINTER(ct.c_float))
    
    CASSERT(_dll.extractvalue_kern(data,gradhidden,gradweight,ni,yi,xi,li,lof,oyf,oxf,fy,fx))

@multicall(0,0)
def linearpoolin(hiddenin,poolingout):
    """
    [in] hiddenin is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YSize, XSize
    [out] poolingout is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YPool, XPool

    linear poolin is a convolution in each layer with a all-one matrix with size (XSize-XPool+1,YSize-YPool+1)
    XSize-XPool+1 must agree with YSize-YPool+1
    Result stores in poolingout
    """
    ASSERTTYPE(hiddenin)
    ASSERTTYPE(poolingout)

    (ni,li,yi,xi)=hiddenin.shape
    (no,lo,yp,xp)=poolingout.shape
    px=xi-xp+1
    py=yi-yp+1

    if ni!=no: raise Exception, "Data count mismatch"
    if li!=lo: raise Exception, "Layers mismatch"
    if px!=py: raise Exception, "Pooling size mismatch"
    if px<=0: raise Exception, "Invalid pooling size"

    hiddenin=hiddenin.ctypes.data_as(ct.POINTER(ct.c_float))
    poolingout=poolingout.ctypes.data_as(ct.POINTER(ct.c_float))

    CASSERT(_dll.pooling_in_kern(hiddenin,poolingout,ni,li,yi,xi,px))

@multicall(0,0)
def linearpoolout(poolingin,hiddenout):
    """
    [in] poolingin is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YPool, XPool
    [out] hiddenout is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YSize, XSize

    linear poolout is a expanded convolution in each layer with a all-one matrix with size (XSize-XPool+1,YSize-YPool+1)
    XSize-XPool+1 must agree with YSize-YPool+1
    Result stores in hiddenout
    """
    ASSERTTYPE(hiddenout)
    ASSERTTYPE(poolingin)

    (ni,li,yi,xi)=hiddenout.shape
    (no,lo,yp,xp)=poolingin.shape
    px=xi-xp+1
    py=yi-yp+1

    if ni!=no: raise Exception, "Data count mismatch"
    if li!=lo: raise Exception, "Layers mismatch"
    if px!=py: raise Exception, "Pooling size mismatch"
    if px<=0: raise Exception, "Invalid pooling size"

    hiddenout=hiddenout.ctypes.data_as(ct.POINTER(ct.c_float))
    poolingin=poolingin.ctypes.data_as(ct.POINTER(ct.c_float))

    CASSERT(_dll.pooling_out_kern(poolingin,hiddenout,ni,li,yi,xi,px))

@multicall(0,0,None)
def linearlayer(hiddenin,poolingout,size):
    """
    [in] hiddenin is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YSize, XSize
    [out] poolingout is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers/size, YSize, XSize

    Sum values in adjacent layers
    """
    ASSERTTYPE(poolingout)
    ASSERTTYPE(hiddenin)

    (ni,li,yi,xi)=hiddenin.shape
    (no,lo,yo,xo)=poolingout.shape

    if ni!=no: raise Exception, "Data count mismatch"
    if lo*size!=li: raise Exception, "Layers mismatch"
    if yi!=yo or xi!=xo: raise Exception, "Data dimension mismatch"
    
    poolingout=poolingout.ctypes.data_as(ct.POINTER(ct.c_float))
    hiddenin=hiddenin.ctypes.data_as(ct.POINTER(ct.c_float))

    _dll.layerpooling_in(hiddenin,poolingout,ni,li,yi,xi,size)

@multicall(0,0,None)
def reverselinearlayer(poolingin,hiddenout,size):
    """
    [in] poolingin is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YSize, XSize
    [out] hiddenout is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers*size, YSize, XSize

    Propagate value evenly to adjacent layers
    """
    ASSERTTYPE(poolingin)
    ASSERTTYPE(hiddenout)

    (ni,li,yi,xi)=poolingin.shape
    (no,lo,yo,xo)=hiddenout.shape

    if ni!=no: raise Exception, "Data count mismatch"
    if li*size!=lo: raise Exception, "Layers mismatch"
    if yi!=yo or xi!=xo: raise Exception, "Data dimension mismatch"
    
    poolingin=poolingin.ctypes.data_as(ct.POINTER(ct.c_float))
    hiddenout=hiddenout.ctypes.data_as(ct.POINTER(ct.c_float))

    _dll.layerpooling_out(poolingin,hiddenout,ni,lo,yi,xi,size)

@multicall(0,0,None)
def inlayermax_keepshape(data,out,size):
    """
    [in] data is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YSize, XSize
    [out] out is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YSize, XSize

    In layer max pooling, scan through whole surface inside one layer, scan size: [-size+1, size-1]
    """
    ASSERTTYPE(data)
    ASSERTTYPE(out)

    (ni,li,yi,xi)=data.shape
    (no,lo,yo,xo)=out.shape

    if data.shape!=out.shape: raise Exception, "Shape mismatch"

    data=data.ctypes.data_as(ct.POINTER(ct.c_float))
    out=out.ctypes.data_as(ct.POINTER(ct.c_float))

    CASSERT(_dll.inlayermax_keepshape(data,out,ni,li,yi,xi,size))

@multicall(0,0,0,None)
def reverseinlayermax_keepshape(data,grad,outgrad,size):
    """
    [in] data is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YSize, XSize
    [in] grad is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YSize, XSize
    [out] outgrad is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YSize, XSize

    In layer max pooling, scan through whole surface inside one layer, scan size: [-size+1, size-1]
    When outputs at some point, it output grad instead of data
    """
    ASSERTTYPE(dat)
    ASSERTTYPE(grad)
    ASSERTTYPE(outgrad)

    (ni,li,yi,xi)=data.shape

    if data.shape!=grad.shape: raise Exception, "Grad shape mismatch"
    if data.shape!=outgrad.shape: raise Exception, "OutGrad shape mismatch"

    data=data.ctypes.data_as(ct.POINTER(ct.c_float))
    grad=grad.ctypes.data_as(ct.POINTER(ct.c_float))
    outgrad=outgrad.ctypes.data_as(ct.POINTER(ct.c_float))

    CASSERT(_dll.reverseinlayermax_keepshape(data,grad,outgrad,ni,li,yi,xi,size))

@multicall(0,0,None)
def alllayermax_keepshape(data,out,size):
    """
    [in] data is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YSize, XSize
    [out] out is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YSize, XSize

    In layer max pooling, scan through whole surface of all layers, scan size: [-size+1, size-1]
    """
    ASSERTTYPE(data)
    ASSERTTYPE(out)

    (ni,li,yi,xi)=data.shape
    (no,lo,yo,xo)=out.shape

    if data.shape!=out.shape: raise Exception, "Shape mismatch"

    data=data.ctypes.data_as(ct.POINTER(ct.c_float))
    out=out.ctypes.data_as(ct.POINTER(ct.c_float))

    CASSERT(_dll.cpu_alllayermax_keepshape(data,out,ni,li,yi,xi,size))

@multicall(0,0,0,None)
def reversealllayermax_keepshape(data,grad,outgrad,size):
    """
    [in] data is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YSize, XSize
    [in] grad is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YSize, XSize
    [out] outgrad is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YSize, XSize

    In layer max pooling, scan through whole surface of all layers, scan size: [-size+1, size-1]
    When outputs at some point, it output grad instead of data
    """
    ASSERTTYPE(dat)
    ASSERTTYPE(grad)
    ASSERTTYPE(outgrad)

    (ni,li,yi,xi)=data.shape

    if data.shape!=grad.shape: raise Exception, "Grad shape mismatch"
    if data.shape!=outgrad.shape: raise Exception, "OutGrad shape mismatch"

    data=data.ctypes.data_as(ct.POINTER(ct.c_float))
    grad=grad.ctypes.data_as(ct.POINTER(ct.c_float))
    outgrad=outgrad.ctypes.data_as(ct.POINTER(ct.c_float))

    CASSERT(_dll.cpu_reversealllayermax_keepshape(data,grad,outgrad,ni,li,yi,xi,size))


@multicall(0,0,None)
def inlayerabsmax_keepshape(data,out,size):
    """
    [in] data is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YSize, XSize
    [out] out is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YSize, XSize

    In layer (abs)max pooling, scan through whole surface inside one layer, scan size: [-size+1, size-1]
    """
    ASSERTTYPE(data)
    ASSERTTYPE(out)

    (ni,li,yi,xi)=data.shape
    (no,lo,yo,xo)=out.shape

    if data.shape!=out.shape: raise Exception, "Shape mismatch"

    data=data.ctypes.data_as(ct.POINTER(ct.c_float))
    out=out.ctypes.data_as(ct.POINTER(ct.c_float))

    CASSERT(_dll.inlayerabsmax_keepshape(data,out,ni,li,yi,xi,size))

@multicall(0,0,0,None)
def reverseinlayerabsmax_keepshape(data,grad,outgrad,size):
    """
    [in] data is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YSize, XSize
    [in] grad is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YSize, XSize
    [out] outgrad is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YSize, XSize

    In layer (abs)max pooling, scan through whole surface inside one layer, scan size: [-size+1, size-1]
    When outputs at some point, it output grad instead of data
    """
    ASSERTTYPE(data)
    ASSERTTYPE(grad)
    ASSERTTYPE(outgrad)

    (ni,li,yi,xi)=data.shape

    if data.shape!=grad.shape: raise Exception, "Grad shape mismatch"
    if data.shape!=outgrad.shape: raise Exception, "OutGrad shape mismatch"

    data=data.ctypes.data_as(ct.POINTER(ct.c_float))
    grad=grad.ctypes.data_as(ct.POINTER(ct.c_float))
    outgrad=outgrad.ctypes.data_as(ct.POINTER(ct.c_float))

    CASSERT(_dll.reverseinlayerabsmax_keepshape(data,grad,outgrad,ni,li,yi,xi,size))

@multicall(0,0,None)
def alllayerabsmax_keepshape(data,out,size):
    """
    [in] data is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YSize, XSize
    [out] out is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YSize, XSize

    In layer (abs)max pooling, scan through whole surface of all layers, scan size: [-size+1, size-1]
    """
    ASSERTTYPE(data)
    ASSERTTYPE(out)

    (ni,li,yi,xi)=data.shape
    (no,lo,yo,xo)=out.shape

    if data.shape!=out.shape: raise Exception, "Shape mismatch"

    data=data.ctypes.data_as(ct.POINTER(ct.c_float))
    out=out.ctypes.data_as(ct.POINTER(ct.c_float))

    CASSERT(_dll.alllayerabsmax_keepshape(data,out,ni,li,yi,xi,size))

@multicall(0,0,0,None)
def reversealllayerabsmax_keepshape(data,grad,outgrad,size):
    """
    [in] data is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YSize, XSize
    [in] grad is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YSize, XSize
    [out] outgrad is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YSize, XSize

    In layer (abs)max pooling, scan through whole surface of all layers, scan size: [-size+1, size-1]
    When outputs at some point, it output grad instead of data
    """
    ASSERTTYPE(dat)
    ASSERTTYPE(grad)
    ASSERTTYPE(outgrad)

    (ni,li,yi,xi)=data.shape

    if data.shape!=grad.shape: raise Exception, "Grad shape mismatch"
    if data.shape!=outgrad.shape: raise Exception, "OutGrad shape mismatch"

    data=data.ctypes.data_as(ct.POINTER(ct.c_float))
    grad=grad.ctypes.data_as(ct.POINTER(ct.c_float))
    outgrad=outgrad.ctypes.data_as(ct.POINTER(ct.c_float))

    CASSERT(_dll.reversealllayerabsmax_keepshape(data,grad,outgrad,ni,li,yi,xi,size))


@multicall(0,0,None)
def maxblock2D(data,out,size):
    """
    [in] data is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YSize, XSize
    [out] out is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YOut((YSize+size-1)/size), XOut((XSize+size-1)/size)

    maxblock does a max-pooling in-layer spatially.
    It splits data into blocks with size, and get max of each block.
    Stores result in out.
    """
    ASSERTTYPE(data)
    ASSERTTYPE(out)

    (ni,li,yi,xi)=data.shape
    (no,lo,yo,xo)=out.shape

    if ni!=no: raise Exception, "Data count mismatch"
    if li!=lo: raise Exception, "Layers mismatch"
    if size<=0: raise Exception, "Invalid pooling size"
    if (yi+size-1)/size!=yo or (xi+size-1)/size!=xo: raise Exception, "Pooling size not correct"


    data=data.ctypes.data_as(ct.POINTER(ct.c_float))
    out=out.ctypes.data_as(ct.POINTER(ct.c_float))

    CASSERT(_dll.maxblock2D(data,out,ni,li,yi,xi,size))

@multicall(0,0,0,None)
def reversemaxblock2D(grad,data,out,size):
    """
    [in] data is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YSize, XSize
    [in] grad is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YOut((YSize+size-1)/size), XOut((XSize+size-1)/size)
    [out] out is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YSize, XSize

    This function does a reverse passing of gradient or values through a max-pooling procedure.
    """
    ASSERTTYPE(grad)
    ASSERTTYPE(data)
    ASSERTTYPE(out)

    (ni,li,yi,xi)=data.shape
    (no,lo,yo,xo)=grad.shape

    if ni!=no: raise Exception, "Data count mismatch"
    if li!=lo: raise Exception, "Layers mismatch"
    if size<=0: raise Exception, "Invalid pooling size"
    if (yi+size-1)/size!=yo or (xi+size-1)/size!=xo: raise Exception, "Pooling size not correct"
    if data.shape!=out.shape: raise Exception, "Out shape not agree with data shape"

    data=data.ctypes.data_as(ct.POINTER(ct.c_float))
    out=out.ctypes.data_as(ct.POINTER(ct.c_float))
    grad=grad.ctypes.data_as(ct.POINTER(ct.c_float))
    
    CASSERT(_dll.reversemaxblock2D(grad,data,out,ni,li,yi,xi,size))

@multicall(0,0,0,None)
def followmaxblock2D(grad,data,out,size):
    """
    [in] data is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YSize, XSize
    [in] grad is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YSize, XSize
    [out] out is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YOut((YSize+size-1)/size), XOut((XSize+size-1)/size)

    This function does a forward passing of gradient or values through a max-pooling procedure.
    """
    ASSERTTYPE(grad)
    ASSERTTYPE(data)
    ASSERTTYPE(out)

    (ni,li,yi,xi)=data.shape
    (no,lo,yo,xo)=out.shape

    if ni!=no: raise Exception, "Data count mismatch"
    if li!=lo: raise Exception, "Layers mismatch"
    if size<=0: raise Exception, "Invalid pooling size"
    if (yi+size-1)/size!=yo or (xi+size-1)/size!=xo: raise Exception, "Pooling size not correct"
    if data.shape!=grad.shape: raise Exception, "Grad shape not agree with data shape"

    data=data.ctypes.data_as(ct.POINTER(ct.c_float))
    out=out.ctypes.data_as(ct.POINTER(ct.c_float))
    grad=grad.ctypes.data_as(ct.POINTER(ct.c_float))
    
    CASSERT(_dll.followmaxblock2D(grad,data,out,ni,li,yi,xi,size))

@multicall(0,0,None)
def squareblock2D(data,out,size):
    """
    [in] data is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YSize, XSize
    [out] out is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YOut((YSize+size-1)/size), XOut((XSize+size-1)/size)

    squareblock does a square-pooling in-layer spatially.
    It splits data into blocks with size, and get square sum of each block, and does a square root (with epsilon added)
    Stores result in out.
    """
    ASSERTTYPE(data)
    ASSERTTYPE(out)

    (ni,li,yi,xi)=data.shape
    (no,lo,yo,xo)=out.shape

    if ni!=no: raise Exception, "Data count mismatch"
    if li!=lo: raise Exception, "Layers mismatch"
    if size<=0: raise Exception, "Invalid pooling size"
    if (yi+size-1)/size!=yo or (xi+size-1)/size!=xo: raise Exception, "Pooling size not correct"

    data=data.ctypes.data_as(ct.POINTER(ct.c_float))
    out=out.ctypes.data_as(ct.POINTER(ct.c_float))

    CASSERT(_dll.squareblock2D(data,out,ni,li,yi,xi,size))

@multicall(0,0,0,0,None)
def reversesquareblock2D(grad,data,outdata,out,size):
    """
    [in] data is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YSize, XSize
    [in] grad is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YOut((YSize+size-1)/size), XOut((XSize+size-1)/size)
    [in] outdata is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YOut((YSize+size-1)/size), XOut((XSize+size-1)/size)
    [out] out is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YSize, XSize

    This function does a reverse passing of gradient or values through a square-pooling procedure.
    """
    ASSERTTYPE(grad)
    ASSERTTYPE(data)
    ASSERTTYPE(outdata)
    ASSERTTYPE(out)

    (ni,li,yi,xi)=data.shape
    (no,lo,yo,xo)=outdata.shape

    if ni!=no: raise Exception, "Data count mismatch"
    if li!=lo: raise Exception, "Layers mismatch"
    if size<=0: raise Exception, "Invalid pooling size"
    if (yi+size-1)/size!=yo or (xi+size-1)/size!=xo: raise Exception, "Pooling size not correct"
    if data.shape!=out.shape: raise Exception, "Out shape not agree with data shape"
    if outdata.shape!=grad.shape: raise Exception, "outdata shape not agree with grad shape"

    data=data.ctypes.data_as(ct.POINTER(ct.c_float))
    out=out.ctypes.data_as(ct.POINTER(ct.c_float))
    grad=grad.ctypes.data_as(ct.POINTER(ct.c_float))
    outdata=outdata.ctypes.data_as(ct.POINTER(ct.c_float))
    
    CASSERT(_dll.reversesquareblock2D(grad,data,outdata,out,ni,li,yi,xi,size))

@multicall(0,0,0,0,None)
def followsquareblock2D(grad,data,outdata,out,size):
    """
    [in] data is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YSize, XSize
    [in] grad is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YSize, XSize
    [in] outdata is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YOut((YSize+size-1)/size), XOut((XSize+size-1)/size)
    [out] out is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YOut((YSize+size-1)/size), XOut((XSize+size-1)/size)

    This function does a forward passing of gradient or values through a square-pooling procedure.
    """
    ASSERTTYPE(grad)
    ASSERTTYPE(data)
    ASSERTTYPE(outdata)
    ASSERTTYPE(out)

    (ni,li,yi,xi)=data.shape
    (no,lo,yo,xo)=outdata.shape

    if ni!=no: raise Exception, "Data count mismatch"
    if li!=lo: raise Exception, "Layers mismatch"
    if size<=0: raise Exception, "Invalid pooling size"
    if (yi+size-1)/size!=yo or (xi+size-1)/size!=xo: raise Exception, "Pooling size not correct"
    if outdata.shape!=out.shape: raise Exception, "Out shape not agree with outdata shape"
    if data.shape!=grad.shape: raise Exception, "data shape not agree with grad shape"

    data=data.ctypes.data_as(ct.POINTER(ct.c_float))
    out=out.ctypes.data_as(ct.POINTER(ct.c_float))
    grad=grad.ctypes.data_as(ct.POINTER(ct.c_float))
    outdata=outdata.ctypes.data_as(ct.POINTER(ct.c_float))
    
    CASSERT(_dll.followsquareblock2D(grad,data,outdata,out,ni,li,yi,xi,size))

@multicall(0,0,None)
def squarelayer(data,out,size):
    """
    [in] data is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YSize, XSize
    [out] out is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, PoolingLayers(Layers/size), YSize, XSize

    This function does a cross layer square pooling.
    It groups adjacent Layers by size, square, add and square root, and stores result in out.
    """
    ASSERTTYPE(data)
    ASSERTTYPE(out)

    (ni,li,yi,xi)=data.shape
    (no,lo,yo,xo)=out.shape

    if ni!=no: raise Exception, "Data count mismatch"
    if yi!=yo or xi!=xo: raise Exception, "Size mismatch"
    if size<=0: raise Exception, "Invalid size"
    if li%size!=0: raise Exception, "Layers must be multiple of size"
    if lo!=li/size: raise Exception, "Output layers must be (input_layers/size)"

    data=data.ctypes.data_as(ct.POINTER(ct.c_float))
    out=out.ctypes.data_as(ct.POINTER(ct.c_float))

    CASSERT(_dll.squarelayer(data,out,ni,li,yi,xi,size))

@multicall(0,0,0,0,None)
def reversesquarelayer(grad,data,outdata,out,size):
    """
    [in] data is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YSize, XSize
    [in] grad is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, PoolingLayers(Layers/size), YSize, XSize
    [in] outdata is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, PoolingLayers(Layers/size), YSize, XSize
    [out] out is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YSize, XSize

    This function does a reverse grad passing across cross layer square pooling.
    Input grad, function calculates grad before pooling and put results in out.
    Algorithm reuses squarelayer's out as outdata to reduce calculation.
    """
    ASSERTTYPE(grad)
    ASSERTTYPE(data)
    ASSERTTYPE(outdata)
    ASSERTTYPE(out)

    (ni,li,yi,xi)=data.shape
    (no,lo,yo,xo)=outdata.shape

    if ni!=no: raise Exception, "Data count mismatch"
    if yi!=yo or xi!=xo: raise Exception, "Size mismatch"
    if size<=0: raise Exception, "Invalid size"
    if li%size!=0: raise Exception, "Layers must be multiple of size"
    if lo!=li/size: raise Exception, "Output layers must be (input_layers/size)"
    if data.shape!=out.shape: raise Exception, "out shape mismatch"
    if outdata.shape!=grad.shape: raise Exception, "grad shape mismatch"

    grad=grad.ctypes.data_as(ct.POINTER(ct.c_float))
    data=data.ctypes.data_as(ct.POINTER(ct.c_float))
    outdata=outdata.ctypes.data_as(ct.POINTER(ct.c_float))
    out=out.ctypes.data_as(ct.POINTER(ct.c_float))
    
    CASSERT(_dll.reversesquarelayer(grad,data,outdata,out,ni,li,yi,xi,size))

@multicall(0,0,0,0,None)
def followsquarelayer(grad,data,outdata,out,size):
    """
    [in] data is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YSize, XSize
    [in] grad is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YSize, XSize
    [in] outdata is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, PoolingLayers(Layers/size), YSize, XSize
    [out] out is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, PoolingLayers(Layers/size), YSize, XSize

    This function does a forwarding grad passing across cross layer square pooling.
    Input grad, function calculates grad before pooling and put results in out.
    Algorithm reuses squarelayer's out as outdata to reduce calculation.
    """
    ASSERTTYPE(grad)
    ASSERTTYPE(data)
    ASSERTTYPE(outdata)
    ASSERTTYPE(out)

    (ni,li,yi,xi)=data.shape
    (no,lo,yo,xo)=outdata.shape

    if ni!=no: raise Exception, "Data count mismatch"
    if yi!=yo or xi!=xo: raise Exception, "Size mismatch"
    if size<=0: raise Exception, "Invalid size"
    if li%size!=0: raise Exception, "Layers must be multiple of size"
    if lo!=li/size: raise Exception, "Output layers must be (input_layers/size)"
    if data.shape!=grad.shape: raise Exception, "grad shape mismatch"
    if outdata.shape!=out.shape: raise Exception, "out shape mismatch"

    grad=grad.ctypes.data_as(ct.POINTER(ct.c_float))
    data=data.ctypes.data_as(ct.POINTER(ct.c_float))
    outdata=outdata.ctypes.data_as(ct.POINTER(ct.c_float))
    out=out.ctypes.data_as(ct.POINTER(ct.c_float))
    
    CASSERT(_dll.followsquarelayer(grad,data,outdata,out,ni,li,yi,xi,size))

@multicall(0,0,None)
def maxlayer(data,out,size):
    """
    [in] data is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YSize, XSize
    [out] out is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, PoolingLayers(Layers/size), YSize, XSize

    This function does a cross layer max pooling.
    It groups adjacent Layers by size, get max of it, and stores result in out.
    """
    ASSERTTYPE(data)
    ASSERTTYPE(out)

    (ni,li,yi,xi)=data.shape
    (no,lo,yo,xo)=out.shape

    if ni!=no: raise Exception, "Data count mismatch"
    if yi!=yo or xi!=xo: raise Exception, "Size mismatch"
    if size<=0: raise Exception, "Invalid size"
    if li%size!=0: raise Exception, "Layers must be multiple of size"
    if lo!=li/size: raise Exception, "Output layers must be (input_layers/size)"

    data=data.ctypes.data_as(ct.POINTER(ct.c_float))
    out=out.ctypes.data_as(ct.POINTER(ct.c_float))

    CASSERT(_dll.maxlayer(data,out,ni,li,yi,xi,size))

@multicall(0,0,0,None)
def reversemaxlayer(grad,data,out,size):
    """
    [in] data is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YSize, XSize
    [in] grad is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, PoolingLayers(Layers/size), YSize, XSize
    [out] out is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YSize, XSize

    This function does a reverse grad passing across cross layer max pooling.
    Input grad, function calculates grad before pooling and put results in out.
    """
    ASSERTTYPE(grad)
    ASSERTTYPE(data)
    ASSERTTYPE(out)

    (ni,li,yi,xi)=data.shape
    (no,lo,yo,xo)=grad.shape

    if ni!=no: raise Exception, "Data count mismatch"
    if yi!=yo or xi!=xo: raise Exception, "Size mismatch"
    if size<=0: raise Exception, "Invalid size"
    if li%size!=0: raise Exception, "Layers must be multiple of size"
    if lo!=li/size: raise Exception, "Output layers must be (input_layers/size)"
    if data.shape!=out.shape: raise Exception, "out shape mismatch"

    grad=grad.ctypes.data_as(ct.POINTER(ct.c_float))
    data=data.ctypes.data_as(ct.POINTER(ct.c_float))
    out=out.ctypes.data_as(ct.POINTER(ct.c_float))
    
    CASSERT(_dll.reversemaxlayer(grad,data,out,ni,li,yi,xi,size))

@multicall(0,0,0,None)
def followmaxlayer(grad,data,out,size):
    """
    [in] data is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YSize, XSize
    [in] grad is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Layers, YSize, XSize
    [out] out is a 4d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, PoolingLayers(Layers/size), YSize, XSize

    This function does a forwarding grad passing across cross layer max pooling.
    Input grad, function calculates grad before pooling and put results in out.
    """
    ASSERTTYPE(grad)
    ASSERTTYPE(data)
    ASSERTTYPE(out)

    (ni,li,yi,xi)=data.shape
    (no,lo,yo,xo)=out.shape

    if ni!=no: raise Exception, "Data count mismatch"
    if yi!=yo or xi!=xo: raise Exception, "Size mismatch"
    if size<=0: raise Exception, "Invalid size"
    if li%size!=0: raise Exception, "Layers must be multiple of size"
    if lo!=li/size: raise Exception, "Output layers must be (input_layers/size)"
    if data.shape!=grad.shape: raise Exception, "grad shape mismatch"

    grad=grad.ctypes.data_as(ct.POINTER(ct.c_float))
    data=data.ctypes.data_as(ct.POINTER(ct.c_float))
    out=out.ctypes.data_as(ct.POINTER(ct.c_float))
    
    CASSERT(_dll.followmaxlayer(grad,data,out,ni,li,yi,xi,size))

@multicall(0,0)
def removeDC(data,out):
    """
    [in] data is a 2d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Content-Dim(reshape first)
    [out] out is a 2d-ndarray with dtype=float32, with same dimension ordering

    out[pd,pc]=data[pd,pc]-(sum(data[1..n,pc])/n)
    """
    ASSERTTYPE(data)
    ASSERTTYPE(out)

    if len(data.shape)!=2: raise Exception, "2d-ndarray needed (data)."
    if len(out.shape)!=2: raise Exception, "2d-ndarray needed (out)."
    if data.shape!=out.shape: raise Exception, "Dimension not the same"
    (nd,nc) = data.shape

    data=data.ctypes.data_as(ct.POINTER(ct.c_float))
    out=out.ctypes.data_as(ct.POINTER(ct.c_float))

    CASSERT(_dll.removeDC(data,out,nd,nc))

#Permutation cannot be multicall
def blockpermutation(data,order):
    """
    [in,out] data is 2d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Data, Content-Dim
    [in] order is 1d-ndarray with dtype=uint32/int32, each element indicates data piece's new position.

    Modify data itself to represent permutation result.
    """
    ASSERTTYPE(data)
    if data.dtype.name in ('uint32','int32'):
        raise Exception, "Order should be dtype uint32/int32, %s got."%v.dtype.name
    
    if len(data.shape)!=2: raise Exception, "2d-ndarray needed (data)."
    order=np.array(order,np.int32)
    if len(order.shape)!=1: raise Exception, "1d-ndarray needed (order)."
    if data.shape[0]!=order.shape[0]: raise Exception, "Invalid order argument"
    (nd,nc) = data.shape

    data=data.ctypes.data_as(ct.POINTER(ct.c_float))
    porder=order.ctypes.data_as(ct.POINTER(ct.c_int32))

    CASSERT(_dll.blockpermutation(data,porder,nd,nc))

def normalize(data):
    """
    [in,out] data is 2d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Vectors, Length-Of-Vector

    Function transforms data to make each vector's square norm to 1.
    """
    ASSERTTYPE(data)
    if len(data.shape)!=2: raise Exception, "Invalid data shape"
    v,l=data.shape
    data=data.ctypes.data_as(ct.POINTER(ct.c_float))

    CASSERT(_dll.normalize(data,v,l))

def balance(data):
    """
    [in,out] data is 2d-ndarray with dtype=float32, with dimension ordering:
        Number_Of_Vectors, Length-Of-Vector

    Function transforms data to make each vector's mean to 0.
    """
    ASSERTTYPE(data)
    if len(data.shape)!=2: raise Exception, "Invalid data shape"
    v,l=data.shape
    data=data.ctypes.data_as(ct.POINTER(ct.c_float))

    _dll.balance(data,v,l)

import lru as _lru,struct as _struct
#In case that one uses variable floating point
_trans_cache = _lru.LRUDict(100000)
TRANSFORM_STACK_LENGTH = 6
_trans_lock = threading.Lock()
def transform_gnumpy(expr,result,a,b=None,c=None,d=None):
    """
    This function carries a configurable stack based expression to gpu, and returns result.
    Here result,a,b,c,d should be gnumpy arrays which carries direct pointer to GPU (single float)
    [out] result, [in] a, [in] b, [in] c, [in] d

    Available functions:
    a,b,c,d,1,[float],id,add,sub,neg,mul,div,mod,sin,cos,tan,cot,sec,csc,
    inv,inc,exp,log,sinh,cosh,tanh,coth,sech,csch,sqr,sqrt,pow,asin,acos,
    atan,fabs,ceil,floor,sigmoid,sigmoidgrad,dup1,dup2,dup3,dup4,>,<,==,
    >=,<=,!=,===,and,or,xor,not,?:,isnan,isinf

    Can insert # after function body, will ignore everything after #

    Example: "a b add # returns a+b"
    """
    global _trans_lock,_trans_cache,_TRANSFORM_TABLE,TRANSFORM_STACK_LENGTH
    ASSERTTYPEG(result)
    resultlen=result._base.shape[0]

    ASSERTTYPEG(a)
    if a._base.shape[0]!=resultlen: raise Exception, "Length of a not agree with result"
    r=1
    if b!=None:
        ASSERTTYPEG(b)
        if b._base.shape[0]!=resultlen: raise Exception, "Length of b not agree with result"
        r=2
    if c!=None:
        ASSERTTYPEG(c)
        if c._base.shape[0]!=resultlen: raise Exception, "Length of c not agree with result"
        r=3
    if d!=None:
        ASSERTTYPEG(d)
        if b._base.shape[0]!=resultlen: raise Exception, "Length of d not agree with result"
        r=4

    if (b is None) and (c is not None) or (d is not None): raise Exception, "Cannot skip None variable (b)."
    if (c is None) and (d is not None): raise Exception, "Cannot skip None variable (c)."

    #Check formula and convert
    with _trans_lock:
        if expr not in _trans_cache:
            stacklen=0
            rightvar=0
            #Translate expr, and check overflow/underflow of stack
            cmdi=expr.split(' ')
            lcmdo=[]
            count=0
            for i in cmdi:
                count+=1
                if i in ['','\t','\n','\r']:
                    count-=1
                    continue
                if i[0]=='#': break
                if i[0]=='[':
                    if i[-1]!=']':
                        raise Exception, "Invalid constant format"
                    ival=i[1:-1]
                    ival=float(ival)
                    stacklen+=1
                    if stacklen>TRANSFORM_STACK_LENGTH: raise Exception, "Stack overflow on instruction %s(%d)"%(i,count)
                    lcmdo.append('\x05')
                    lcmdo.append(_struct.pack('f',ival))
                elif i not in _TRANSFORM_TABLE:
                    raise Exception, "Unknown instruction: %s(%d)"%(i,count)
                else:
                    var,cmd,stk,stkupd=_TRANSFORM_TABLE[i]
                    if var>rightvar: rightvar=var
                    if stk>stacklen: raise Exception, "Stack underflow on instruction %s(%d)"%(i,count)
                    stacklen+=stkupd
                    if stacklen>TRANSFORM_STACK_LENGTH: raise Exception, "Stack overflow on instruction %s(%d)"%(i,count)
                    lcmdo.append(cmd)
            cmdo=''.join(lcmdo)
            if len(cmdo)>1024: raise Exception, "Instruction too long"
            _trans_cache[expr]=(cmdo, rightvar, stacklen)
        cmdo, rightvar, stacklen = _trans_cache[expr]
    if stacklen<1: raise Exception, "Insufficient return stack size"
    if rightvar>r: raise Exception, "Insufficient variable, needs %d, given %d."%(rightvar,r)
    
    _transform_kern_gnumpy(a, cmdo, b, c, d, result)

@multicall(None,0,0,[0],[0],[0])
def transformD(expr,result,a,b=None,c=None,d=None):
    """
    This function carries a configurable stack based expression to gpu, and returns result.
    [out] result, [in] a, [in] b, [in] c, [in] d

    Available functions:
    a,b,c,d,1,[float],id,add,sub,neg,mul,div,mod,sin,cos,tan,cot,sec,csc,
    inv,inc,exp,log,sinh,cosh,tanh,coth,sech,csch,sqr,sqrt,pow,asin,acos,
    atan,fabs,ceil,floor,sigmoid,sigmoidgrad,dup1,dup2,dup3,dup4,>,<,==,
    >=,<=,!=,===,and,or,xor,not,?:,isnan,isinf

    Can insert # after function body, will ignore everything after #

    Example: "a b add # returns a+b"
    """
    global _trans_lock,_trans_cache,_TRANSFORM_TABLE,TRANSFORM_STACK_LENGTH
    ASSERTTYPED(result)
    if len(result.shape)!=1: raise Exception, "Invalid result shape"
    resultlen=result.shape[0]
    ASSERTTYPED(a)
    if len(a.shape)!=1: raise Exception, "Invalid a shape"
    if a.shape[0]!=resultlen: raise Exception, "Length of a not agree with result"
    r=1
    if b!=None:
        ASSERTTYPED(b)
        if len(b.shape)!=1: raise Exception, "Invalid b shape"
        if b.shape[0]!=resultlen: raise Exception, "Length of b not agree with result"
        r=2
    if c!=None:
        ASSERTTYPED(c)
        if len(b.shape)!=1: raise Exception, "Invalid c shape"
        if b.shape[0]!=resultlen: raise Exception, "Length of c not agree with result"
        r=3
    if d!=None:
        ASSERTTYPED(d)
        if len(b.shape)!=1: raise Exception, "Invalid d shape"
        if b.shape[0]!=resultlen: raise Exception, "Length of d not agree with result"
        r=4

    if (b is None) and (c is not None) or (d is not None): raise Exception, "Cannot skip None variable (b)."
    if (c is None) and (d is not None): raise Exception, "Cannot skip None variable (c)."

    #Check formula and convert
    with _trans_lock:
        if '\x01\x02\x03'+expr not in _trans_cache:
            stacklen=0
            rightvar=0
            #Translate expr, and check overflow/underflow of stack
            cmdi=expr.split(' ')
            lcmdo=[]
            count=0
            for i in cmdi:
                count+=1
                if i in ['','\t','\n','\r']:
                    count-=1
                    continue
                if i[0]=='#': break
                if i[0]=='[':
                    if i[-1]!=']':
                        raise Exception, "Invalid constant format"
                    ival=i[1:-1]
                    ival=float(ival)
                    stacklen+=1
                    if stacklen>TRANSFORM_STACK_LENGTH: raise Exception, "Stack overflow on instruction %s(%d)"%(i,count)
                    lcmdo.append('\x05')
                    lcmdo.append(_struct.pack('d',ival))
                elif i not in _TRANSFORM_TABLE:
                    raise Exception, "Unknown instruction: %s(%d)"%(i,count)
                else:
                    var,cmd,stk,stkupd=_TRANSFORM_TABLE[i]
                    if var>rightvar: rightvar=var
                    if stk>stacklen: raise Exception, "Stack underflow on instruction %s(%d)"%(i,count)
                    stacklen+=stkupd
                    if stacklen>TRANSFORM_STACK_LENGTH: raise Exception, "Stack overflow on instruction %s(%d)"%(i,count)
                    lcmdo.append(cmd)
            cmdo=''.join(lcmdo)
            if len(cmdo)>1024: raise Exception, "Instruction too long"
            _trans_cache['\x01\x02\x03'+expr]=(cmdo, rightvar, stacklen)
        cmdo, rightvar, stacklen = _trans_cache['\x01\x02\x03'+expr]
    if stacklen<1: raise Exception, "Insufficient return stack size"
    if rightvar>r: raise Exception, "Insufficient variable, needs %d, given %d."%(rightvar,r)

    _transformD_kern(a, cmdo, b, c, d, result)

@multicall(None,0,0,[0],[0],[0])
def transform(expr,result,a,b=None,c=None,d=None):
    """
    This function carries a configurable stack based expression to gpu, and returns result.
    [out] result, [in] a, [in] b, [in] c, [in] d

    Available functions:
    a,b,c,d,1,[float],id,add,sub,neg,mul,div,mod,sin,cos,tan,cot,sec,csc,
    inv,inc,exp,log,sinh,cosh,tanh,coth,sech,csch,sqr,sqrt,pow,asin,acos,
    atan,fabs,ceil,floor,sigmoid,sigmoidgrad,dup1,dup2,dup3,dup4,>,<,==,
    >=,<=,!=,===,and,or,xor,not,?:,isnan,isinf

    Can insert # after function body, will ignore everything after #

    Example: "a b add # returns a+b"
    """
    global _trans_lock,_trans_cache,_TRANSFORM_TABLE,TRANSFORM_STACK_LENGTH
    ASSERTTYPE(result)
    if len(result.shape)!=1: raise Exception, "Invalid result shape"
    resultlen=result.shape[0]
    ASSERTTYPE(a)
    if len(a.shape)!=1: raise Exception, "Invalid a shape"
    if a.shape[0]!=resultlen: raise Exception, "Length of a not agree with result"
    r=1
    if b!=None:
        ASSERTTYPE(b)
        if len(b.shape)!=1: raise Exception, "Invalid b shape"
        if b.shape[0]!=resultlen: raise Exception, "Length of b not agree with result"
        r=2
    if c!=None:
        ASSERTTYPE(c)
        if len(b.shape)!=1: raise Exception, "Invalid c shape"
        if b.shape[0]!=resultlen: raise Exception, "Length of c not agree with result"
        r=3
    if d!=None:
        ASSERTTYPE(d)
        if len(b.shape)!=1: raise Exception, "Invalid d shape"
        if b.shape[0]!=resultlen: raise Exception, "Length of d not agree with result"
        r=4

    if (b is None) and (c is not None) or (d is not None): raise Exception, "Cannot skip None variable (b)."
    if (c is None) and (d is not None): raise Exception, "Cannot skip None variable (c)."

    #Check formula and convert
    with _trans_lock:
        if expr not in _trans_cache:
            stacklen=0
            rightvar=0
            #Translate expr, and check overflow/underflow of stack
            cmdi=expr.split(' ')
            lcmdo=[]
            count=0
            for i in cmdi:
                count+=1
                if i in ['','\t','\n','\r']:
                    count-=1
                    continue
                if i[0]=='#': break
                if i[0]=='[':
                    if i[-1]!=']':
                        raise Exception, "Invalid constant format"
                    ival=i[1:-1]
                    ival=float(ival)
                    stacklen+=1
                    if stacklen>TRANSFORM_STACK_LENGTH: raise Exception, "Stack overflow on instruction %s(%d)"%(i,count)
                    lcmdo.append('\x05')
                    lcmdo.append(_struct.pack('f',ival))
                elif i not in _TRANSFORM_TABLE:
                    raise Exception, "Unknown instruction: %s(%d)"%(i,count)
                else:
                    var,cmd,stk,stkupd=_TRANSFORM_TABLE[i]
                    if var>rightvar: rightvar=var
                    if stk>stacklen: raise Exception, "Stack underflow on instruction %s(%d)"%(i,count)
                    stacklen+=stkupd
                    if stacklen>TRANSFORM_STACK_LENGTH: raise Exception, "Stack overflow on instruction %s(%d)"%(i,count)
                    lcmdo.append(cmd)
            cmdo=''.join(lcmdo)
            if len(cmdo)>1024: raise Exception, "Instruction too long"
            _trans_cache[expr]=(cmdo, rightvar, stacklen)
        cmdo, rightvar, stacklen = _trans_cache[expr]
    if stacklen<1: raise Exception, "Insufficient return stack size"
    if rightvar>r: raise Exception, "Insufficient variable, needs %d, given %d."%(rightvar,r)

    _transform_kern(a, cmdo, b, c, d, result)

def transform2_gnumpy(expr,result,result2,a,b=None,c=None,d=None):
    """
    This function carries a configurable stack based expression to gpu, and returns 2 results in stack pop order.
    Here result,result2,a,b,c,d should be gnumpy arrays which carries direct pointer to GPU (single float)
    [out] result, [out] result2, [in] a, [in] b, [in] c, [in] d


    Available functions:
    a,b,c,d,1,[float],id,add,sub,neg,mul,div,mod,sin,cos,tan,cot,sec,csc,
    inv,inc,exp,log,sinh,cosh,tanh,coth,sech,csch,sqr,sqrt,pow,asin,acos,
    atan,fabs,ceil,floor,sigmoid,sigmoidgrad,dup1,dup2,dup3,dup4,>,<,==,
    >=,<=,!=,===,and,or,xor,not,?:,swap,isnan,isinf

    Can insert # after function body, will ignore everything after #

    Example: "a b add a b sub # returns b-a a+b in result, result2"
    """
    global _trans_cache,_trans_lock,_TRANSFORM_TABLE,TRANSFORM_STACK_LENGTH
    ASSERTTYPEG(result)
    resultlen=result._base.shape[0]
    ASSERTTYPEG(result2)
    if resultlen!=result2._base.shape[0]: raise Exception, "Length of result2 not agree with result"
    ASSERTTYPEG(a)
    if a._base.shape[0]!=resultlen: raise Exception, "Length of a not agree with result"
    r=1
    if b!=None:
        ASSERTTYPEG(b)
        if b._base.shape[0]!=resultlen: raise Exception, "Length of b not agree with result"
        r=2
    if c!=None:
        ASSERTTYPEG(c)
        if c._base.shape[0]!=resultlen: raise Exception, "Length of c not agree with result"
        r=3
    if d!=None:
        ASSERTTYPEG(d)
        if b._base.shape[0]!=resultlen: raise Exception, "Length of d not agree with result"
        r=4

    if (b is None) and (c is not None) or (d is not None): raise Exception, "Cannot skip None variable (b)."
    if (c is None) and (d is not None): raise Exception, "Cannot skip None variable (c)."

    #Check formula and convert
    with _trans_lock:
        if expr not in _trans_cache:
            stacklen=0
            rightvar=0
            #Translate expr, and check overflow/underflow of stack
            cmdi=expr.split(' ')
            lcmdo=[]
            count=0
            for i in cmdi:
                count+=1
                if i in ['','\t','\n','\r']:
                    count-=1

                    continue
                if i[0]=='#': break
                if i[0]=='[':
                    if i[-1]!=']':
                        raise Exception, "Invalid constant format"
                    ival=i[1:-1]
                    ival=float(ival)
                    stacklen+=1
                    if stacklen>TRANSFORM_STACK_LENGTH: raise Exception, "Stack overflow on instruction %s(%d)"%(i,count)
                    lcmdo.append('\x05')
                    lcmdo.append(_struct.pack('f',ival))
                elif i not in _TRANSFORM_TABLE:
                    raise Exception, "Unknown instruction: %s(%d)"%(i,count)
                else:
                    var,cmd,stk,stkupd=_TRANSFORM_TABLE[i]
                    if var>rightvar: rightvar=var
                    if stk>stacklen: raise Exception, "Stack underflow on instruction %s(%d)"%(i,count)
                    stacklen+=stkupd
                    if stacklen>TRANSFORM_STACK_LENGTH: raise Exception, "Stack overflow on instruction %s(%d)"%(i,count)
                    lcmdo.append(cmd)
            cmdo=''.join(lcmdo)
            if len(cmdo)>1024: raise Exception, "Instruction too long"
            _trans_cache[expr]=(cmdo, rightvar, stacklen)
        cmdo, rightvar, stacklen = _trans_cache[expr]
    if stacklen<2: raise Exception, "Insufficient return stack size"
    if rightvar>r: raise Exception, "Insufficient variable, needs %d, given %d."%(rightvar,r)

    _transform2_kern_gnumpy(a, cmdo, b, c, d, result, result2)

@multicall(None,0,0,0,[0],[0],[0])
def transform2(expr,result,result2,a,b=None,c=None,d=None):
    """
    This function carries a configurable stack based expression to gpu, and returns 2 results in stack pop order.
    [out] result, [in] a, [in] b, [in] c, [in] d

    Available functions:
    a,b,c,d,1,[float],id,add,sub,neg,mul,div,mod,sin,cos,tan,cot,sec,csc,
    inv,inc,exp,log,sinh,cosh,tanh,coth,sech,csch,sqr,sqrt,pow,asin,acos,
    atan,fabs,ceil,floor,sigmoid,sigmoidgrad,dup1,dup2,dup3,dup4,>,<,==,
    >=,<=,!=,===,and,or,xor,not,?:,swap,isnan,isinf

    Can insert # after function body, will ignore everything after #

    Example: "a b add a b sub # returns b-a a+b in result, result2"
    """
    global _trans_cache,_trans_lock,_TRANSFORM_TABLE,TRANSFORM_STACK_LENGTH
    ASSERTTYPE(result)
    if len(result.shape)!=1: raise Exception, "Invalid result shape"
    resultlen=result.shape[0]
    ASSERTTYPE(result2)
    if len(result2.shape)!=1: raise Exception, "Invalid result2 shape"
    if resultlen!=result2.shape[0]: raise Exception, "Length of result2 not agree with result"
    ASSERTTYPE(a)
    if len(a.shape)!=1: raise Exception, "Invalid a shape"
    if a.shape[0]!=resultlen: raise Exception, "Length of a not agree with result"
    r=1
    if b!=None:
        ASSERTTYPE(b)
        if len(b.shape)!=1: raise Exception, "Invalid b shape"
        if b.shape[0]!=resultlen: raise Exception, "Length of b not agree with result"
        r=2
    if c!=None:
        ASSERTTYPE(c)
        if len(b.shape)!=1: raise Exception, "Invalid c shape"
        if b.shape[0]!=resultlen: raise Exception, "Length of c not agree with result"
        r=3
    if d!=None:
        ASSERTTYPE(d)
        if len(b.shape)!=1: raise Exception, "Invalid d shape"
        if b.shape[0]!=resultlen: raise Exception, "Length of d not agree with result"
        r=4

    if (b is None) and (c is not None) or (d is not None): raise Exception, "Cannot skip None variable (b)."
    if (c is None) and (d is not None): raise Exception, "Cannot skip None variable (c)."

    #Check formula and convert
    with _trans_lock:
        if expr not in _trans_cache:
            stacklen=0
            rightvar=0
            #Translate expr, and check overflow/underflow of stack
            cmdi=expr.split(' ')
            lcmdo=[]
            count=0
            for i in cmdi:
                count+=1
                if i in ['','\t','\n','\r']:
                    count-=1
                    continue
                if i[0]=='#': break
                if i[0]=='[':
                    if i[-1]!=']':
                        raise Exception, "Invalid constant format"
                    ival=i[1:-1]
                    ival=float(ival)
                    stacklen+=1
                    if stacklen>TRANSFORM_STACK_LENGTH: raise Exception, "Stack overflow on instruction %s(%d)"%(i,count)
                    lcmdo.append('\x05')
                    lcmdo.append(_struct.pack('f',ival))
                elif i not in _TRANSFORM_TABLE:
                    raise Exception, "Unknown instruction: %s(%d)"%(i,count)
                else:
                    var,cmd,stk,stkupd=_TRANSFORM_TABLE[i]
                    if var>rightvar: rightvar=var
                    if stk>stacklen: raise Exception, "Stack underflow on instruction %s(%d)"%(i,count)
                    stacklen+=stkupd
                    if stacklen>TRANSFORM_STACK_LENGTH: raise Exception, "Stack overflow on instruction %s(%d)"%(i,count)
                    lcmdo.append(cmd)
            cmdo=''.join(lcmdo)
            if len(cmdo)>1024: raise Exception, "Instruction too long"
            _trans_cache[expr]=(cmdo, rightvar, stacklen)
        cmdo, rightvar, stacklen = _trans_cache[expr]
    if stacklen<2: raise Exception, "Insufficient return stack size"
    if rightvar>r: raise Exception, "Insufficient variable, needs %d, given %d."%(rightvar,r)

    _transform2_kern(a, cmdo, b, c, d, result, result2)

_TRANSFORM_TABLE = {
#       Variable uses, Code translated, Stack min element, Stack offset
    '1': (0, '\x00', 0, 1),
    'a': (1, '\x01', 0, 1),
    'b': (2, '\x02', 0, 1),
    'c': (3, '\x03', 0, 1),
    'd': (4, '\x04', 0, 1),
    'id': (0, '\x06', 0, 1),
    'add': (0, '\x07', 2, -1),
    'sub': (0, '\x08', 2, -1),
    'neg': (0, '\x09', 1, 0),
    'mul': (0, '\x0a', 2, -1),

    'div': (0, '\x0b', 2, -1),
    'mod': (0, '\x0c', 2, -1),
    'sin': (0, '\x0d', 1, 0),
    'cos': (0, '\x0e', 1, 0),
    'tan': (0, '\x0f', 1, 0),
    'cot': (0, '\x10', 1, 0),
    'sec': (0, '\x11', 1, 0),
    'csc': (0, '\x12', 1, 0),
    'inv': (0, '\x13', 1, 0),
    'inc': (0, '\x14', 1, 0),

    'exp': (0, '\x15', 1, 0),
    'log': (0, '\x16', 1, 0),
    'sinh': (0, '\x17', 1, 0),
    'cosh': (0, '\x18', 1, 0),
    'tanh': (0, '\x19', 1, 0),
    'coth': (0, '\x1a', 1, 0),
    'sech': (0, '\x1b', 1, 0),
    'csch': (0, '\x1c', 1, 0),
    'sqr': (0, '\x1d', 1, 0),
    'sqrt': (0, '\x1e', 1, 0),

    'pow': (0, '\x1f', 2, -1),
    'asin': (0, '\x20', 1, 0),
    'acos': (0, '\x21', 1, 0),
    'atan': (0, '\x22', 1, 0),
    'fabs': (0, '\x23', 1, 0),
    'ceil': (0, '\x24', 1, 0),
    'floor': (0, '\x25', 1, 0),
    'sigmoid': (0, '\x26', 1, 0),
    'sigmoidgrad': (0, '\x27', 1, 0),
    'dup1': (0, '\x28', 1, 1),
    
    'dup2': (0, '\x29', 2, 1),
    'dup3': (0, '\x2a', 3, 1),
    'dup4': (0, '\x2b', 4, 1),
    '>': (0, '\x2c', 2, -1),
    '<': (0, '\x2d', 2, -1),
    '==': (0, '\x2e', 2, -1),
    '>=': (0, '\x2f', 2, -1),
    '<=': (0, '\x30', 2, -1),
    '!=': (0, '\x31', 2, -1),
    '===': (0, '\x32', 2, -1),

    'and': (0, '\x33', 2, -1),
    'or': (0, '\x34', 2, -1),
    'xor': (0, '\x35', 2, -1),
    'not': (0, '\x36', 1, 0),
    '?:': (0, '\x37', 3, -2),
    'swap': (0, '\x38', 2, 0),
    'isnan': (0, '\x39', 1, 0),
    'isinf': (0, '\x3a', 1, 0),
}


