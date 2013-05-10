/*
CUDANPLEARN Kernel definitions
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
*/

#ifndef _NPLEARN_KERNELS_H
#define _NPLEARN_KERNELS_H

#define SYNC_THREADS 1
#define CUDA_ERROR -1
#define INSUFFICIENT_MEMORY -2
#define INVALID_ARGUMENT -3
#include <stdio.h>
#include "npconstant.cuh"
#define CUDAASSERT(x) {(x); cudaError_t __err = cudaGetLastError(); \
	if (cudaSuccess != __err) { \
	sprintf(errstr,"%s(%d): %d %s(%s)",__FILE__,__LINE__,__err,cudaGetErrorString(__err),custom); \
	return CUDA_ERROR;}}

typedef unsigned int uint32;
typedef unsigned long uint64;

//Skip cuda symbol warning
#ifndef __global__
#define __global__
#define __restrict__
#define __constant__
dim3 __dim3_dummy;
#define blockIdx __dim3_dummy
#define gridDim __dim3_dummy
#define blockDim __dim3_dummy
#define threadIdx __dim3_dummy
#endif

//convolutions
__global__ void kkeepconv4D(const float*devicedata,const float*devicefilters,float*deviceout,uint32 pdo,uint32 blockcount,uint32 dc,uint32 dy,uint32 dx,uint32 fy,uint32 fx);
__global__ void kkeepconv4Dalllayer(const float*devicedata,const float*devicefilters,float*deviceout,uint32 pdo,uint32 blockcount,uint32 dc,uint32 dy,uint32 dx,uint32 fc,uint32 fy,uint32 fx);
__global__ void kgradconvkeep4D(const float*devicedata,const float*devicegrad,float*devicefilters,uint32 pdo,uint32 dy,uint32 dx,uint32 dc,uint32 fc,uint32 fy,uint32 fx);
__global__ void kreverseconvkeep4D(float*devicedata,const float*devicefilters,const float*deviceout,uint32 pdo,uint32 blockcount,uint32 dy,uint32 dx,uint32 dc,uint32 fc,uint32 fy,uint32 fx);
__global__ void kconvolution4D(const float*devicedata,const float*devicefilters,float*deviceout,uint32 pdo,uint32 blockcount,uint32 dy,uint32 dx,uint32 dc,uint32 fc,uint32 fy,uint32 fx,uint32 oy,uint32 ox);
__global__ void kconvolution4D_sm(const float*devicedata,const float*devicefilters,float*deviceout,uint32 pdo,uint32 blockcount,uint32 dy,uint32 dx,uint32 dc,uint32 fc,uint32 fy,uint32 fx,uint32 oy,uint32 ox);
__global__ void kgradconvolution4D(const float*devicedata,const float*devicegrad,float*devicefilters,uint32 pdo,uint32 dy,uint32 dx,uint32 dc,uint32 fc,uint32 fy,uint32 fx,uint32 oy,uint32 ox);
__global__ void kreverseconvolution4D(float*devicedata,const float*devicefilters,const float*deviceout,uint32 pdo,uint32 blockcount,uint32 dy,uint32 dx,uint32 dc,uint32 fc,uint32 fy,uint32 fx,uint32 oy,uint32 ox);
__global__ void kreverseconvolution4D_outorder(float*devicedata,const float*devicefilters,const float*deviceout,uint32 pdo,uint32 blockcount,uint32 dc,uint32 fc,uint32 fy,uint32 fx,uint32 oy,uint32 ox);
__global__ void kreverseconvolution4D_smallfilter(float*devicedata, const float*devicefilters,const float*deviceout,uint32 pdo,uint32 dy,uint32 dx,uint32 dc,uint32 fc,uint32 fy,uint32 fx,uint32 oy,uint32 ox);

__global__ void tohidden_dataparallel(const float* const data, const float* weight, float* out, int const dalign, int const d);
__global__ void fromhidden2_dataparallel(const float* const hidden, const float* const weight, float*out, int dalign, int d);
__global__ void extractvalue_dataparallel(const float* const data, const float* const hidden, float* dweight, int d);
__global__ void pooling_in_dataparallel(const float* const hiddenin, float* poolingout, int d, int wc, int wyo, int wxo, int poolsize);
__global__ void pooling_back_dataparallel(const float* const poolingin, float* hiddenout, int d, int wc, int wyo, int wxo, int poolsize);

__global__ void kinlayermax(const float* const hiddenin, float* poolingout, int d, int dc, int dy, int dx, int poolsize);
__global__ void kreverseinlayermax(const float* const hiddenin, const float* const grad, float* hiddengrad, int d, int dc, int dy, int dx, int poolsize);
__global__ void kalllayermax(const float* const hiddenin, float* poolingout, int d, int dc, int dy, int dx, int poolsize);
__global__ void kreversealllayermax(const float* const hiddenin, const float* const grad, float* hiddengrad, int d, int dc, int dy, int dx, int poolsize);
__global__ void kinlayerabsmax(const float* const hiddenin, float* poolingout, int d, int dc, int dy, int dx, int poolsize);
__global__ void kreverseinlayerabsmax(const float* const hiddenin, const float* const grad, float* hiddengrad, int d, int dc, int dy, int dx, int poolsize);
__global__ void kalllayerabsmax(const float* const hiddenin, float* poolingout, int d, int dc, int dy, int dx, int poolsize);
__global__ void kreversealllayerabsmax(const float* const hiddenin, const float* const grad, float* hiddengrad, int d, int dc, int dy, int dx, int poolsize);

__global__ void kmaxblock2D(const float*devicedata,float*deviceout,uint32 pdo,uint32 blockcount,uint32 dc,uint32 dy,uint32 dx,uint32 oy,uint32 ox,uint32 size);
__global__ void kreversemaxblock2D(const float*devicedata,const float*devicegrad,float*deviceout,const float*deviceoutdata,uint32 pdo,uint32 blockcount,uint32 dc,uint32 dy,uint32 dx,uint32 oy,uint32 ox,uint32 size);
__global__ void kfollowmaxblock2D(const float*devicedata,const float*devicegrad,float*deviceout,uint32 pdo,uint32 blockcount,uint32 dc,uint32 dy,uint32 dx,uint32 oy,uint32 ox,uint32 size);

__global__ void ksquareblock2D(const float*devicedata,float*deviceout,uint32 pdo,uint32 blockcount,uint32 dc,uint32 dy,uint32 dx,uint32 oy,uint32 ox,uint32 size);
__global__ void kreversesquareblock2D(const float*devicedata,const float*devicegrad,float*deviceout,const float*deviceoutdata,uint32 pdo,uint32 blockcount,uint32 dc,uint32 dy,uint32 dx,uint32 oy,uint32 ox,uint32 size);
__global__ void kfollowsquareblock2D(const float*devicedata,const float*devicegrad,float*deviceout,const float*deviceoutdata,uint32 pdo,uint32 blockcount,uint32 dc,uint32 dy,uint32 dx,uint32 oy,uint32 ox,uint32 size);

__global__ void ksquarelayer(const float*devicedata,float*out,uint32 pdo,uint32 blockcount,uint32 oc,uint32 len,uint32 size);
__global__ void kreversesquarelayer(const float*devicedata,const float*devicegrad,float*deviceout,const float*deviceoutdata,uint32 pdo,uint32 blockcount,uint32 oc,uint32 len,uint32 size);
__global__ void kfollowsquarelayer(const float*devicedata,const float*devicegrad,float*deviceout,const float*deviceoutdata,uint32 pdo,uint32 blockcount,uint32 oc,uint32 len,uint32 size);

__global__ void kmaxlayer(const float*devicedata,float*out,uint32 pdo,uint32 blockcount,uint32 oc,uint32 len,uint32 size);
__global__ void kreversemaxlayer(const float*devicedata,const float*devicegrad,float*deviceout,uint32 pdo,uint32 blockcount,uint32 oc,uint32 len,uint32 size);
__global__ void kfollowmaxlayer(const float*devicedata,const float*devicegrad,float*deviceout,uint32 pdo,uint32 blockcount,uint32 oc,uint32 len,uint32 size);

__global__ void kaccumDC(float*data,float*counts,int nd,int count);
__global__ void kdivi(float*data,float val,int count);
__global__ void ksub(float*data,float*sub,int nd,int count);

__global__ void normalize_kern(float*weight, int olen, int ilen);

#define MAX_TRANSFORM 1024
void __set_op_transform(const unsigned char*operates,int len);
__global__ void ktransform(const float*d0,const float*d1,const float*d2,const float*d3,float*o,uint64 len,uint32 oplen,uint64 ps);
__global__ void ktransformD(const double*d0,const double*d1,const double*d2,const double*d3,double*o,uint64 len,uint32 oplen,uint64 ps);
__global__ void ktransform2(const float*d0,const float*d1,const float*d2,const float*d3,float*o,float*o2,uint64 len,uint32 oplen,uint64 ps);

#ifndef _WIN32
#define DLL
#else
#define DLL __declspec(dllexport)
#endif

void setconstant(int cy1, int cx1, int cc1, int wc1, int wyo1, int wxo1, int wyp1, int wxp1);
#define EPSILON 0.00001f
#endif
