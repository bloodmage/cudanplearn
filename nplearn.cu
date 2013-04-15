/*
CUDANPLEARN General kernel support functions
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

#include <stdlib.h>
#include <cuda_runtime.h>
#include "nplearn.cuh"
#include "npconstant.cuh"
#include <string.h>
#include <algorithm>
#include <utility>
#include <stdio.h>

inline bool checkCUDAError() {
    cudaError_t err = cudaGetLastError();
    return cudaSuccess != err;
}

template<typename T>
inline T&min(const T&a,const T&b)
{return a<b?a:b;}


extern "C" {
	
	/*ToHidden Kernel 说明

	计算并行方式：逐数据组并行
	data为: {y坐标(cy), {x坐标(cx), {通道(cc), {数据组(d)}}}} 存储结构,  d为16倍数
	weight为: {输出通道(wc), {输出y坐标(wyo), {输出x坐标(wxo), {输入y坐标(wyp), {输入x坐标(wxp), {输入通道(wop)}}}}}} 存储结构 ( cc==wop )
	out为： {通道(wc), {y坐标(wyo), {x坐标(wxo), {数据组(d)}}}} 存储结构

	实现：
	data分长度为 32*7*n的段，并且按照Kernel需求的方式放置
	weight直接存储于显存中
	逐段处理data，以小于100M的最接近的分解方案进行分段
	 */
	DLL extern int tohidden_kern(float*data, float*weight, float*out, int d, int cy, int cx, int cc, int wc, int wyo, int wxo, int wyp, int wxp) {
		setconstant(cy,cx,cc,wc,wyo,wxo,wyp,wxp);
		int blocksize = cy*cx*cc*sizeof(float);
		int outblocksize = wc*wyo*wxo*sizeof(float);

		int datablock = min(CUDAMemory / blocksize, CUDAMemory / outblocksize);
		if (datablock>d) datablock=d+CUDAAligns-1;
		datablock = datablock/CUDAAligns*CUDAAligns;
		if (datablock>5600) datablock=5600;
		if (datablock==0) return INSUFFICIENT_MEMORY;
		int outsize = datablock*wc*wyo*wxo*sizeof(float);

		//Initial devicedata, deviceweight, deviceout, hostout
		float*hostdata,*devicedata,*deviceweight,*deviceout,*hostout;
		hostout=(float*)malloc(outsize);
		hostdata=(float*)malloc(datablock*blocksize);

		CUDAASSERT(cudaMalloc(&deviceweight,wc*wyo*wxo*wyp*wxp*cc*sizeof(float)));
		CUDAASSERT(cudaMemcpy(deviceweight,weight,wc*wyo*wxo*wyp*wxp*cc*sizeof(float),cudaMemcpyHostToDevice));
		CUDAASSERT(cudaMalloc(&devicedata,datablock*blocksize));
		CUDAASSERT(cudaMalloc(&deviceout,outsize));

		for (unsigned long long pd=0;pd<d;pd+=datablock) {
			//Prepare hostdata
			int pdo=0;
			for (;pdo<datablock;pdo++) if (pdo+pd<d)
				for (int py=0;py<cy;py++)
					for (int px=0;px<cx;px++)
						for (int pc=0;pc<cc;pc++)
							hostdata[((py*cx+px)*cc+pc)*datablock+pdo]=data[(((pd+pdo)*cy+py)*cx+px)*cc+pc];
			else break;

			dim3 gridDim;
			long xval = (datablock*wc*wyo*wxo+CUDALines-1)/CUDALines;
			gridDim.y=(xval+1023)/1024; if (xval>=1024) gridDim.x=1024; else gridDim.x=xval;
			gridDim.z=(gridDim.y+1023)/1024; if (gridDim.y>=1024) gridDim.y=1024;
			CUDAASSERT(cudaMemcpy(devicedata,hostdata,datablock*blocksize,cudaMemcpyHostToDevice));
			tohidden_dataparallel<<<gridDim,CUDALines>>>(devicedata,deviceweight,deviceout,datablock,pdo);
			CUDAASSERT(cudaMemcpy(hostout,deviceout,outsize,cudaMemcpyDeviceToHost));

			for (pdo=0;pdo<datablock;pdo++) if (pdo+pd<d)
				for (int pc=0;pc<wc;pc++)
					for (int py=0;py<wyo;py++)
						for (int px=0;px<wxo;px++)
							out[(((pd+pdo)*wc+pc)*wyo+py)*wxo+px]=hostout[((pc*wyo+py)*wxo+px)*datablock+pdo];
			else break;
		}
		//Release memory associated
		CUDAASSERT(cudaFree(devicedata));
		CUDAASSERT(cudaFree(deviceweight));
		CUDAASSERT(cudaFree(deviceout));
		free(hostout);
		free(hostdata);
		return 0;
	}
	/*FromHidden Kernel 说明

	计算并行方式：逐数据组并行
	hidden为: {通道(wc), {y坐标(wyo), {x坐标(wxo), {数据组(d)}}}} 存储结构,  d为16倍数
	weight为: {输出通道(wc), {输出y坐标(wyo), {输出x坐标(wxo), {输入y坐标(wyp), {输入x坐标(wxp), {输入通道(wop)}}}}}} 存储结构 ( cc==wop )
	out为： {y坐标(cy), {x坐标(cx), {通道(cc), {数据组(d)}}}} 存储结构

	实现：
	hidden分长度为 32*7*n的段，并且按照Kernel需求的方式放置
	weight直接存储于显存中
	逐段处理data，以小于100M的最接近的分解方案进行分段
	 */
	DLL extern int fromhidden_kern(float*hidden, float*weight, float*dataout, int d, int cy, int cx, int cc, int wc, int wyo, int wxo, int wyp, int wxp) {
		setconstant(cy,cx,cc,wc,wyo,wxo,wyp,wxp);
		int blocksize = cy*cx*cc*sizeof(float);
		int outblocksize = wc*wyo*wxo*sizeof(float);

		int datablock = min(CUDAMemory / blocksize, CUDAMemory / outblocksize);
		if (datablock>d) datablock=d+CUDAAligns-1;
		datablock = datablock/CUDAAligns*CUDAAligns;
		if (datablock>5600) datablock=5600;
		if (datablock==0) return INSUFFICIENT_MEMORY;
		int outsize = datablock*wc*wyo*wxo*sizeof(float);

		float*hostdata,*devicedata,*deviceweight,*deviceout,*hostout;
		CUDAASSERT(cudaMalloc(&deviceweight,wc*wyo*wxo*wyp*wxp*cc*sizeof(float)));
		CUDAASSERT(cudaMemcpy(deviceweight,weight,wc*wyo*wxo*wyp*wxp*cc*sizeof(float),cudaMemcpyHostToDevice));
		CUDAASSERT(cudaMalloc(&deviceout,outsize));
		CUDAASSERT(cudaMalloc(&devicedata,datablock*cy*cx*cc*sizeof(float)));
		hostout=(float*)malloc(outsize);
		hostdata=(float*)malloc(datablock*blocksize);

		for (unsigned long long pd=0;pd<d;pd+=datablock) {
			int pdo=0;
			for (;pdo<datablock;pdo++) if (pdo+pd<d)
				for (int pc=0;pc<wc;pc++)
					for (int py=0;py<wyo;py++)
						for (int px=0;px<wxo;px++)
							hostout[((pc*wyo+py)*wxo+px)*datablock+pdo]=hidden[(((pd+pdo)*wc+pc)*wyo+py)*wxo+px];
			else break;

			dim3 gridDim;
			gridDim.x=(datablock*cy*cx*cc+CUDALines-1)/CUDALines;
			gridDim.y=(gridDim.x+1023)/1024; if (gridDim.x>=1024) gridDim.x=1024;
			gridDim.z=(gridDim.y+1023)/1024; if (gridDim.y>=1024) gridDim.y=1024;

			CUDAASSERT(cudaMemcpy(deviceout,hostout,outsize,cudaMemcpyHostToDevice));
			fromhidden2_dataparallel<<<gridDim,CUDALines>>>(deviceout,deviceweight,devicedata,datablock,pdo);
			CUDAASSERT(cudaMemcpy(hostdata,devicedata,datablock*blocksize,cudaMemcpyDeviceToHost));

			for (pdo=0;pdo<datablock;pdo++) if (pdo+pd<d)
				for (int py=0;py<cy;py++)
					for (int px=0;px<cx;px++)
						for (int pc=0;pc<cc;pc++)
							dataout[(((pd+pdo)*cy+py)*cx+px)*cc+pc]=hostdata[((py*cx+px)*cc+pc)*datablock+pdo];
			else break;
		}
		CUDAASSERT(cudaFree(devicedata));
		CUDAASSERT(cudaFree(deviceweight));
		CUDAASSERT(cudaFree(deviceout));
		free(hostout);
		free(hostdata);
		return 0;
	}
	/*ExtractValue Kernel 说明

	计算并行方式：数据组内广泛并行 wxp*wop 32倍数部分位于blockIdx.x，其他坐标位于blockIdx.y和blockIdx.z

	data为： {数据组(d), {y坐标(cy), {x坐标(cx), {通道(cc)}}}} 存储结构
	hidden为: {数据组(d), {通道(wc), {y坐标(wyo), {x坐标(wxo)}}}} 存储结构
	dweight为: {输出通道(wc), {输出y坐标(wyo), {输出x坐标(wxo), {输入y坐标(wyp), {输入x坐标(wxp), {输入通道(wop)}}}}}} 存储结构 ( cc==wop )

	实现：
	按照data拆成100M的段放置于内存中
	weight在显存中累积
	每个thread负责计算weight的一个值
	 */
	DLL extern int extractvalue_kern(float*data, float*hidden, float*weight, int d, int cy, int cx, int cc, int wc, int wyo, int wxo, int wyp, int wxp) {
		setconstant(cy,cx,cc,wc,wyo,wxo,wyp,wxp);
		int blocksize = cy*cx*cc*sizeof(float);
		int outblocksize = wc*wyo*wxo*sizeof(float);

		int datablock = min(CUDAMemory / blocksize, CUDAMemory / outblocksize);
		if (datablock>d) datablock=d;
		if (datablock>5120) datablock=5120;
		if (datablock==0) return INSUFFICIENT_MEMORY;
		int splits = wxo*wxp*cc;
		if (splits>1024) splits=1024;
		
		//Initial devicedata, deviceweight, deviceout, hostout
		float*devicedata,*deviceweight,*devicehidden;
		CUDAASSERT(cudaMalloc(&deviceweight,wc*wyo*wxo*wyp*wxp*cc*sizeof(float)));
		CUDAASSERT(cudaMemset(deviceweight,0,wc*wyo*wxo*wyp*wxp*cc*sizeof(float)));
		CUDAASSERT(cudaMalloc(&devicedata,datablock*blocksize));
		CUDAASSERT(cudaMalloc(&devicehidden,datablock*outblocksize));

		for (unsigned long long pd=0;pd<d;pd+=datablock) {
			int sz=datablock;
			if (pd+datablock>=d) sz=d-pd;

			int totaldim=wc*wyo*wxo*wyp*wxp*cc;
			dim3 gridDim;
			gridDim.x=(totaldim+splits-1)/splits;
			gridDim.y=(gridDim.x+1023)/1024; if (gridDim.x>=1024) gridDim.x=1024;
			gridDim.z=(gridDim.y+1023)/1024; if (gridDim.y>=1024) gridDim.y=1024;

			CUDAASSERT(cudaMemcpy(devicedata,data+pd*cy*cx*cc,sz*blocksize,cudaMemcpyHostToDevice));
			CUDAASSERT(cudaMemcpy(devicehidden,hidden+pd*wc*wyo*wxo,sz*outblocksize,cudaMemcpyHostToDevice));
			extractvalue_dataparallel<<<gridDim,splits>>>(devicedata,devicehidden,deviceweight,sz);
		}
		//Copyout result
		CUDAASSERT(cudaMemcpy(weight,deviceweight,wc*wyo*wxo*wyp*wxp*cc*sizeof(float),cudaMemcpyDeviceToHost));
		//Release memory associated
		CUDAASSERT(cudaFree(devicedata));
		CUDAASSERT(cudaFree(deviceweight));
		CUDAASSERT(cudaFree(devicehidden));
		return 0;
	}
	DLL extern int pooling_in_kern(float*hiddenin, float*poolingout, int d, int wc, int wyo, int wxo, int poolsize) //层内只取有效范围的pooling过程
	{
		int blocksize = wc*wyo*wxo*sizeof(float);
		int cyo=wyo-poolsize+1;
		int cxo=wxo-poolsize+1;
		int poolblocksize = wc*cyo*cxo*sizeof(float);
		
		int datablock = CUDAMemory / blocksize;
		if (datablock>d) datablock=d;
		if (datablock==0) return INSUFFICIENT_MEMORY;
		
		float*devicehidden,*devicepooling;
		CUDAASSERT(cudaMalloc(&devicehidden,datablock*blocksize));
		CUDAASSERT(cudaMalloc(&devicepooling,datablock*poolblocksize));
		
		for (unsigned long long pd=0;pd<d;pd+=datablock) {
			CUDAASSERT(cudaMemcpy(devicehidden,hiddenin+(wc*wyo*wxo*pd),(pd+datablock<=d)?datablock*blocksize:(d-pd)*blocksize,cudaMemcpyHostToDevice));
			pooling_in_dataparallel<<<(cyo*cxo+CUDALines-1)/CUDALines,CUDALines>>>(devicehidden,devicepooling,(pd+datablock<=d)?datablock:(d-pd),wc,wyo,wxo,poolsize);
			CUDAASSERT(cudaMemcpy(poolingout+(wc*cyo*cxo*pd),devicepooling,(pd+datablock<=d)?datablock*poolblocksize:(d-pd)*poolblocksize,cudaMemcpyDeviceToHost));
		}

		CUDAASSERT(cudaFree(devicehidden));
		CUDAASSERT(cudaFree(devicepooling));
		return 0;
	}
	DLL extern int pooling_out_kern(float*poolingin, float*hiddenout, int d, int wc, int wyo, int wxo, int poolsize) //层内向两侧延伸的pooling过程
	{
		int blocksize = wc*wyo*wxo*sizeof(float);
		int cyo=wyo-poolsize+1;
		int cxo=wxo-poolsize+1;
		int poolblocksize = wc*cyo*cxo*sizeof(float);
		
		int datablock = CUDAMemory / blocksize;
		if (datablock>d) datablock=d;
		if (datablock==0) return INSUFFICIENT_MEMORY;
		float*devicehidden,*devicepooling;

		CUDAASSERT(cudaMalloc(&devicehidden,datablock*blocksize));
		CUDAASSERT(cudaMalloc(&devicepooling,datablock*poolblocksize));
		
		for (unsigned long long pd=0;pd<d;pd+=datablock) {
			CUDAASSERT(cudaMemcpy(devicepooling,poolingin+(wc*cyo*cxo*pd),((pd+datablock<=d)?datablock*poolblocksize:(d-pd)*poolblocksize),cudaMemcpyHostToDevice));
			CUDAASSERT(cudaMemset(devicehidden,0,datablock*blocksize));
			pooling_back_dataparallel<<<(cyo*cxo+CUDALines-1)/CUDALines,CUDALines>>>(devicepooling,devicehidden,(pd+datablock<=d)?datablock:(d-pd),wc,wyo,wxo,poolsize);
			CUDAASSERT(cudaMemcpy(hiddenout+(wc*wyo*wxo*pd),devicehidden,((pd+datablock<=d)?datablock*blocksize:(d-pd)*blocksize),cudaMemcpyDeviceToHost));
		}

		CUDAASSERT(cudaFree(devicehidden));
		CUDAASSERT(cudaFree(devicepooling));
		return 0;
	}
	DLL extern void layerpooling_in(float*__restrict hiddenin, float*__restrict poolingout, int d, int wc, int wyo, int wxo, int poolsize)
	{
		int outl=wc/poolsize;
		memset(poolingout,0,d*outl*wyo*wxo*sizeof(float));
		int wn=wyo*wxo;
		for (unsigned long long pd=0;pd<d;pd++)
			for (int pp=0;pp<outl;pp++)
				for (int pp2=0;pp2<poolsize;pp2++)
					for (int pn=0;pn<wn;pn++)
							poolingout[(pd*outl+pp)*wn+pn]+=hiddenin[(pd*wc+pp*poolsize+pp2)*wn+pn];
	}
	DLL extern void layerpooling_out(float*__restrict poolingin, float*__restrict hiddenout, int d, int wc, int wyo, int wxo, int poolsize)
	{
		int outl=wc/poolsize;
		int wn=wyo*wxo;
		for (unsigned long long pd=0;pd<d;pd++)
			for (int pp=0;pp<outl;pp++)
				for (int pp2=0;pp2<poolsize;pp2++)
					for (int pn=0;pn<wn;pn++)
							hiddenout[(pd*wc+pp*poolsize+pp2)*wn+pn]=poolingin[(pd*outl+pp)*wn+pn];
	}

	DLL extern int inlayermax_keepshape(float*hiddenin, float*poolingout, int d, int dc, int dy, int dx, int poolsize)
	{
		int blocksize = dc*dy*dx*sizeof(float);		
		int datablock = CUDAMemory / blocksize;
		if (datablock>d) datablock=d;
		if (datablock==0) return INSUFFICIENT_MEMORY;
		
		float*devicehidden,*devicepooling;
		CUDAASSERT(cudaMalloc(&devicehidden,datablock*blocksize));
		CUDAASSERT(cudaMalloc(&devicepooling,datablock*blocksize));
		
		for (unsigned long long pd=0;pd<d;pd+=datablock) {
			CUDAASSERT(cudaMemcpy(devicehidden,hiddenin+dc*dy*dx*pd,(pd+datablock<=d)?datablock*blocksize:(d-pd)*blocksize,cudaMemcpyHostToDevice));
			kinlayermax<<<(dy*dx+CUDALines-1)/CUDALines,CUDALines>>>(devicehidden,devicepooling,(pd+datablock<=d)?datablock:(d-pd),dc,dy,dx,poolsize);
			CUDAASSERT(cudaMemcpy(poolingout+dc*dy*dx*pd,devicepooling,(pd+datablock<=d)?datablock*blocksize:(d-pd)*blocksize,cudaMemcpyDeviceToHost));
		}

		CUDAASSERT(cudaFree(devicehidden));
		CUDAASSERT(cudaFree(devicepooling));
		return 0;
	}
	//follow version of inlayermax is the same function as reverseinlayermax
	DLL extern int reverseinlayermax_keepshape(float*hiddenin, float*grad, float*outgrad, int d, int dc, int dy, int dx, int poolsize)
	{
		int blocksize = dc*dy*dx*sizeof(float);
		int datablock = CUDAMemory / blocksize;
		if (datablock>d) datablock=d;
		if (datablock==0) return INSUFFICIENT_MEMORY;
		float*devicehidden,*devicegrad,*devicehgrad;

		CUDAASSERT(cudaMalloc(&devicehidden,datablock*blocksize));
		CUDAASSERT(cudaMalloc(&devicegrad,datablock*blocksize));
		CUDAASSERT(cudaMalloc(&devicehgrad,datablock*blocksize));
		
		for (unsigned long long pd=0;pd<d;pd+=datablock) {
			CUDAASSERT(cudaMemcpy(devicehidden,hiddenin+(dc*dy*dx*pd),((pd+datablock<=d)?datablock*blocksize:(d-pd)*blocksize),cudaMemcpyHostToDevice));
			CUDAASSERT(cudaMemcpy(devicegrad,grad+(dc*dy*dx*pd),((pd+datablock<=d)?datablock*blocksize:(d-pd)*blocksize),cudaMemcpyHostToDevice));
			kreverseinlayermax<<<(dy*dx+CUDALines-1)/CUDALines,CUDALines>>>(devicehidden,devicegrad,devicehgrad,(pd+datablock<=d)?datablock:(d-pd),dc,dy,dx,poolsize);
			CUDAASSERT(cudaMemcpy(outgrad+(dc*dy*dx*pd),devicehgrad,((pd+datablock<=d)?datablock*blocksize:(d-pd)*blocksize),cudaMemcpyDeviceToHost));
		}

		CUDAASSERT(cudaFree(devicehidden));
		CUDAASSERT(cudaFree(devicegrad));
		CUDAASSERT(cudaFree(devicehgrad));
		return 0;
	}

	DLL extern int alllayermax_keepshape(float*hiddenin, float*poolingout, int d, int dc, int dy, int dx, int poolsize)
	{
		int blocksize = dc*dy*dx*sizeof(float);		
		int datablock = CUDAMemory / blocksize;
		if (datablock>d) datablock=d;
		if (datablock==0) return INSUFFICIENT_MEMORY;
		
		float*devicehidden,*devicepooling;
		CUDAASSERT(cudaMalloc(&devicehidden,datablock*blocksize));
		CUDAASSERT(cudaMalloc(&devicepooling,datablock*blocksize));
		
		for (unsigned long long pd=0;pd<d;pd+=datablock) {
			CUDAASSERT(cudaMemcpy(devicehidden,hiddenin+dc*dy*dx*pd,(pd+datablock<=d)?datablock*blocksize:(d-pd)*blocksize,cudaMemcpyHostToDevice));
			kalllayermax<<<(dy*dx*dc+CUDALines-1)/CUDALines,CUDALines>>>(devicehidden,devicepooling,(pd+datablock<=d)?datablock:(d-pd),dc,dy,dx,poolsize);
			CUDAASSERT(cudaMemcpy(poolingout+dc*dy*dx*pd,devicepooling,(pd+datablock<=d)?datablock*blocksize:(d-pd)*blocksize,cudaMemcpyDeviceToHost));
		}

		CUDAASSERT(cudaFree(devicehidden));
		CUDAASSERT(cudaFree(devicepooling));
		return 0;
	}

	//follow version of inlayermax is the same function as reverseinlayermax
	DLL extern int reversealllayermax_keepshape(float*hiddenin, float*grad, float*outgrad, int d, int dc, int dy, int dx, int poolsize)
	{
		int blocksize = dc*dy*dx*sizeof(float);
		int datablock = CUDAMemory / blocksize;
		if (datablock>d) datablock=d;
		if (datablock==0) return INSUFFICIENT_MEMORY;
		float*devicehidden,*devicegrad,*devicehgrad;

		CUDAASSERT(cudaMalloc(&devicehidden,datablock*blocksize));
		CUDAASSERT(cudaMalloc(&devicegrad,datablock*blocksize));
		CUDAASSERT(cudaMalloc(&devicehgrad,datablock*blocksize));
		
		for (unsigned long long pd=0;pd<d;pd+=datablock) {
			CUDAASSERT(cudaMemcpy(devicehidden,hiddenin+(dc*dy*dx*pd),((pd+datablock<=d)?datablock*blocksize:(d-pd)*blocksize),cudaMemcpyHostToDevice));
			CUDAASSERT(cudaMemcpy(devicegrad,grad+(dc*dy*dx*pd),((pd+datablock<=d)?datablock*blocksize:(d-pd)*blocksize),cudaMemcpyHostToDevice));
			kreversealllayermax<<<(dy*dx*dc+CUDALines-1)/CUDALines,CUDALines>>>(devicehidden,devicegrad,devicehgrad,(pd+datablock<=d)?datablock:(d-pd),dc,dy,dx,poolsize);
			CUDAASSERT(cudaMemcpy(outgrad+(dc*dy*dx*pd),devicehgrad,((pd+datablock<=d)?datablock*blocksize:(d-pd)*blocksize),cudaMemcpyDeviceToHost));
		}

		CUDAASSERT(cudaFree(devicehidden));
		CUDAASSERT(cudaFree(devicegrad));
		CUDAASSERT(cudaFree(devicehgrad));
		return 0;
	}

	DLL extern int inlayerabsmax_keepshape(float*hiddenin, float*poolingout, int d, int dc, int dy, int dx, int poolsize)
	{
		int blocksize = dc*dy*dx*sizeof(float);		
		int datablock = CUDAMemory / blocksize;
		if (datablock>d) datablock=d;
		if (datablock==0) return INSUFFICIENT_MEMORY;
		
		float*devicehidden,*devicepooling;
		CUDAASSERT(cudaMalloc(&devicehidden,datablock*blocksize));
		CUDAASSERT(cudaMalloc(&devicepooling,datablock*blocksize));
		
		for (unsigned long long pd=0;pd<d;pd+=datablock) {
			CUDAASSERT(cudaMemcpy(devicehidden,hiddenin+dc*dy*dx*pd,(pd+datablock<=d)?datablock*blocksize:(d-pd)*blocksize,cudaMemcpyHostToDevice));
			kinlayerabsmax<<<(dy*dx+CUDALines-1)/CUDALines,CUDALines>>>(devicehidden,devicepooling,(pd+datablock<=d)?datablock:(d-pd),dc,dy,dx,poolsize);
			CUDAASSERT(cudaMemcpy(poolingout+dc*dy*dx*pd,devicepooling,(pd+datablock<=d)?datablock*blocksize:(d-pd)*blocksize,cudaMemcpyDeviceToHost));
		}

		CUDAASSERT(cudaFree(devicehidden));
		CUDAASSERT(cudaFree(devicepooling));
		return 0;
	}
	//follow version of inlayermax is the same function as reverseinlayermax
	DLL extern int reverseinlayerabsmax_keepshape(float*hiddenin, float*grad, float*outgrad, int d, int dc, int dy, int dx, int poolsize)
	{
		int blocksize = dc*dy*dx*sizeof(float);
		int datablock = CUDAMemory / blocksize;
		if (datablock>d) datablock=d;
		if (datablock==0) return INSUFFICIENT_MEMORY;
		float*devicehidden,*devicegrad,*devicehgrad;

		CUDAASSERT(cudaMalloc(&devicehidden,datablock*blocksize));
		CUDAASSERT(cudaMalloc(&devicegrad,datablock*blocksize));
		CUDAASSERT(cudaMalloc(&devicehgrad,datablock*blocksize));
		
		for (unsigned long long pd=0;pd<d;pd+=datablock) {
			CUDAASSERT(cudaMemcpy(devicehidden,hiddenin+(dc*dy*dx*pd),((pd+datablock<=d)?datablock*blocksize:(d-pd)*blocksize),cudaMemcpyHostToDevice));
			CUDAASSERT(cudaMemcpy(devicegrad,grad+(dc*dy*dx*pd),((pd+datablock<=d)?datablock*blocksize:(d-pd)*blocksize),cudaMemcpyHostToDevice));
			kreverseinlayerabsmax<<<(dy*dx+CUDALines-1)/CUDALines,CUDALines>>>(devicehidden,devicegrad,devicehgrad,(pd+datablock<=d)?datablock:(d-pd),dc,dy,dx,poolsize);
			CUDAASSERT(cudaMemcpy(outgrad+(dc*dy*dx*pd),devicehgrad,((pd+datablock<=d)?datablock*blocksize:(d-pd)*blocksize),cudaMemcpyDeviceToHost));
		}

		CUDAASSERT(cudaFree(devicehidden));
		CUDAASSERT(cudaFree(devicegrad));
		CUDAASSERT(cudaFree(devicehgrad));
		return 0;
	}

	DLL extern int alllayerabsmax_keepshape(float*hiddenin, float*poolingout, int d, int dc, int dy, int dx, int poolsize)
	{
		int blocksize = dc*dy*dx*sizeof(float);		
		int datablock = CUDAMemory / blocksize;
		if (datablock>d) datablock=d;
		if (datablock==0) return INSUFFICIENT_MEMORY;
		
		float*devicehidden,*devicepooling;
		CUDAASSERT(cudaMalloc(&devicehidden,datablock*blocksize));
		CUDAASSERT(cudaMalloc(&devicepooling,datablock*blocksize));
		
		for (unsigned long long pd=0;pd<d;pd+=datablock) {
			CUDAASSERT(cudaMemcpy(devicehidden,hiddenin+dc*dy*dx*pd,(pd+datablock<=d)?datablock*blocksize:(d-pd)*blocksize,cudaMemcpyHostToDevice));
			kalllayerabsmax<<<(dy*dx*dc+CUDALines-1)/CUDALines,CUDALines>>>(devicehidden,devicepooling,(pd+datablock<=d)?datablock:(d-pd),dc,dy,dx,poolsize);
			CUDAASSERT(cudaMemcpy(poolingout+dc*dy*dx*pd,devicepooling,(pd+datablock<=d)?datablock*blocksize:(d-pd)*blocksize,cudaMemcpyDeviceToHost));
		}

		CUDAASSERT(cudaFree(devicehidden));
		CUDAASSERT(cudaFree(devicepooling));
		return 0;
	}

	//follow version of inlayermax is the same function as reverseinlayermax
	DLL extern int reversealllayerabsmax_keepshape(float*hiddenin, float*grad, float*outgrad, int d, int dc, int dy, int dx, int poolsize)
	{
		int blocksize = dc*dy*dx*sizeof(float);
		int datablock = CUDAMemory / blocksize;
		if (datablock>d) datablock=d;
		if (datablock==0) return INSUFFICIENT_MEMORY;
		float*devicehidden,*devicegrad,*devicehgrad;

		CUDAASSERT(cudaMalloc(&devicehidden,datablock*blocksize));
		CUDAASSERT(cudaMalloc(&devicegrad,datablock*blocksize));
		CUDAASSERT(cudaMalloc(&devicehgrad,datablock*blocksize));
		
		for (unsigned long long pd=0;pd<d;pd+=datablock) {
			CUDAASSERT(cudaMemcpy(devicehidden,hiddenin+(dc*dy*dx*pd),((pd+datablock<=d)?datablock*blocksize:(d-pd)*blocksize),cudaMemcpyHostToDevice));
			CUDAASSERT(cudaMemcpy(devicegrad,grad+(dc*dy*dx*pd),((pd+datablock<=d)?datablock*blocksize:(d-pd)*blocksize),cudaMemcpyHostToDevice));
			kreversealllayerabsmax<<<(dy*dx*dc+CUDALines-1)/CUDALines,CUDALines>>>(devicehidden,devicegrad,devicehgrad,(pd+datablock<=d)?datablock:(d-pd),dc,dy,dx,poolsize);
			CUDAASSERT(cudaMemcpy(outgrad+(dc*dy*dx*pd),devicehgrad,((pd+datablock<=d)?datablock*blocksize:(d-pd)*blocksize),cudaMemcpyDeviceToHost));
		}

		CUDAASSERT(cudaFree(devicehidden));
		CUDAASSERT(cudaFree(devicegrad));
		CUDAASSERT(cudaFree(devicehgrad));
		return 0;
	}

	/*
	This function does a max block pooling
	data is [nd [dc [dy [dx [float]]]]]
	out is [nd [dc [(dy+size-1)/size [(dx+size-1)/size [float]]]]]
	*/
	DLL extern int maxblock2D(float*data,float*out,int nd,int dc,int dy,int dx,int size)
	{
		int oy=(dy+size-1)/size,ox=(dx+size-1)/size;

		int datablocksize=dc*dy*dx*sizeof(float);
		int outblocksize=dc*oy*ox*sizeof(float);
		int blockcount=CUDAMemory/datablocksize;
		if (blockcount>nd) blockcount=nd+CUDAAligns-1;
		blockcount=blockcount/CUDAAligns*CUDAAligns;
		if (blockcount==0) return INSUFFICIENT_MEMORY;

		float*hostdata,*hostout;
		float*devicedata,*deviceout;
		hostdata=(float*)malloc(blockcount*datablocksize);
		hostout=(float*)malloc(blockcount*outblocksize);
		CUDAASSERT(cudaMalloc(&devicedata,blockcount*datablocksize));
		CUDAASSERT(cudaMalloc(&deviceout,blockcount*outblocksize));
		for (unsigned long long pd=0;pd<nd;pd+=blockcount) {
			int pdo=0;
			for (;pdo<blockcount;pdo++) if (pdo+pd<nd)
				for (int pc=0;pc<dc;pc++)
					for (int py=0;py<dy;py++)
						for (int px=0;px<dx;px++)
							hostdata[((pc*dy+py)*dx+px)*blockcount+pdo]=data[(((pdo+pd)*dc+pc)*dy+py)*dx+px];
			else break;

			unsigned long long xval=(blockcount*outblocksize/sizeof(float)+CUDALines-1)/CUDALines;
			dim3 gridDim;
			gridDim.y=(xval+1023)/1024; if (xval>=1024) gridDim.x=1024; else gridDim.x=xval;
			gridDim.z=(gridDim.y+1023)/1024; if (gridDim.y>=1024) gridDim.y=1024;
			CUDAASSERT(cudaMemcpy(devicedata,hostdata,blockcount*datablocksize,cudaMemcpyHostToDevice));

			kmaxblock2D<<<gridDim,CUDALines>>>(devicedata,deviceout,pdo,blockcount,dc,dy,dx,oy,ox,size);
			CUDAASSERT(cudaMemcpy(hostout,deviceout,blockcount*outblocksize,cudaMemcpyDeviceToHost));

			for (pdo=0;pdo<blockcount;pdo++) if (pdo+pd<nd)
				for (int pc=0;pc<dc;pc++)
					for (int py=0;py<oy;py++)
						for (int px=0;px<ox;px++)
							out[(((pdo+pd)*dc+pc)*oy+py)*ox+px]=hostout[((pc*oy+py)*ox+px)*blockcount+pdo];
			else break;
		}
		CUDAASSERT(cudaFree(devicedata));
		CUDAASSERT(cudaFree(deviceout));
		free(hostdata);
		free(hostout);
		return 0;
	}
	/*
	MaxBlock2D
	This function does a reverse grad passing on max pooling
	*/
	DLL extern int reversemaxblock2D(float*grad,float*data,float*out,int nd,int dc,int dy,int dx,int size)
	{
		int oy=(dy+size-1)/size,ox=(dx+size-1)/size;

		int datablocksize=dc*dy*dx*sizeof(float);
		int outblocksize=dc*oy*ox*sizeof(float);
		int blockcount=CUDAMemory/datablocksize;
		if (blockcount>nd) blockcount=nd+CUDAAligns-1;
		blockcount=blockcount/CUDAAligns*CUDAAligns;
		if (blockcount==0) return INSUFFICIENT_MEMORY;

		float*hostgrad,*hostdata,*hostout;
		float*devicegrad,*devicedata,*deviceout;
		hostgrad=(float*)malloc(blockcount*outblocksize);
		hostdata=(float*)malloc(blockcount*datablocksize);
		hostout=(float*)malloc(blockcount*datablocksize);
		CUDAASSERT(cudaMalloc(&devicegrad,blockcount*outblocksize));
		CUDAASSERT(cudaMalloc(&devicedata,blockcount*datablocksize));
		CUDAASSERT(cudaMalloc(&deviceout,blockcount*datablocksize));

		for (unsigned long long pd=0;pd<nd;pd+=blockcount) {
			int pdo=0;
			for (;pdo<blockcount;pdo++) if (pdo+pd<nd)
			{
				for (int pc=0;pc<dc;pc++)
					for (int py=0;py<dy;py++)
						for (int px=0;px<dx;px++)
							hostdata[((pc*dy+py)*dx+px)*blockcount+pdo]=data[(((pdo+pd)*dc+pc)*dy+py)*dx+px];
				for (int pc=0;pc<dc;pc++)
					for (int py=0;py<oy;py++)
						for (int px=0;px<ox;px++)
							hostgrad[((pc*oy+py)*ox+px)*blockcount+pdo]=grad[(((pdo+pd)*dc+pc)*oy+py)*ox+px];
			}
			else break;

			unsigned long long xval=(blockcount*outblocksize/sizeof(float)+CUDALines-1)/CUDALines;
			dim3 gridDim;
			gridDim.y=(xval+1023)/1024; if (xval>=1024) gridDim.x=1024; else gridDim.x=xval;
			gridDim.z=(gridDim.y+1023)/1024; if (gridDim.y>=1024) gridDim.y=1024;
			CUDAASSERT(cudaMemcpy(devicedata,hostdata,blockcount*datablocksize,cudaMemcpyHostToDevice));
			CUDAASSERT(cudaMemcpy(devicegrad,hostgrad,blockcount*outblocksize,cudaMemcpyHostToDevice));

			kreversemaxblock2D<<<gridDim,CUDALines>>>(devicedata,devicegrad,deviceout,NULL,pdo,blockcount,dc,dy,dx,oy,ox,size);
			CUDAASSERT(cudaMemcpy(hostout,deviceout,blockcount*datablocksize,cudaMemcpyDeviceToHost));

			for (pdo=0;pdo<blockcount;pdo++) if (pdo+pd<nd)
				for (int pc=0;pc<dc;pc++)
					for (int py=0;py<dy;py++)
						for (int px=0;px<dx;px++)
							out[(((pdo+pd)*dc+pc)*dy+py)*dx+px]=hostout[((pc*dy+py)*dx+px)*blockcount+pdo];
			else break;
		}
		CUDAASSERT(cudaFree(devicedata));
		CUDAASSERT(cudaFree(deviceout));
		CUDAASSERT(cudaFree(devicegrad));
		free(hostdata);
		free(hostout);
		free(hostgrad);
		return 0;
	}
	/*
	This function passes weight or grad by following original max-pooling path
	Likes a inverse version of reversemaxblock2D
	*/
	DLL extern int followmaxblock2D(float*grad,float*data,float*out,int nd,int dc,int dy,int dx,int size)
	{
		int oy=(dy+size-1)/size,ox=(dx+size-1)/size;
		
		int datablocksize=dc*dy*dx*sizeof(float);
		int outblocksize=dc*oy*ox*sizeof(float);
		int blockcount=CUDAMemory/datablocksize;
		if (blockcount>nd) blockcount=nd+CUDAAligns-1;
		blockcount=blockcount/CUDAAligns*CUDAAligns;
		if (blockcount==0) return INSUFFICIENT_MEMORY;

		float*hostgrad,*hostdata,*hostout;
		float*devicegrad,*devicedata,*deviceout;
		hostgrad=(float*)malloc(blockcount*datablocksize);
		hostdata=(float*)malloc(blockcount*datablocksize);
		hostout=(float*)malloc(blockcount*outblocksize);
		CUDAASSERT(cudaMalloc(&devicegrad,blockcount*datablocksize));
		CUDAASSERT(cudaMalloc(&devicedata,blockcount*datablocksize));
		CUDAASSERT(cudaMalloc(&deviceout,blockcount*outblocksize));

		for (unsigned long long pd=0;pd<nd;pd+=blockcount) {
			int pdo=0;
			for (;pdo<blockcount;pdo++) if (pdo+pd<nd)
			{
				for (int pc=0;pc<dc;pc++)
					for (int py=0;py<dy;py++)
						for (int px=0;px<dx;px++)
							hostdata[((pc*dy+py)*dx+px)*blockcount+pdo]=data[(((pdo+pd)*dc+pc)*dy+py)*dx+px];
				for (int pc=0;pc<dc;pc++)
					for (int py=0;py<dy;py++)
						for (int px=0;px<dx;px++)
							hostgrad[((pc*dy+py)*dx+px)*blockcount+pdo]=grad[(((pdo+pd)*dc+pc)*dy+py)*dx+px];
			}
			else break;

			unsigned long long xval=(blockcount*outblocksize/sizeof(float)+CUDALines-1)/CUDALines;
			dim3 gridDim;
			gridDim.y=(xval+1023)/1024; if (xval>=1024) gridDim.x=1024; else gridDim.x=xval;
			gridDim.z=(gridDim.y+1023)/1024; if (gridDim.y>=1024) gridDim.y=1024;
			CUDAASSERT(cudaMemcpy(devicedata,hostdata,blockcount*datablocksize,cudaMemcpyHostToDevice));
			CUDAASSERT(cudaMemcpy(devicegrad,hostgrad,blockcount*datablocksize,cudaMemcpyHostToDevice));

			kfollowmaxblock2D<<<gridDim,CUDALines>>>(devicedata,devicegrad,deviceout,pdo,blockcount,dc,dy,dx,oy,ox,size);
			CUDAASSERT(cudaMemcpy(hostout,deviceout,blockcount*outblocksize,cudaMemcpyDeviceToHost));

			for (pdo=0;pdo<blockcount;pdo++) if (pdo+pd<nd)
				for (int pc=0;pc<dc;pc++)
					for (int py=0;py<oy;py++)
						for (int px=0;px<ox;px++)
							out[(((pdo+pd)*dc+pc)*oy+py)*ox+px]=hostout[((pc*oy+py)*ox+px)*blockcount+pdo];
			else break;
		}
		CUDAASSERT(cudaFree(devicedata));
		CUDAASSERT(cudaFree(deviceout));
		CUDAASSERT(cudaFree(devicegrad));
		free(hostdata);
		free(hostout);
		free(hostgrad);
		return 0;
	}
	/*
	This function does a square pooling
	*/
	DLL extern int squareblock2D(float*data,float*out,int nd,int dc,int dy,int dx,int size)
	{
		int oy=(dy+size-1)/size,ox=(dx+size-1)/size;

		int datablocksize=dc*dy*dx*sizeof(float);
		int outblocksize=dc*oy*ox*sizeof(float);
		int blockcount=CUDAMemory/datablocksize;
		if (blockcount>nd) blockcount=nd+CUDAAligns-1;
		blockcount=blockcount/CUDAAligns*CUDAAligns;
		if (blockcount==0) return INSUFFICIENT_MEMORY;

		float*hostdata,*hostout;
		float*devicedata,*deviceout;
		hostdata=(float*)malloc(blockcount*datablocksize);
		hostout=(float*)malloc(blockcount*outblocksize);
		CUDAASSERT(cudaMalloc(&devicedata,blockcount*datablocksize));
		CUDAASSERT(cudaMalloc(&deviceout,blockcount*outblocksize));
		for (unsigned long long pd=0;pd<nd;pd+=blockcount) {
			int pdo=0;
			for (;pdo<blockcount;pdo++) if (pdo+pd<nd)
				for (int pc=0;pc<dc;pc++)
					for (int py=0;py<dy;py++)
						for (int px=0;px<dx;px++)
							hostdata[((pc*dy+py)*dx+px)*blockcount+pdo]=data[(((pdo+pd)*dc+pc)*dy+py)*dx+px];
			else break;

			unsigned long long xval=(blockcount*outblocksize/sizeof(float)+CUDALines-1)/CUDALines;
			dim3 gridDim;
			gridDim.y=(xval+1023)/1024; if (xval>=1024) gridDim.x=1024; else gridDim.x=xval;
			gridDim.z=(gridDim.y+1023)/1024; if (gridDim.y>=1024) gridDim.y=1024;
			CUDAASSERT(cudaMemcpy(devicedata,hostdata,blockcount*datablocksize,cudaMemcpyHostToDevice));

			ksquareblock2D<<<gridDim,CUDALines>>>(devicedata,deviceout,pdo,blockcount,dc,dy,dx,oy,ox,size);
			CUDAASSERT(cudaMemcpy(hostout,deviceout,blockcount*outblocksize,cudaMemcpyDeviceToHost));

			for (pdo=0;pdo<blockcount;pdo++) if (pdo+pd<nd)
				for (int pc=0;pc<dc;pc++)
					for (int py=0;py<oy;py++)
						for (int px=0;px<ox;px++)
							out[(((pdo+pd)*dc+pc)*oy+py)*ox+px]=hostout[((pc*oy+py)*ox+px)*blockcount+pdo];
			else break;
		}
		CUDAASSERT(cudaFree(devicedata));
		CUDAASSERT(cudaFree(deviceout));
		free(hostdata);
		free(hostout);
		return 0;
	}
	/*
	This function does a reverse grad passing on square pooling
	*/
	DLL extern int reversesquareblock2D(float*grad,float*data,float*outdata,float*out,int nd,int dc,int dy,int dx,int size)
	{
		int oy=(dy+size-1)/size,ox=(dx+size-1)/size;

		int datablocksize=dc*dy*dx*sizeof(float);
		int outblocksize=dc*oy*ox*sizeof(float);
		int blockcount=CUDAMemory/datablocksize;
		if (blockcount>nd) blockcount=nd+CUDAAligns-1;
		blockcount=blockcount/CUDAAligns*CUDAAligns;
		if (blockcount==0) return INSUFFICIENT_MEMORY;

		float*hostgrad,*hostdata,*hostoutdata,*hostout;
		float*devicegrad,*devicedata,*deviceoutdata,*deviceout;
		hostgrad=(float*)malloc(blockcount*outblocksize);
		hostdata=(float*)malloc(blockcount*datablocksize);
		hostout=(float*)malloc(blockcount*datablocksize);
		hostoutdata=(float*)malloc(blockcount*outblocksize);
		CUDAASSERT(cudaMalloc(&devicegrad,blockcount*outblocksize));
		CUDAASSERT(cudaMalloc(&devicedata,blockcount*datablocksize));
		CUDAASSERT(cudaMalloc(&deviceout,blockcount*datablocksize));
		CUDAASSERT(cudaMalloc(&deviceoutdata,blockcount*outblocksize));

		for (unsigned long long pd=0;pd<nd;pd+=blockcount) {
			int pdo=0;
			for (;pdo<blockcount;pdo++) if (pdo+pd<nd)
			{
				for (int pc=0;pc<dc;pc++)
					for (int py=0;py<dy;py++)
						for (int px=0;px<dx;px++)
							hostdata[((pc*dy+py)*dx+px)*blockcount+pdo]=data[(((pdo+pd)*dc+pc)*dy+py)*dx+px];
				for (int pc=0;pc<dc;pc++)
					for (int py=0;py<oy;py++)
						for (int px=0;px<ox;px++)
							hostgrad[((pc*oy+py)*ox+px)*blockcount+pdo]=grad[(((pdo+pd)*dc+pc)*oy+py)*ox+px];
				for (int pc=0;pc<dc;pc++)
					for (int py=0;py<oy;py++)
						for (int px=0;px<ox;px++)
							hostoutdata[((pc*oy+py)*ox+px)*blockcount+pdo]=outdata[(((pdo+pd)*dc+pc)*oy+py)*ox+px];
			}
			else break;

			unsigned long long xval=(blockcount*outblocksize/sizeof(float)+CUDALines-1)/CUDALines;
			dim3 gridDim;
			gridDim.y=(xval+1023)/1024; if (xval>=1024) gridDim.x=1024; else gridDim.x=xval;
			gridDim.z=(gridDim.y+1023)/1024; if (gridDim.y>=1024) gridDim.y=1024;
			CUDAASSERT(cudaMemcpy(devicedata,hostdata,blockcount*datablocksize,cudaMemcpyHostToDevice));
			CUDAASSERT(cudaMemcpy(devicegrad,hostgrad,blockcount*outblocksize,cudaMemcpyHostToDevice));
			CUDAASSERT(cudaMemcpy(deviceoutdata,hostoutdata,blockcount*outblocksize,cudaMemcpyHostToDevice));

			kreversesquareblock2D<<<gridDim,CUDALines>>>(devicedata,devicegrad,deviceout,deviceoutdata,pdo,blockcount,dc,dy,dx,oy,ox,size);
			CUDAASSERT(cudaMemcpy(hostout,deviceout,blockcount*datablocksize,cudaMemcpyDeviceToHost));

			for (pdo=0;pdo<blockcount;pdo++) if (pdo+pd<nd)
				for (int pc=0;pc<dc;pc++)
					for (int py=0;py<dy;py++)
						for (int px=0;px<dx;px++)
							out[(((pdo+pd)*dc+pc)*dy+py)*dx+px]=hostout[((pc*dy+py)*dx+px)*blockcount+pdo];
			else break;
		}
		CUDAASSERT(cudaFree(devicedata));
		CUDAASSERT(cudaFree(deviceout));
		CUDAASSERT(cudaFree(devicegrad));
		CUDAASSERT(cudaFree(deviceoutdata));
		free(hostdata);
		free(hostout);
		free(hostgrad);
		free(hostoutdata);
		return 0;
	}
	/*
	This function passes weight or grad by following original square-pooling path
	Likes a inverse version of reversesquareblock2D
	*/
	DLL extern int followsquareblock2D(float*grad,float*data,float*outdata,float*out,int nd,int dc,int dy,int dx,int size)
	{
		int oy=(dy+size-1)/size,ox=(dx+size-1)/size;

		int datablocksize=dc*dy*dx*sizeof(float);
		int outblocksize=dc*oy*ox*sizeof(float);
		int blockcount=CUDAMemory/datablocksize;
		if (blockcount>nd) blockcount=nd+CUDAAligns-1;
		blockcount=blockcount/CUDAAligns*CUDAAligns;
		if (blockcount==0) return INSUFFICIENT_MEMORY;

		float*hostgrad,*hostdata,*hostoutdata,*hostout;
		float*devicegrad,*devicedata,*deviceoutdata,*deviceout;
		hostgrad=(float*)malloc(blockcount*datablocksize);
		hostdata=(float*)malloc(blockcount*datablocksize);
		hostout=(float*)malloc(blockcount*outblocksize);
		hostoutdata=(float*)malloc(blockcount*outblocksize);
		CUDAASSERT(cudaMalloc(&devicegrad,blockcount*datablocksize));
		CUDAASSERT(cudaMalloc(&devicedata,blockcount*datablocksize));
		CUDAASSERT(cudaMalloc(&deviceout,blockcount*outblocksize));
		CUDAASSERT(cudaMalloc(&deviceoutdata,blockcount*outblocksize));

		for (unsigned long long pd=0;pd<nd;pd+=blockcount) {
			int pdo=0;
			for (;pdo<blockcount;pdo++) if (pdo+pd<nd)
			{
				for (int pc=0;pc<dc;pc++)
					for (int py=0;py<dy;py++)
						for (int px=0;px<dx;px++)
							hostdata[((pc*dy+py)*dx+px)*blockcount+pdo]=data[(((pdo+pd)*dc+pc)*dy+py)*dx+px];
				for (int pc=0;pc<dc;pc++)
					for (int py=0;py<dy;py++)
						for (int px=0;px<dx;px++)
							hostgrad[((pc*dy+py)*dx+px)*blockcount+pdo]=grad[(((pdo+pd)*dc+pc)*dy+py)*dx+px];
				for (int pc=0;pc<dc;pc++)
					for (int py=0;py<oy;py++)
						for (int px=0;px<ox;px++)
							hostoutdata[((pc*oy+py)*ox+px)*blockcount+pdo]=outdata[(((pdo+pd)*dc+pc)*oy+py)*ox+px];
			}
			else break;

			unsigned long long xval=(blockcount*outblocksize/sizeof(float)+CUDALines-1)/CUDALines;
			dim3 gridDim;
			gridDim.y=(xval+1023)/1024; if (xval>=1024) gridDim.x=1024; else gridDim.x=xval;
			gridDim.z=(gridDim.y+1023)/1024; if (gridDim.y>=1024) gridDim.y=1024;
			CUDAASSERT(cudaMemcpy(devicedata,hostdata,blockcount*datablocksize,cudaMemcpyHostToDevice));
			CUDAASSERT(cudaMemcpy(devicegrad,hostgrad,blockcount*datablocksize,cudaMemcpyHostToDevice));
			CUDAASSERT(cudaMemcpy(deviceoutdata,hostoutdata,blockcount*outblocksize,cudaMemcpyHostToDevice));

			kfollowsquareblock2D<<<gridDim,CUDALines>>>(devicedata,devicegrad,deviceout,deviceoutdata,pdo,blockcount,dc,dy,dx,oy,ox,size);
			CUDAASSERT(cudaMemcpy(hostout,deviceout,blockcount*outblocksize,cudaMemcpyDeviceToHost));

			for (pdo=0;pdo<blockcount;pdo++) if (pdo+pd<nd)
				for (int pc=0;pc<dc;pc++)
					for (int py=0;py<oy;py++)
						for (int px=0;px<ox;px++)
							out[(((pdo+pd)*dc+pc)*oy+py)*ox+px]=hostout[((pc*oy+py)*ox+px)*blockcount+pdo];
			else break;
		}
		CUDAASSERT(cudaFree(devicedata));
		CUDAASSERT(cudaFree(deviceout));
		CUDAASSERT(cudaFree(devicegrad));
		CUDAASSERT(cudaFree(deviceoutdata));
		free(hostdata);
		free(hostout);
		free(hostgrad);
		free(hostoutdata);
		return 0;
	}
	/*
	This function does a cross-layer square pooling
	*/
	DLL extern int squarelayer(float*data,float*out,int nd,int dc,int dy,int dx,int size)
	{
		int oc=dc/size;

		int datablocksize=dc*dy*dx*sizeof(float);
		int outblocksize=oc*dy*dx*sizeof(float);
		int blockcount=CUDAMemory/datablocksize;
		if (blockcount>nd) blockcount=nd+CUDAAligns-1;
		blockcount=blockcount/CUDAAligns*CUDAAligns;
		if (blockcount==0) return INSUFFICIENT_MEMORY;

		float*hostdata,*hostout;
		float*devicedata,*deviceout;
		hostdata=(float*)malloc(blockcount*datablocksize);
		hostout=(float*)malloc(blockcount*outblocksize);
		CUDAASSERT(cudaMalloc(&devicedata,blockcount*datablocksize));
		CUDAASSERT(cudaMalloc(&deviceout,blockcount*outblocksize));
		for (unsigned long long pd=0;pd<nd;pd+=blockcount) {
			int pdo=0;
			for (;pdo<blockcount;pdo++) if (pdo+pd<nd)
				for (int pc=0;pc<dc;pc++)
					for (int py=0;py<dy;py++)
						for (int px=0;px<dx;px++)
							hostdata[((pc*dy+py)*dx+px)*blockcount+pdo]=data[(((pdo+pd)*dc+pc)*dy+py)*dx+px];
			else break;

			unsigned long long xval=(blockcount*outblocksize/sizeof(float)+CUDALines-1)/CUDALines;
			dim3 gridDim;
			gridDim.y=(xval+1023)/1024; if (xval>=1024) gridDim.x=1024; else gridDim.x=xval;
			gridDim.z=(gridDim.y+1023)/1024; if (gridDim.y>=1024) gridDim.y=1024;
			CUDAASSERT(cudaMemcpy(devicedata,hostdata,blockcount*datablocksize,cudaMemcpyHostToDevice));

			ksquarelayer<<<gridDim,CUDALines>>>(devicedata,deviceout,pdo,blockcount,oc,dy*dx,size);
			CUDAASSERT(cudaMemcpy(hostout,deviceout,blockcount*outblocksize,cudaMemcpyDeviceToHost));

			for (pdo=0;pdo<blockcount;pdo++) if (pdo+pd<nd)
				for (int pc=0;pc<oc;pc++)
					for (int py=0;py<dy;py++)
						for (int px=0;px<dx;px++)
							out[(((pdo+pd)*oc+pc)*dy+py)*dx+px]=hostout[((pc*dy+py)*dx+px)*blockcount+pdo];
			else break;
		}
		CUDAASSERT(cudaFree(devicedata));
		CUDAASSERT(cudaFree(deviceout));
		free(hostdata);
		free(hostout);
		return 0;
	}
	/*
	This function does a grad passing on a cross-layer square pooling
	*/
	DLL extern int reversesquarelayer(float*grad,float*data,float*outdata,float*out,int nd,int dc,int dy,int dx,int size)
	{
		int oc=dc/size;

		int datablocksize=dc*dy*dx*sizeof(float);
		int outblocksize=oc*dy*dx*sizeof(float);
		int blockcount=CUDAMemory/datablocksize;
		if (blockcount>nd) blockcount=nd+CUDAAligns-1;
		blockcount=blockcount/CUDAAligns*CUDAAligns;
		if (blockcount==0) return INSUFFICIENT_MEMORY;

		float*hostgrad,*hostdata,*hostoutdata,*hostout;
		float*devicegrad,*devicedata,*deviceoutdata,*deviceout;
		hostgrad=(float*)malloc(blockcount*outblocksize);
		hostdata=(float*)malloc(blockcount*datablocksize);
		hostout=(float*)malloc(blockcount*datablocksize);
		hostoutdata=(float*)malloc(blockcount*outblocksize);
		CUDAASSERT(cudaMalloc(&devicegrad,blockcount*outblocksize));
		CUDAASSERT(cudaMalloc(&devicedata,blockcount*datablocksize));
		CUDAASSERT(cudaMalloc(&deviceout,blockcount*datablocksize));
		CUDAASSERT(cudaMalloc(&deviceoutdata,blockcount*outblocksize));

		for (unsigned long long pd=0;pd<nd;pd+=blockcount) {
			int pdo=0;
			for (;pdo<blockcount;pdo++) if (pdo+pd<nd)
			{
				for (int pc=0;pc<dc;pc++)
					for (int py=0;py<dy;py++)
						for (int px=0;px<dx;px++)
							hostdata[((pc*dy+py)*dx+px)*blockcount+pdo]=data[(((pdo+pd)*dc+pc)*dy+py)*dx+px];
				for (int pc=0;pc<oc;pc++)
					for (int py=0;py<dy;py++)
						for (int px=0;px<dx;px++)
							hostgrad[((pc*dy+py)*dx+px)*blockcount+pdo]=grad[(((pdo+pd)*oc+pc)*dy+py)*dx+px];
				for (int pc=0;pc<oc;pc++)
					for (int py=0;py<dy;py++)
						for (int px=0;px<dx;px++)
							hostoutdata[((pc*dy+py)*dx+px)*blockcount+pdo]=outdata[(((pdo+pd)*oc+pc)*dy+py)*dx+px];
			}
			else break;

			unsigned long long xval=(blockcount*outblocksize/sizeof(float)+CUDALines-1)/CUDALines;
			dim3 gridDim;
			gridDim.y=(xval+1023)/1024; if (xval>=1024) gridDim.x=1024; else gridDim.x=xval;
			gridDim.z=(gridDim.y+1023)/1024; if (gridDim.y>=1024) gridDim.y=1024;
			CUDAASSERT(cudaMemcpy(devicedata,hostdata,blockcount*datablocksize,cudaMemcpyHostToDevice));
			CUDAASSERT(cudaMemcpy(devicegrad,hostgrad,blockcount*outblocksize,cudaMemcpyHostToDevice));
			CUDAASSERT(cudaMemcpy(deviceoutdata,hostoutdata,blockcount*outblocksize,cudaMemcpyHostToDevice));

			kreversesquarelayer<<<gridDim,CUDALines>>>(devicedata,devicegrad,deviceout,deviceoutdata,pdo,blockcount,oc,dy*dx,size);
			CUDAASSERT(cudaMemcpy(hostout,deviceout,blockcount*datablocksize,cudaMemcpyDeviceToHost));

			for (pdo=0;pdo<blockcount;pdo++) if (pdo+pd<nd)
				for (int pc=0;pc<dc;pc++)
					for (int py=0;py<dy;py++)
						for (int px=0;px<dx;px++)
							out[(((pdo+pd)*dc+pc)*dy+py)*dx+px]=hostout[((pc*dy+py)*dx+px)*blockcount+pdo];
			else break;
		}
		CUDAASSERT(cudaFree(devicedata));
		CUDAASSERT(cudaFree(deviceout));
		CUDAASSERT(cudaFree(devicegrad));
		CUDAASSERT(cudaFree(deviceoutdata));
		free(hostdata);
		free(hostout);
		free(hostgrad);
		free(hostoutdata);
		return 0;
	}
	/*
	This function passes weight or grad by following original cross-layer square pooling path
	Likes a inverse version of reversesquarelayer
	*/
	DLL extern int followsquarelayer(float*grad,float*data,float*outdata,float*out,int nd,int dc,int dy,int dx,int size)
	{
		int oc=dc/size;

		int datablocksize=dc*dy*dx*sizeof(float);
		int outblocksize=oc*dy*dx*sizeof(float);
		int blockcount=CUDAMemory/datablocksize;
		if (blockcount>nd) blockcount=nd+CUDAAligns-1;
		blockcount=blockcount/CUDAAligns*CUDAAligns;
		if (blockcount==0) return INSUFFICIENT_MEMORY;

		float*hostgrad,*hostdata,*hostoutdata,*hostout;
		float*devicegrad,*devicedata,*deviceoutdata,*deviceout;
		hostgrad=(float*)malloc(blockcount*datablocksize);
		hostdata=(float*)malloc(blockcount*datablocksize);
		hostout=(float*)malloc(blockcount*outblocksize);
		hostoutdata=(float*)malloc(blockcount*outblocksize);
		CUDAASSERT(cudaMalloc(&devicegrad,blockcount*datablocksize));
		CUDAASSERT(cudaMalloc(&devicedata,blockcount*datablocksize));
		CUDAASSERT(cudaMalloc(&deviceout,blockcount*outblocksize));
		CUDAASSERT(cudaMalloc(&deviceoutdata,blockcount*outblocksize));

		for (unsigned long long pd=0;pd<nd;pd+=blockcount) {
			int pdo=0;
			for (;pdo<blockcount;pdo++) if (pdo+pd<nd)
			{
				for (int pc=0;pc<dc;pc++)
					for (int py=0;py<dy;py++)
						for (int px=0;px<dx;px++)
							hostdata[((pc*dy+py)*dx+px)*blockcount+pdo]=data[(((pdo+pd)*dc+pc)*dy+py)*dx+px];
				for (int pc=0;pc<dc;pc++)
					for (int py=0;py<dy;py++)
						for (int px=0;px<dx;px++)
							hostgrad[((pc*dy+py)*dx+px)*blockcount+pdo]=grad[(((pdo+pd)*dc+pc)*dy+py)*dx+px];
				for (int pc=0;pc<oc;pc++)
					for (int py=0;py<dy;py++)
						for (int px=0;px<dx;px++)
							hostoutdata[((pc*dy+py)*dx+px)*blockcount+pdo]=outdata[(((pdo+pd)*oc+pc)*dy+py)*dx+px];
			}
			else break;

			unsigned long long xval=(blockcount*outblocksize/sizeof(float)+CUDALines-1)/CUDALines;
			dim3 gridDim;
			gridDim.y=(xval+1023)/1024; if (xval>=1024) gridDim.x=1024; else gridDim.x=xval;
			gridDim.z=(gridDim.y+1023)/1024; if (gridDim.y>=1024) gridDim.y=1024;
			CUDAASSERT(cudaMemcpy(devicedata,hostdata,blockcount*datablocksize,cudaMemcpyHostToDevice));
			CUDAASSERT(cudaMemcpy(devicegrad,hostgrad,blockcount*datablocksize,cudaMemcpyHostToDevice));
			CUDAASSERT(cudaMemcpy(deviceoutdata,hostoutdata,blockcount*outblocksize,cudaMemcpyHostToDevice));

			kfollowsquarelayer<<<gridDim,CUDALines>>>(devicedata,devicegrad,deviceout,deviceoutdata,pdo,blockcount,oc,dy*dx,size);
			CUDAASSERT(cudaMemcpy(hostout,deviceout,blockcount*outblocksize,cudaMemcpyDeviceToHost));

			for (pdo=0;pdo<blockcount;pdo++) if (pdo+pd<nd)
				for (int pc=0;pc<oc;pc++)
					for (int py=0;py<dy;py++)
						for (int px=0;px<dx;px++)
							out[(((pdo+pd)*oc+pc)*dy+py)*dx+px]=hostout[((pc*dy+py)*dx+px)*blockcount+pdo];
			else break;
		}
		CUDAASSERT(cudaFree(devicedata));
		CUDAASSERT(cudaFree(deviceout));
		CUDAASSERT(cudaFree(devicegrad));
		CUDAASSERT(cudaFree(deviceoutdata));
		free(hostdata);
		free(hostout);
		free(hostgrad);
		free(hostoutdata);
		return 0;
	}
	/*
	This function does a cross-layer max pooling
	*/
	DLL extern int maxlayer(float*data,float*out,int nd,int dc,int dy,int dx,int size)
	{
		int oc=dc/size;

		int datablocksize=dc*dy*dx*sizeof(float);
		int outblocksize=oc*dy*dx*sizeof(float);
		int blockcount=CUDAMemory/datablocksize;
		if (blockcount>nd) blockcount=nd+CUDAAligns-1;
		blockcount=blockcount/CUDAAligns*CUDAAligns;
		if (blockcount==0) return INSUFFICIENT_MEMORY;

		float*hostdata,*hostout;
		float*devicedata,*deviceout;
		hostdata=(float*)malloc(blockcount*datablocksize);
		hostout=(float*)malloc(blockcount*outblocksize);
		CUDAASSERT(cudaMalloc(&devicedata,blockcount*datablocksize));
		CUDAASSERT(cudaMalloc(&deviceout,blockcount*outblocksize));
		for (unsigned long long pd=0;pd<nd;pd+=blockcount) {
			int pdo=0;
			for (;pdo<blockcount;pdo++) if (pdo+pd<nd)
				for (int pc=0;pc<dc;pc++)
					for (int py=0;py<dy;py++)
						for (int px=0;px<dx;px++)
							hostdata[((pc*dy+py)*dx+px)*blockcount+pdo]=data[(((pdo+pd)*dc+pc)*dy+py)*dx+px];
			else break;

			unsigned long long xval=(blockcount*outblocksize/sizeof(float)+CUDALines-1)/CUDALines;
			dim3 gridDim;
			gridDim.y=(xval+1023)/1024; if (xval>=1024) gridDim.x=1024; else gridDim.x=xval;
			gridDim.z=(gridDim.y+1023)/1024; if (gridDim.y>=1024) gridDim.y=1024;
			CUDAASSERT(cudaMemcpy(devicedata,hostdata,blockcount*datablocksize,cudaMemcpyHostToDevice));

			kmaxlayer<<<gridDim,CUDALines>>>(devicedata,deviceout,pdo,blockcount,oc,dy*dx,size);
			CUDAASSERT(cudaMemcpy(hostout,deviceout,blockcount*outblocksize,cudaMemcpyDeviceToHost));

			for (pdo=0;pdo<blockcount;pdo++) if (pdo+pd<nd)
				for (int pc=0;pc<oc;pc++)
					for (int py=0;py<dy;py++)
						for (int px=0;px<dx;px++)
							out[(((pdo+pd)*oc+pc)*dy+py)*dx+px]=hostout[((pc*dy+py)*dx+px)*blockcount+pdo];
			else break;
		}
		CUDAASSERT(cudaFree(devicedata));
		CUDAASSERT(cudaFree(deviceout));
		free(hostdata);
		free(hostout);
		return 0;
	}
	/*
	This function does a grad passing on a cross-layer max pooling
	*/
	DLL extern int reversemaxlayer(float*grad,float*data,float*out,int nd,int dc,int dy,int dx,int size)
	{
		int oc=dc/size;

		int datablocksize=dc*dy*dx*sizeof(float);
		int outblocksize=oc*dy*dx*sizeof(float);
		int blockcount=CUDAMemory/datablocksize;
		if (blockcount>nd) blockcount=nd+CUDAAligns-1;
		blockcount=blockcount/CUDAAligns*CUDAAligns;
		if (blockcount==0) return INSUFFICIENT_MEMORY;

		float*hostgrad,*hostdata,*hostout;
		float*devicegrad,*devicedata,*deviceout;
		hostgrad=(float*)malloc(blockcount*outblocksize);
		hostdata=(float*)malloc(blockcount*datablocksize);
		hostout=(float*)malloc(blockcount*datablocksize);
		CUDAASSERT(cudaMalloc(&devicegrad,blockcount*outblocksize));
		CUDAASSERT(cudaMalloc(&devicedata,blockcount*datablocksize));
		CUDAASSERT(cudaMalloc(&deviceout,blockcount*datablocksize));

		for (unsigned long long pd=0;pd<nd;pd+=blockcount) {
			int pdo=0;
			for (;pdo<blockcount;pdo++) if (pdo+pd<nd)
			{
				for (int pc=0;pc<dc;pc++)
					for (int py=0;py<dy;py++)
						for (int px=0;px<dx;px++)
							hostdata[((pc*dy+py)*dx+px)*blockcount+pdo]=data[(((pdo+pd)*dc+pc)*dy+py)*dx+px];
				for (int pc=0;pc<oc;pc++)
					for (int py=0;py<dy;py++)
						for (int px=0;px<dx;px++)
							hostgrad[((pc*dy+py)*dx+px)*blockcount+pdo]=grad[(((pdo+pd)*oc+pc)*dy+py)*dx+px];
			}
			else break;

			unsigned long long xval=(blockcount*outblocksize/sizeof(float)+CUDALines-1)/CUDALines;
			dim3 gridDim;
			gridDim.y=(xval+1023)/1024; if (xval>=1024) gridDim.x=1024; else gridDim.x=xval;
			gridDim.z=(gridDim.y+1023)/1024; if (gridDim.y>=1024) gridDim.y=1024;
			CUDAASSERT(cudaMemcpy(devicedata,hostdata,blockcount*datablocksize,cudaMemcpyHostToDevice));
			CUDAASSERT(cudaMemcpy(devicegrad,hostgrad,blockcount*outblocksize,cudaMemcpyHostToDevice));

			kreversemaxlayer<<<gridDim,CUDALines>>>(devicedata,devicegrad,deviceout,pdo,blockcount,oc,dy*dx,size);
			CUDAASSERT(cudaMemcpy(hostout,deviceout,blockcount*datablocksize,cudaMemcpyDeviceToHost));

			for (pdo=0;pdo<blockcount;pdo++) if (pdo+pd<nd)
				for (int pc=0;pc<dc;pc++)
					for (int py=0;py<dy;py++)
						for (int px=0;px<dx;px++)
							out[(((pdo+pd)*dc+pc)*dy+py)*dx+px]=hostout[((pc*dy+py)*dx+px)*blockcount+pdo];
			else break;
		}
		CUDAASSERT(cudaFree(devicedata));
		CUDAASSERT(cudaFree(deviceout));
		CUDAASSERT(cudaFree(devicegrad));
		free(hostdata);
		free(hostout);
		free(hostgrad);
		return 0;
	}
	/*
	This function passes weight or grad by following original cross-layer max pooling path
	Likes a inverse version of reversemaxlayer
	*/
	DLL extern int followmaxlayer(float*grad,float*data,float*out,int nd,int dc,int dy,int dx,int size)
	{
		int oc=dc/size;

		int datablocksize=dc*dy*dx*sizeof(float);
		int outblocksize=oc*dy*dx*sizeof(float);
		int blockcount=CUDAMemory/datablocksize;
		if (blockcount>nd) blockcount=nd+CUDAAligns-1;
		blockcount=blockcount/CUDAAligns*CUDAAligns;
		if (blockcount==0) return INSUFFICIENT_MEMORY;

		float*hostgrad,*hostdata,*hostout;
		float*devicegrad,*devicedata,*deviceout;
		hostgrad=(float*)malloc(blockcount*datablocksize);
		hostdata=(float*)malloc(blockcount*datablocksize);
		hostout=(float*)malloc(blockcount*outblocksize);
		CUDAASSERT(cudaMalloc(&devicegrad,blockcount*datablocksize));
		CUDAASSERT(cudaMalloc(&devicedata,blockcount*datablocksize));
		CUDAASSERT(cudaMalloc(&deviceout,blockcount*outblocksize));

		for (unsigned long long pd=0;pd<nd;pd+=blockcount) {
			int pdo=0;
			for (;pdo<blockcount;pdo++) if (pdo+pd<nd)
			{
				for (int pc=0;pc<dc;pc++)
					for (int py=0;py<dy;py++)
						for (int px=0;px<dx;px++)
							hostdata[((pc*dy+py)*dx+px)*blockcount+pdo]=data[(((pdo+pd)*dc+pc)*dy+py)*dx+px];
				for (int pc=0;pc<dc;pc++)
					for (int py=0;py<dy;py++)
						for (int px=0;px<dx;px++)
							hostgrad[((pc*dy+py)*dx+px)*blockcount+pdo]=grad[(((pdo+pd)*dc+pc)*dy+py)*dx+px];
			}
			else break;

			unsigned long long xval=(blockcount*outblocksize/sizeof(float)+CUDALines-1)/CUDALines;
			dim3 gridDim;
			gridDim.y=(xval+1023)/1024; if (xval>=1024) gridDim.x=1024; else gridDim.x=xval;
			gridDim.z=(gridDim.y+1023)/1024; if (gridDim.y>=1024) gridDim.y=1024;
			CUDAASSERT(cudaMemcpy(devicedata,hostdata,blockcount*datablocksize,cudaMemcpyHostToDevice));
			CUDAASSERT(cudaMemcpy(devicegrad,hostgrad,blockcount*datablocksize,cudaMemcpyHostToDevice));

			kfollowmaxlayer<<<gridDim,CUDALines>>>(devicedata,devicegrad,deviceout,pdo,blockcount,oc,dy*dx,size);
			CUDAASSERT(cudaMemcpy(hostout,deviceout,blockcount*outblocksize,cudaMemcpyDeviceToHost));

			for (pdo=0;pdo<blockcount;pdo++) if (pdo+pd<nd)
				for (int pc=0;pc<oc;pc++)
					for (int py=0;py<dy;py++)
						for (int px=0;px<dx;px++)
							out[(((pdo+pd)*oc+pc)*dy+py)*dx+px]=hostout[((pc*dy+py)*dx+px)*blockcount+pdo];
			else break;
		}
		CUDAASSERT(cudaFree(devicedata));
		CUDAASSERT(cudaFree(deviceout));
		CUDAASSERT(cudaFree(devicegrad));
		free(hostdata);
		free(hostout);
		free(hostgrad);
		return 0;
	}
	DLL extern int removeDC(float*data,float*out,int nd,int count)
	{
		int blockcount=CUDAMemory/(count*sizeof(float));
		if (blockcount==0) return INSUFFICIENT_MEMORY;

		float*devicedata,*devicecount;

		CUDAASSERT(cudaMalloc(&devicedata,blockcount*count*sizeof(float)));
		CUDAASSERT(cudaMalloc(&devicecount,count*sizeof(float)));
		CUDAASSERT(cudaMemset(devicecount,0,count*sizeof(float)));
		unsigned long long xval=(count+CUDALines-1)/CUDALines;
		dim3 gridDim;
		gridDim.y=(xval+1023)/1024; if (xval>=1024) gridDim.x=1024; else gridDim.x=xval;
		gridDim.z=(gridDim.y+1023)/1024; if (gridDim.y>=1024) gridDim.y=1024;

		for (unsigned long long pd=0;pd<nd;pd+=blockcount)
		{
			int pdo=min(blockcount,(int)(nd-pd));
			CUDAASSERT(cudaMemcpy(devicedata,data+pd*count,pdo*count*sizeof(float),cudaMemcpyHostToDevice));
			kaccumDC<<<gridDim,CUDALines>>>(devicedata,devicecount,pdo,count);
		}
		CUDAASSERT(cudaThreadSynchronize());
		kdivi<<<gridDim,CUDALines>>>(devicecount,nd,count);
		CUDAASSERT(cudaThreadSynchronize());
		for (unsigned long long pd=0;pd<nd;pd+=blockcount)
		{
			int pdo=min(blockcount,(int)(nd-pd));
			CUDAASSERT(cudaMemcpy(devicedata,data+pd*count,pdo*count*sizeof(float),cudaMemcpyHostToDevice));
			ksub<<<gridDim,CUDALines>>>(devicedata,devicecount,pdo,count);
			CUDAASSERT(cudaMemcpy(out+pd*count,devicedata,pdo*count*sizeof(float),cudaMemcpyDeviceToHost));
		}
		CUDAASSERT(cudaFree(devicedata));
		CUDAASSERT(cudaFree(devicecount));

		return 0;
	}
	/*
	Block permutation (on cpu), order is what data i will be
	*/
	DLL extern int blockpermutation(float*data,int*order,int len,int count)
	{
		int*flags,*flagorder;
		flags=(int*)malloc(len*sizeof(int));
		flagorder=(int*)malloc(len*sizeof(int));
		memset(flags,0,len*sizeof(int));
		for (int i=0;i<len;i++)
		{
			if (order[i]>=len||order[i]<0) {
				free(flags);
				free(flagorder);
				return INVALID_ARGUMENT;
			}
			flags[order[i]]++;
			flagorder[order[i]]=i;
		}

		for (int i=0;i<len;i++)
			if (flags[i]!=1) {
				free(flags);
				free(flagorder);
				return INVALID_ARGUMENT;
			}

		memset(flags,0,len*sizeof(int));

		float*tmp;
		tmp=(float*)malloc(count*sizeof(float));

		for (int i=0;i<len;i++)
		{
			if (flags[i]==1) continue;
			if (flagorder[i]==i) continue;
			memcpy(tmp,&data[((unsigned long long )(i))*count],count*sizeof(float));
			int p;
			for (p=i;flags[flagorder[p]]==0;p=flagorder[p])
			{
				flags[p]=1;
				memcpy(&data[((unsigned long long )(p))*count],&data[((unsigned long long )(flagorder[p]))*count],count*sizeof(float));
			}
			memcpy(&data[((unsigned long long )(p))*count],tmp,count*sizeof(float));
		}

		free(flags);
		free(flagorder);
		free(tmp);

		return 0;
	}
	DLL extern int normalize(float* weight, int out, int in)
	{
		float* datapool;
		int blocksize=65536;
		if (blocksize*in*sizeof(float)>CUDAMemory)
			blocksize = CUDAMemory/in/sizeof(float);
		if (blocksize==0) return INSUFFICIENT_MEMORY;

		CUDAASSERT(cudaMalloc(&datapool,blocksize*in*sizeof(float)));
		for (unsigned long long pd=0;pd<out;pd+=blocksize) {
			CUDAASSERT(cudaMemcpy(datapool,weight+pd*in,(pd+blocksize>=out?out-pd:blocksize)*in*sizeof(float),cudaMemcpyHostToDevice));
			normalize_kern<<<((pd+blocksize>=out?out-pd:blocksize)+CUDALines-1)/CUDALines,CUDALines>>>(datapool,pd+blocksize>=out?out-pd:blocksize,in);
			CUDAASSERT(cudaMemcpy(weight+pd*in,datapool,(pd+blocksize>=out?out-pd:blocksize)*in*sizeof(float),cudaMemcpyDeviceToHost));
		}
		CUDAASSERT(cudaFree(datapool));

		return 0;
	}
	/*CPU Function: balance weights*/
	DLL extern void balance(float* weight, int count,int dim)
	{
		for (unsigned long long i=0;i<count;i++)
		{
			float sum=0;
			for (int j=0;j<dim;j++) sum+=weight[i*dim+j];
			sum/=dim;
			for (int j=0;j<dim;j++) weight[i*dim+j]-=sum;
		}
	}
	/*
	This function does a general map-style transformation
	op:
	0:<1
	1:<[d0]
	2:<[d1]
	3:<[d2]
	4:<[d3]
	5:<[(float)next 4 byte]
	6:<id
	7:<<+>
	8:<<->
	9:<->
	10:<<*>
	11:<</>
	12:<<%>
	13:<sin>
	14:<cos>
	15:<tan>
	16:<cot>
	17:<sec>
	18:<csc>
	19:<1/(.+eps)>
	20:<1+>
	21:<exp>
	22:<log>
	23:<sinh>
	24:<cosh>
	25:<tanh>
	26:<coth>
	27:<sech>
	28:<csch>
	29:<sqr>
	30:<sqrt>
	31:<<pow>
	32:<asin>
	33:<acos>
	34:<atan>
	35:<fabs>
	36:<ceil>
	37:<floor>
	38:<sigmoid>
	39:<x(1-x)>
	40:<dup-1>>
	41:<dup-2>>
	42:<dup-3>>
	43:<dup-4>>
	44:<<(>)>   (not float out)
	45:<<(<)>   (not float out)
	46:<<(== (1e-6))>   (not float out)
	47:<<(>= (1e-6))>   (not float out)
	48:<<(<= (1e-6))>   (not float out)
	49:<<(!= (1e-6))>   (not float out)
	50:<<(== (exact)>	(not float out)
	51:<<and>   (not float out)
	52:<<or>   (not float out)
	53:<<xor>   (not float out)
	54:<not>   (not float out)
	55:<<<[0]?[-1]:[-2]>
	56:<<swap>>
	57:<isnan>   (not float out)
	58:<isinf>    (not float out)

	Every function with multiple args, it is feed on sequence of poping stack
	returns on stack value
	*/
	DLL extern int transform(float*d0,unsigned char*operates,int oplen,unsigned long size,float*out,float*d1,float*d2,float*d3)
	{
		if (oplen>MAX_TRANSFORM) return INVALID_ARGUMENT;
		__set_op_transform(operates,oplen);
		//Scan how many variables needed
		float*darr[4]={d0,d1,d2,d3};
		int largest=0;
		for (int p=0;p<oplen;p++)
			if (operates[p]==5) p+=4;
			else if (operates[p]<5)
				if (operates[p]>largest) largest=operates[p];
		float*deviced[4];
		for (int i=0;i<largest;i++)
			CUDAASSERT(cudaMalloc(&deviced[i],CUDAMemory));
		float*deviceout;
		CUDAASSERT(cudaMalloc(&deviceout,CUDAMemory));
		int sect=CUDAMemory/sizeof(float)/(largest==0?1:largest);
		if (sect>1024*256*CUDAAligns) sect=1024*256*CUDAAligns;
		int dsect=(sect+CUDALines-1)/CUDALines;
		for (unsigned long long ps=0;ps<size;ps+=sect)
		{
			for (int i=0;i<largest;i++)
				CUDAASSERT(cudaMemcpy(deviced[i],&darr[i][ps],((size-ps>=sect)?sect:(size-ps))*sizeof(float),cudaMemcpyHostToDevice));
			dim3 gridDim;

			gridDim.y=(dsect+1023)/1024; if (dsect>=1024) gridDim.x=1024; else gridDim.x=dsect;
			gridDim.z=(gridDim.y+1023)/1024; if (gridDim.y>=1024) gridDim.y=1024;
			sprintf(custom,"(%d,%d,%d,%d)",gridDim.x,gridDim.y,gridDim.z,CUDAAligns);
			ktransform<<<gridDim,CUDALines>>>(deviced[0],deviced[1],deviced[2],deviced[3],deviceout,(size-ps>=sect)?sect:(size-ps),oplen,ps);
			CUDAASSERT(cudaMemcpy(&out[ps],deviceout,((size-ps>=sect)?sect:(size-ps))*sizeof(float),cudaMemcpyDeviceToHost));
		}
		for (int i=0;i<largest;i++)
			CUDAASSERT(cudaFree(deviced[i]));
		CUDAASSERT(cudaFree(deviceout));
		return 0;
	}
	DLL extern int transformD(double*d0,unsigned char*operates,int oplen,unsigned long size,double*out,double*d1,double*d2,double*d3)
	{
		if (oplen>MAX_TRANSFORM) return INVALID_ARGUMENT;
		__set_op_transform(operates,oplen);
		//Scan how many variables needed
		double*darr[4]={d0,d1,d2,d3};
		int largest=0;
		for (int p=0;p<oplen;p++)
			if (operates[p]==5) p+=8;
			else if (operates[p]<5)
				if (operates[p]>largest) largest=operates[p];
		double*deviced[4];
		for (int i=0;i<largest;i++)
			CUDAASSERT(cudaMalloc(&deviced[i],CUDAMemory));
		double*deviceout;
		CUDAASSERT(cudaMalloc(&deviceout,CUDAMemory));
		int sect=CUDAMemory/sizeof(double)/(largest==0?1:largest);
		if (sect>1024*256*CUDAAligns) sect=1024*256*CUDAAligns;
		int dsect=(sect+CUDALines-1)/CUDALines;
		for (unsigned long long ps=0;ps<size;ps+=sect)
		{
			for (int i=0;i<largest;i++)
				CUDAASSERT(cudaMemcpy(deviced[i],&darr[i][ps],((size-ps>=sect)?sect:(size-ps))*sizeof(double),cudaMemcpyHostToDevice));
			dim3 gridDim;

			gridDim.y=(dsect+1023)/1024; if (dsect>=1024) gridDim.x=1024; else gridDim.x=dsect;
			gridDim.z=(gridDim.y+1023)/1024; if (gridDim.y>=1024) gridDim.y=1024;
			sprintf(custom,"(%d,%d,%d,%d)",gridDim.x,gridDim.y,gridDim.z,CUDAAligns);
			ktransformD<<<gridDim,CUDALines>>>(deviced[0],deviced[1],deviced[2],deviced[3],deviceout,(size-ps>=sect)?sect:(size-ps),oplen,ps);
			CUDAASSERT(cudaMemcpy(&out[ps],deviceout,((size-ps>=sect)?sect:(size-ps))*sizeof(double),cudaMemcpyDeviceToHost));
		}
		for (int i=0;i<largest;i++)
			CUDAASSERT(cudaFree(deviced[i]));
		CUDAASSERT(cudaFree(deviceout));
		return 0;
	}
    DLL extern int transformgpu(float*d0,unsigned char*operates,int oplen,unsigned long size,float*out,float*d1,float*d2,float*d3)
    {
        if (oplen>MAX_TRANSFORM) return INVALID_ARGUMENT;
        __set_op_transform(operates,oplen);
        
        dim3 gridDim;
        int dsect=(size+CUDALines-1)/CUDALines;
        gridDim.y=(dsect+1023)/1024; if (dsect>=1024) gridDim.x=1024; else gridDim.x=dsect;
        gridDim.z=(gridDim.y+1023)/1024; if (gridDim.y>=1024) gridDim.y=1024;

        sprintf(custom,"(%d,%d,%d,%d)",gridDim.x,gridDim.y,gridDim.z,CUDAAligns);
        ktransform<<<gridDim,CUDALines>>>(d0,d1,d2,d3,out,size,oplen,0);
        CUDAASSERT(cudaThreadSynchronize());
        return 0;
    }
	DLL extern int transform2(float*d0,unsigned char*operates,int oplen,unsigned long size,float*out,float*out2,float*d1,float*d2,float*d3)
	{
		if (oplen>MAX_TRANSFORM) return INVALID_ARGUMENT;
		__set_op_transform(operates,oplen);
		//Scan how many variables needed
		float*darr[4]={d0,d1,d2,d3};
		int largest=0;
		for (int p=0;p<oplen;p++)
			if (operates[p]==5) p+=4;
			else if (operates[p]<5)
				if (operates[p]>largest) largest=operates[p];
		float*deviced[4];
		for (int i=0;i<largest;i++)
			CUDAASSERT(cudaMalloc(&deviced[i],CUDAMemory));
		float*deviceout,*deviceout2;
		CUDAASSERT(cudaMalloc(&deviceout,CUDAMemory));
		CUDAASSERT(cudaMalloc(&deviceout2,CUDAMemory));
		int sect=CUDAMemory/sizeof(float)/(largest==0?1:largest);
		if (sect>1024*256*CUDAAligns) sect=1024*256*CUDAAligns;
		int dsect=(sect+CUDALines-1)/CUDALines;
		for (unsigned long long ps=0;ps<size;ps+=sect)
		{
			for (int i=0;i<largest;i++)
				CUDAASSERT(cudaMemcpy(deviced[i],&darr[i][ps],((size-ps>=sect)?sect:(size-ps))*sizeof(float),cudaMemcpyHostToDevice));
			dim3 gridDim;

			gridDim.y=(dsect+1023)/1024; if (dsect>=1024) gridDim.x=1024; else gridDim.x=dsect;
			gridDim.z=(gridDim.y+1023)/1024; if (gridDim.y>=1024) gridDim.y=1024;
			sprintf(custom,"(%d,%d,%d,%d)",gridDim.x,gridDim.y,gridDim.z,CUDAAligns);
			ktransform2<<<gridDim,CUDALines>>>(deviced[0],deviced[1],deviced[2],deviced[3],deviceout,deviceout2,(size-ps>=sect)?sect:(size-ps),oplen,ps);
			CUDAASSERT(cudaMemcpy(&out[ps],deviceout,((size-ps>=sect)?sect:(size-ps))*sizeof(float),cudaMemcpyDeviceToHost));
			CUDAASSERT(cudaMemcpy(&out2[ps],deviceout2,((size-ps>=sect)?sect:(size-ps))*sizeof(float),cudaMemcpyDeviceToHost));
		}
		for (int i=0;i<largest;i++)
			CUDAASSERT(cudaFree(deviced[i]));
		CUDAASSERT(cudaFree(deviceout));
		CUDAASSERT(cudaFree(deviceout2));
		return 0;
	}
    DLL extern int transformgpu2(float*d0,unsigned char*operates,int oplen,unsigned long size,float*out,float*out2,float*d1,float*d2,float*d3)
    {
        if (oplen>MAX_TRANSFORM) return INVALID_ARGUMENT;
        __set_op_transform(operates,oplen);

        dim3 gridDim;
        int dsect=(size+CUDALines-1)/CUDALines;
        gridDim.y=(dsect+1023)/1024; if (dsect>=1024) gridDim.x=1024; else gridDim.x=dsect;
        gridDim.z=(gridDim.y+1023)/1024; if (gridDim.y>=1024) gridDim.y=1024;

        sprintf(custom,"(%d,%d,%d,%d)",gridDim.x,gridDim.y,gridDim.z,CUDAAligns);
        ktransform2<<<gridDim,CUDALines>>>(d0,d1,d2,d3,out,out2,size,oplen,0);
        CUDAASSERT(cudaThreadSynchronize());
        return 0;
    }
}

