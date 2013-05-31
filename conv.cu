/*
CUDANPLEARN Convolution kernel support functions
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


inline bool checkCUDAError() {
    cudaError_t err = cudaGetLastError();
    return cudaSuccess != err;
}

template<typename T>
inline T&min(const T&a,const T&b)
{return a<b?a:b;}

/*
Convolution related functions
*/

extern "C" {
	/*
    The function to do size-keep convolution
    data is [nd [dc [dy [dx [float]]]]]
    filter is [dc [fy [fx [float]]]]
    out is [nd [dc [dy [dx [float]]]]]
    */
    DLL extern int convkeep4D(float*data,float*filters,int nd,int dc,int dy,int dx,int fy,int fx,float*out)
    {
        int datablocksize=dy*dx*dc*sizeof(float);
        int blockcount=CUDAMemory/datablocksize;
        if (blockcount>nd) blockcount=nd+CUDAAligns-1;
        blockcount=blockcount/CUDAAligns*CUDAAligns;
        if (blockcount==0) return INSUFFICIENT_MEMORY;
        
        float*devicefilters,*devicedata,*deviceout;
        CUDAASSERT(cudaMalloc(&devicefilters,dc*fy*fx*sizeof(float)));
        CUDAASSERT(cudaMemcpy(devicefilters,filters,dc*fy*fx*sizeof(float),cudaMemcpyHostToDevice));

        float*hostdata,*hostout;
        hostdata=(float*)malloc(blockcount*datablocksize);
        hostout=(float*)malloc(blockcount*datablocksize);

        CUDAASSERT(cudaMalloc(&devicedata,blockcount*datablocksize));
        CUDAASSERT(cudaMalloc(&deviceout,blockcount*datablocksize));
        
        for (unsigned long long pd=0;pd<nd;pd+=blockcount) {
            int pdo=0;
            for (;pdo<blockcount;pdo++) if (pdo+pd<nd)
                for (int pc=0;pc<dc;pc++)
                    for (int py=0;py<dy;py++)
                        for (int px=0;px<dx;px++)
                            hostdata[((pc*dy+py)*dx+px)*blockcount+pdo]=data[(((pdo+pd)*dc+pc)*dy+py)*dx+px];
            else break;

            dim3 gridDim;
            unsigned long long xval=(blockcount*datablocksize/sizeof(float)+CUDALines-1)/CUDALines;
            gridDim.y=(xval+1023)/1024; if (xval>1024) gridDim.x=1024; else gridDim.x=xval;
            gridDim.z=(gridDim.y+1023)/1024; if (gridDim.y>=1024) gridDim.y=1024;
            CUDAASSERT(cudaMemcpy(devicedata,hostdata,blockcount*datablocksize,cudaMemcpyHostToDevice));

            kkeepconv4D<<<gridDim,CUDALines>>>(devicedata,devicefilters,deviceout,pdo,blockcount,dc,dy,dx,fy,fx);
            CUDAASSERT(cudaMemcpy(hostout,deviceout,blockcount*datablocksize,cudaMemcpyDeviceToHost));

            pdo=0;
            for (;pdo<blockcount;pdo++) if (pdo+pd<nd)
                for (int pc=0;pc<dc;pc++)
                    for (int py=0;py<dy;py++)
                        for (int px=0;px<dx;px++)
                            out[(((pdo+pd)*dc+pc)*dy+py)*dx+px]=hostout[((pc*dy+py)*dx+px)*blockcount+pdo];
            else break;
        }
        CUDAASSERT(cudaFree(deviceout));
        CUDAASSERT(cudaFree(devicedata));
        CUDAASSERT(cudaFree(devicefilters));

        free(hostdata);
        free(hostout);
        return 0;
    }
    DLL extern int convkeep4Dalllayer(float*data,float*filters,int nd,int dc,int dy,int dx,int fc,int fy,int fx,float*out)
    {
        int datablocksize=dy*dx*dc*sizeof(float);
        int outblocksize=fc*dy*dx*sizeof(float);
        int blockcount=CUDAMemory/max(datablocksize,outblocksize);
        if (blockcount>nd) blockcount=nd+CUDAAligns-1;
        blockcount=blockcount/CUDAAligns*CUDAAligns;
        if (blockcount==0) return INSUFFICIENT_MEMORY;
        
        float*devicefilters,*devicedata,*deviceout;
        CUDAASSERT(cudaMalloc(&devicefilters,dc*fy*fx*fc*sizeof(float)));
        CUDAASSERT(cudaMemcpy(devicefilters,filters,dc*fy*fx*fc*sizeof(float),cudaMemcpyHostToDevice));

        float*hostdata,*hostout;
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
                            hostdata[((pc*dy+py)*dx+px)*blockcount+pdo]=data[(((pdo+pd)*dy+py)*dx+px)*dc+pc];
            else break;

            dim3 gridDim;
            unsigned long long xval=(blockcount*outblocksize/sizeof(float)+CUDALines-1)/CUDALines;
            gridDim.y=(xval+1023)/1024; if (xval>1024) gridDim.x=1024; else gridDim.x=xval;
            gridDim.z=(gridDim.y+1023)/1024; if (gridDim.y>=1024) gridDim.y=1024;
            CUDAASSERT(cudaMemcpy(devicedata,hostdata,blockcount*datablocksize,cudaMemcpyHostToDevice));

            kkeepconv4Dalllayer<<<gridDim,CUDALines>>>(devicedata,devicefilters,deviceout,pdo,blockcount,dc,dy,dx,fc,fy,fx);
            CUDAASSERT(cudaMemcpy(hostout,deviceout,blockcount*outblocksize,cudaMemcpyDeviceToHost));

            pdo=0;
            for (;pdo<blockcount;pdo++) if (pdo+pd<nd)
                for (int pc=0;pc<fc;pc++)
                    for (int py=0;py<dy;py++)
                        for (int px=0;px<dx;px++)
                            out[(((pdo+pd)*fc+pc)*dy+py)*dx+px]=hostout[((pc*dy+py)*dx+px)*blockcount+pdo];
            else break;
        }
        CUDAASSERT(cudaFree(deviceout));
        CUDAASSERT(cudaFree(devicedata));
        CUDAASSERT(cudaFree(devicefilters));

        free(hostdata);
        free(hostout);
        return 0;
    }


    DLL extern int gradconvkeep4D(float*data,float*grad,float*filterout,int nd,int dc,int dy,int dx,int fc,int fy,int fx)
    {
		int datablocksize=dy*dx*dc*sizeof(float);
        int outblocksize=dy*dx*fc*sizeof(float);
		int blockcount=CUDAMemory/max(datablocksize,outblocksize);
		if (blockcount>nd) blockcount=nd;
		if (blockcount==0) return INSUFFICIENT_MEMORY;

		float*devicefilters,*devicedata,*devicegrad;
		CUDAASSERT(cudaMalloc(&devicefilters,fc*fy*fx*dc*sizeof(float)));
		CUDAASSERT(cudaMemset(devicefilters,0,fc*fy*fx*dc*sizeof(float)));
		CUDAASSERT(cudaMalloc(&devicedata,blockcount*datablocksize));
		CUDAASSERT(cudaMalloc(&devicegrad,blockcount*outblocksize));

		for (unsigned long long pd=0;pd<nd;pd+=blockcount) {
			int pdo=min(blockcount,(int)(nd-pd));
			CUDAASSERT(cudaMemcpy(devicedata,data+pd*dy*dx*dc,pdo*datablocksize,cudaMemcpyHostToDevice));
			CUDAASSERT(cudaMemcpy(devicegrad,grad+pd*dy*dx*fc,pdo*outblocksize,cudaMemcpyHostToDevice));

			unsigned long long xval=(fc*fy*fx*dc+CUDALines-1)/CUDALines;
			dim3 gridDim;
			gridDim.y=(xval+1023)/1024; if (xval>=1024) gridDim.x=1024; else gridDim.x=xval;
			gridDim.z=(gridDim.y+1023)/1024; if (gridDim.y>=1024) gridDim.y=1024;

			kgradconvkeep4D<<<gridDim,CUDALines>>>(devicedata,devicegrad,devicefilters,pdo,dy,dx,dc,fc,fy,fx);
			CUDAASSERT(cudaThreadSynchronize());
		}
		CUDAASSERT(cudaMemcpy(filterout,devicefilters,fc*fy*fx*dc*sizeof(float),cudaMemcpyDeviceToHost));
		CUDAASSERT(cudaFree(devicefilters));
		CUDAASSERT(cudaFree(devicedata));
		CUDAASSERT(cudaFree(devicegrad));
		return 0;
	
    }
    DLL extern int reverseconvkeep4D(float*grad,float*filter,float*dataout,int nd,int dc,int dy,int dx,int fc,int fy,int fx)
    {

		int datablocksize=dy*dx*dc*sizeof(float);
		int outblocksize=fc*dy*dx*sizeof(float);
		int blockcount=min(CUDAMemory/datablocksize,CUDAMemory/outblocksize);
		if (blockcount>nd) blockcount=nd+CUDAAligns-1;
		blockcount=blockcount/CUDAAligns*CUDAAligns;
		if (blockcount==0) return INSUFFICIENT_MEMORY;

		float*devicefilters,*devicedata,*deviceout;
		CUDAASSERT(cudaMalloc(&devicefilters,fc*fy*fx*dc*sizeof(float)));
		CUDAASSERT(cudaMemcpy(devicefilters,filter,fc*fy*fx*dc*sizeof(float),cudaMemcpyHostToDevice));

		float*hostdata,*hostout;
		hostdata=(float*)malloc(blockcount*datablocksize);
		hostout=(float*)malloc(blockcount*outblocksize);

		CUDAASSERT(cudaMalloc(&devicedata,blockcount*datablocksize));
		CUDAASSERT(cudaMalloc(&deviceout,blockcount*outblocksize));

		for (unsigned long long pd=0;pd<nd;pd+=blockcount) {
			int pdo=0;
			for (;pdo<blockcount;pdo++) if (pdo+pd<nd)
				for (int pc=0;pc<fc;pc++)
					for (int py=0;py<dy;py++)
						for (int px=0;px<dx;px++)
							hostout[((pc*dy+py)*dx+px)*blockcount+pdo]=grad[(((pdo+pd)*fc+pc)*dy+py)*dx+px];
			else break;

			unsigned long long xval=(blockcount*outblocksize/sizeof(float)+CUDALines-1)/CUDALines;
			dim3 gridDim;
			gridDim.y=(xval+1023)/1024; if (xval>=1024) gridDim.x=1024; else gridDim.x=xval;
			gridDim.z=(gridDim.y+1023)/1024; if (gridDim.y>=1024) gridDim.y=1024;
			CUDAASSERT(cudaMemcpy(deviceout,hostout,blockcount*outblocksize,cudaMemcpyHostToDevice));
			CUDAASSERT(cudaMemset(devicedata,0,blockcount*datablocksize));
			kreverseconvkeep4D<<<gridDim,CUDALines>>>(devicedata,devicefilters,deviceout,pdo,blockcount,dy,dx,dc,fc,fy,fx);
			CUDAASSERT(cudaMemcpy(hostdata,devicedata,blockcount*datablocksize,cudaMemcpyDeviceToHost));

			for (pdo=0;pdo<blockcount;pdo++) if (pdo+pd<nd)
				for (int py=0;py<dy;py++)
					for (int px=0;px<dx;px++)
						for (int pc=0;pc<dc;pc++)
							dataout[(((pdo+pd)*dy+py)*dx+px)*dc+pc]=hostdata[((py*dx+px)*dc+pc)*blockcount+pdo];
			else break;

		}
		CUDAASSERT(cudaFree(devicefilters));
		CUDAASSERT(cudaFree(devicedata));
		CUDAASSERT(cudaFree(deviceout));
		free(hostdata);
		free(hostout);
		return 0;
    }
	/*
	The function to do 4D convolution
	data is [nd [dy [dx [dc [float]]]]]
	filter is [fc [fy [fx [dc [float]]]]]
	out is [nd [fc [dy-fy+1 [dx-fx+1 [float]]]]]
	*/
	DLL extern int convolution4D(float*data,float*filters,int nd,int dy,int dx,int dc,int fc,int fy,int fx,float*out)
	{
		int oy=dy-fy+1, ox=dx-fx+1;

		int datablocksize=dy*dx*dc*sizeof(float);
		int outblocksize=fc*oy*ox*sizeof(float);
		int blockcount=min(CUDAMemory/datablocksize,CUDAMemory/outblocksize);
		if (blockcount>nd) blockcount=nd+CUDAAligns-1;
		blockcount=blockcount/CUDAAligns*CUDAAligns;
		if (blockcount==0) return INSUFFICIENT_MEMORY;

		float*devicefilters,*devicedata,*deviceout;
		CUDAASSERT(cudaMalloc(&devicefilters,fc*fy*fx*dc*sizeof(float)));
		CUDAASSERT(cudaMemcpy(devicefilters,filters,fc*fy*fx*dc*sizeof(float),cudaMemcpyHostToDevice));

		float*hostdata,*hostout;
		hostdata=(float*)malloc(blockcount*datablocksize);
		hostout=(float*)malloc(blockcount*outblocksize);

		CUDAASSERT(cudaMalloc(&devicedata,blockcount*datablocksize));
		CUDAASSERT(cudaMalloc(&deviceout,blockcount*outblocksize));

		for (unsigned long long pd=0;pd<nd;pd+=blockcount) {
			int pdo=0;
			for (;pdo<blockcount;pdo++) if (pdo+pd<nd)
				for (int py=0;py<dy;py++)
					for (int px=0;px<dx;px++)
						for (int pc=0;pc<dc;pc++)
							hostdata[((py*dx+px)*dc+pc)*blockcount+pdo]=data[(((pdo+pd)*dy+py)*dx+px)*dc+pc];
			else break;

			unsigned long long xval=(blockcount*outblocksize/sizeof(float)+CUDALines-1)/CUDALines;
			dim3 gridDim;
			gridDim.y=(xval+1023)/1024; if (xval>=1024) gridDim.x=1024; else gridDim.x=xval;
			gridDim.z=(gridDim.y+1023)/1024; if (gridDim.y>=1024) gridDim.y=1024;
			CUDAASSERT(cudaMemcpy(devicedata,hostdata,blockcount*datablocksize,cudaMemcpyHostToDevice));

			kconvolution4D<<<gridDim,CUDALines>>>(devicedata,devicefilters,deviceout,pdo,blockcount,dy,dx,dc,fc,fy,fx,oy,ox);
			CUDAASSERT(cudaMemcpy(hostout,deviceout,blockcount*outblocksize,cudaMemcpyDeviceToHost));

			for (pdo=0;pdo<blockcount;pdo++) if (pdo+pd<nd)
				for (int pc=0;pc<fc;pc++)
					for (int py=0;py<oy;py++)
						for (int px=0;px<ox;px++)
							out[(((pdo+pd)*fc+pc)*oy+py)*ox+px]=hostout[((pc*oy+py)*ox+px)*blockcount+pdo];
			else break;

		}
		CUDAASSERT(cudaFree(devicefilters));
		CUDAASSERT(cudaFree(devicedata));
		CUDAASSERT(cudaFree(deviceout));
		free(hostdata);
		free(hostout);
		return 0;
	}

	/*
	This function does a grad extraction of convolution4D
	*/
	DLL extern int gradconvolution4D(float*data,float*grad,float*filterout,int nd,int dy,int dx,int dc,int fc,int fy,int fx)
	{
		int oy=dy-fy+1, ox=dx-fx+1;

		int datablocksize=dy*dx*dc*sizeof(float);
		int outblocksize=fc*oy*ox*sizeof(float);
		int blockcount=min(CUDAMemory/datablocksize,CUDAMemory/outblocksize);
		if (blockcount>nd) blockcount=nd;
		if (blockcount==0) return INSUFFICIENT_MEMORY;

		float*devicefilters,*devicedata,*devicegrad;
		CUDAASSERT(cudaMalloc(&devicefilters,fc*fy*fx*dc*sizeof(float)));
		CUDAASSERT(cudaMemset(devicefilters,0,fc*fy*fx*dc*sizeof(float)));
		CUDAASSERT(cudaMalloc(&devicedata,blockcount*datablocksize));
		CUDAASSERT(cudaMalloc(&devicegrad,blockcount*outblocksize));

		for (unsigned long long pd=0;pd<nd;pd+=blockcount) {
			int pdo=min(blockcount,(int)(nd-pd));
			CUDAASSERT(cudaMemcpy(devicedata,data+pd*dy*dx*dc,pdo*datablocksize,cudaMemcpyHostToDevice));
			CUDAASSERT(cudaMemcpy(devicegrad,grad+pd*oy*ox*fc,pdo*outblocksize,cudaMemcpyHostToDevice));

			unsigned long long xval=(fc*fy*fx*dc+CUDALines-1)/CUDALines;
			dim3 gridDim;
			gridDim.y=(xval+1023)/1024; if (xval>=1024) gridDim.x=1024; else gridDim.x=xval;
			gridDim.z=(gridDim.y+1023)/1024; if (gridDim.y>=1024) gridDim.y=1024;

			kgradconvolution4D<<<gridDim,CUDALines>>>(devicedata,devicegrad,devicefilters,pdo,dy,dx,dc,fc,fy,fx,oy,ox);
			CUDAASSERT(cudaThreadSynchronize());
		}
		CUDAASSERT(cudaMemcpy(filterout,devicefilters,fc*fy*fx*dc*sizeof(float),cudaMemcpyDeviceToHost));
		CUDAASSERT(cudaFree(devicefilters));
		CUDAASSERT(cudaFree(devicedata));
		CUDAASSERT(cudaFree(devicegrad));
		return 0;
	}
	/*
	This function does a reverse passing on convolution4D
	*/
	DLL extern int reverseconvolution4D(float*grad,float*filters,float*out,int nd,int dy,int dx,int dc,int fc,int fy,int fx)
	{
		int oy=dy-fy+1, ox=dx-fx+1;

		int datablocksize=dy*dx*dc*sizeof(float);
		int outblocksize=fc*oy*ox*sizeof(float);
		int blockcount=min(CUDAMemory/datablocksize,CUDAMemory/outblocksize);
		if (blockcount>nd) blockcount=nd+CUDAAligns-1;
		blockcount=blockcount/CUDAAligns*CUDAAligns;
		if (blockcount==0) return INSUFFICIENT_MEMORY;

		float*devicefilters,*devicedata,*deviceout;
		CUDAASSERT(cudaMalloc(&devicefilters,fc*fy*fx*dc*sizeof(float)));
		CUDAASSERT(cudaMemcpy(devicefilters,filters,fc*fy*fx*dc*sizeof(float),cudaMemcpyHostToDevice));

		float*hostdata,*hostout;
		hostdata=(float*)malloc(blockcount*datablocksize);
		hostout=(float*)malloc(blockcount*outblocksize);

		CUDAASSERT(cudaMalloc(&devicedata,blockcount*datablocksize));
		CUDAASSERT(cudaMalloc(&deviceout,blockcount*outblocksize));

		for (unsigned long long pd=0;pd<nd;pd+=blockcount) {
			int pdo=0;
			for (;pdo<blockcount;pdo++) if (pdo+pd<nd)
				for (int pc=0;pc<fc;pc++)
					for (int py=0;py<oy;py++)
						for (int px=0;px<ox;px++)
							hostout[((pc*oy+py)*ox+px)*blockcount+pdo]=grad[(((pdo+pd)*fc+pc)*oy+py)*ox+px];
			else break;

			unsigned long long xval=(blockcount*outblocksize/sizeof(float)+CUDALines-1)/CUDALines;
			dim3 gridDim;
			gridDim.y=(xval+1023)/1024; if (xval>=1024) gridDim.x=1024; else gridDim.x=xval;
			gridDim.z=(gridDim.y+1023)/1024; if (gridDim.y>=1024) gridDim.y=1024;
			CUDAASSERT(cudaMemcpy(deviceout,hostout,blockcount*outblocksize,cudaMemcpyHostToDevice));
			CUDAASSERT(cudaMemset(devicedata,0,blockcount*datablocksize));
			kreverseconvolution4D<<<gridDim,CUDALines>>>(devicedata,devicefilters,deviceout,pdo,blockcount,dy,dx,dc,fc,fy,fx,oy,ox);
			CUDAASSERT(cudaMemcpy(hostdata,devicedata,blockcount*datablocksize,cudaMemcpyDeviceToHost));

			for (pdo=0;pdo<blockcount;pdo++) if (pdo+pd<nd)
				for (int py=0;py<dy;py++)
					for (int px=0;px<dx;px++)
						for (int pc=0;pc<dc;pc++)
							out[(((pdo+pd)*dy+py)*dx+px)*dc+pc]=hostdata[((py*dx+px)*dc+pc)*blockcount+pdo];
			else break;

		}
		CUDAASSERT(cudaFree(devicefilters));
		CUDAASSERT(cudaFree(devicedata));
		CUDAASSERT(cudaFree(deviceout));
		free(hostdata);
		free(hostout);
		return 0;
	}
	/*
	This function does a reverse passing on convolution4D
	*/
	DLL extern int reverseconvolution4D_outordered(float*grad,float*filters,float*out,int nd,int dy,int dx,int dc,int fc,int fy,int fx)
	{
		int oy=dy-fy+1, ox=dx-fx+1;

		int datablocksize=dy*dx*dc*sizeof(float);
		int outblocksize=fc*oy*ox*sizeof(float);
		int blockcount=min(CUDAMemory/datablocksize,CUDAMemory/outblocksize);
		if (blockcount>nd) blockcount=nd+CUDAAligns-1;
		blockcount=blockcount/CUDAAligns*CUDAAligns;
		if (blockcount==0) return INSUFFICIENT_MEMORY;

		float*devicefilters,*devicedata,*deviceout;
		CUDAASSERT(cudaMalloc(&devicefilters,fc*fy*fx*dc*sizeof(float)));
		float*hostfilters;
		hostfilters=(float*)malloc(fc*fy*fx*dc*sizeof(float));
		//Rearrange order to [dc fc -fy -fx]
		for (int pc=0;pc<dc;pc++)
			for (int pco=0;pco<fc;pco++)
				for (int py=0;py<fy;py++)
					for (int px=0;px<fx;px++)
						hostfilters[((pc*fc+pco)*fy+py)*fx+px]=filters[((pco*fy+(fy-py-1))*fx+(fx-px-1))*dc+pc];
		CUDAASSERT(cudaMemcpy(devicefilters,hostfilters,fc*fy*fx*dc*sizeof(float),cudaMemcpyHostToDevice));
		free(hostfilters);

		float*hostdata,*hostout;
		hostdata=(float*)malloc(blockcount*datablocksize);
		hostout=(float*)malloc(blockcount*outblocksize);

		CUDAASSERT(cudaMalloc(&devicedata,blockcount*datablocksize));
		CUDAASSERT(cudaMalloc(&deviceout,blockcount*outblocksize));

		for (unsigned long long pd=0;pd<nd;pd+=blockcount) {
			int pdo=0;
			for (;pdo<blockcount;pdo++) if (pdo+pd<nd)
				for (int pc=0;pc<fc;pc++)
					for (int py=0;py<oy;py++)
						for (int px=0;px<ox;px++)
							hostout[((pc*oy+py)*ox+px)*blockcount+pdo]=grad[(((pdo+pd)*fc+pc)*oy+py)*ox+px];
			else break;

			unsigned long long xval=(blockcount*datablocksize/sizeof(float)+CUDALines-1)/CUDALines;
			dim3 gridDim;
			gridDim.y=(xval+1023)/1024; if (xval>=1024) gridDim.x=1024; else gridDim.x=xval;
			gridDim.z=(gridDim.y+1023)/1024; if (gridDim.y>=1024) gridDim.y=1024;
			CUDAASSERT(cudaMemcpy(deviceout,hostout,blockcount*outblocksize,cudaMemcpyHostToDevice));
			CUDAASSERT(cudaMemset(devicedata,0,blockcount*datablocksize));
			kreverseconvolution4D_outorder<<<gridDim,CUDALines>>>(devicedata,devicefilters,deviceout,pdo,blockcount,dc,fc,fy,fx,oy,ox);
			CUDAASSERT(cudaMemcpy(hostdata,devicedata,blockcount*datablocksize,cudaMemcpyDeviceToHost));

			for (pdo=0;pdo<blockcount;pdo++) if (pdo+pd<nd)
				for (int py=0;py<dy;py++)
					for (int px=0;px<dx;px++)
						for (int pc=0;pc<dc;pc++)
							out[(((pdo+pd)*dy+py)*dx+px)*dc+pc]=hostdata[((py*dx+px)*dc+pc)*blockcount+pdo];
			else break;

		}
		CUDAASSERT(cudaFree(devicefilters));
		CUDAASSERT(cudaFree(devicedata));
		CUDAASSERT(cudaFree(deviceout));
		free(hostdata);
		free(hostout);
		return 0;
	}
	DLL extern int reverseconvolution4D_smallfilter(float*grad,float*filters,float*out,int nd,int dy,int dx,int dc,int fc,int fy,int fx)
	{
		int oy=dy-fy+1, ox=dx-fx+1;

		int datablocksize=dy*dx*dc*sizeof(float);
		int outblocksize=fc*oy*ox*sizeof(float);
		int blockcount=min(CUDAMemory/datablocksize,CUDAMemory/outblocksize);
		if (blockcount>nd) blockcount=nd+CUDAAligns-1;
		blockcount=blockcount/CUDAAligns*CUDAAligns;
		if (blockcount==0) return INSUFFICIENT_MEMORY;

		float*devicefilters,*devicedata,*deviceout;
		CUDAASSERT(cudaMalloc(&devicefilters,fc*fy*fx*dc*sizeof(float)));
		float*hostfilters;
		hostfilters=(float*)malloc(fc*fy*fx*dc*sizeof(float));
		//Rearrange order to [dc fc fy fx]
		for (int pc=0;pc<dc;pc++)
			for (int pco=0;pco<fc;pco++)
				for (int py=0;py<fy;py++)
					for (int px=0;px<fx;px++)
						hostfilters[((pc*fc+pco)*fy+py)*fx+px]=filters[((pco*fy+py)*fx+px)*dc+pc];
		CUDAASSERT(cudaMemcpy(devicefilters,hostfilters,fc*fy*fx*dc*sizeof(float),cudaMemcpyHostToDevice));
		free(hostfilters);

		float*hostdata;
		hostdata=(float*)malloc(blockcount*datablocksize);

		CUDAASSERT(cudaMalloc(&devicedata,blockcount*datablocksize));
		CUDAASSERT(cudaMalloc(&deviceout,blockcount*outblocksize));

		for (unsigned long long pd=0;pd<nd;pd+=blockcount) {
			int pdo=min(blockcount,(int)(nd-pd));

			CUDAASSERT(cudaMemcpy(deviceout,&grad[pd*fc*oy*ox],pdo*outblocksize,cudaMemcpyHostToDevice));
			CUDAASSERT(cudaMemset(devicedata,0,pdo*datablocksize));
			//printf("pdo:%d,dy:%d,dx:%d,dc:%d,fc:%d,fy:%d,fx:%d,oy:%d,ox:%d\n",pdo,dy,dx,dc,fc,fy,fx,oy,ox);
			kreverseconvolution4D_smallfilter<<<pdo*dc,256>>>(devicedata,devicefilters,deviceout,pdo,dy,dx,dc,fc,fy,fx,oy,ox);
			CUDAASSERT(cudaMemcpy(hostdata,devicedata,pdo*datablocksize,cudaMemcpyDeviceToHost));

			//nd dc dy dx
			for (pdo=0;pdo<blockcount;pdo++) if (pdo+pd<nd)
				for (int py=0;py<dy;py++)
					for (int px=0;px<dx;px++)
						for (int pc=0;pc<dc;pc++)
							out[(((pdo+pd)*dy+py)*dx+px)*dc+pc]=hostdata[((pdo*dc+pc)*dy+py)*dx+px];
			else break;

		}
		CUDAASSERT(cudaFree(devicefilters));
		CUDAASSERT(cudaFree(devicedata));
		CUDAASSERT(cudaFree(deviceout));
		free(hostdata);
		return 0;
	}
}
