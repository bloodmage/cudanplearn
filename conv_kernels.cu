/*
CUDANPLEARN Cuda kernel functions
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

#include "nplearn.cuh"
#include<cuda.h>
#include<cuda_runtime.h>
#include "float.h"

__global__ void kkeepconv4D(const float*__restrict__ devicedata,const float*__restrict__ devicefilters,float*__restrict__ deviceout,uint32 pdo,uint32 blockcount,uint32 dc,uint32 dy,uint32 dx,uint32 fy,uint32 fx)
{
    int di= ((blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x; //pdo
    if (di>=blockcount*dc*dy*dx) return;
    int px=di/blockcount; di%=blockcount;
    int py=px/dx; px%=dx;
    int pc=py/dy; py%=dy;
    if (di>=pdo) return;
    float otmp=0.0f;

    py-=fy/2; px-=fx/2;
    for (int fyi=0;fyi<fy;fyi++) if (fyi+py>=0&&fyi+py<dy)
        for (int fxi=0;fxi<fx;fxi++) if (fxi+px>=0&&fxi+px<dx)
            otmp+=devicefilters[(pc*fy+fyi)*fx+fxi]*devicedata[((pc*dy+fyi+py)*dx+fxi+px)*blockcount+di];
    deviceout[((pc*dy+py+fy/2)*dx+px+fx/2)*blockcount+di]=otmp;
}
__global__ void kkeepconv4Dalllayer(const float*__restrict__ devicedata,const float*__restrict__ devicefilters,float*__restrict__ deviceout,uint32 pdo,uint32 blockcount,uint32 dc,uint32 dy,uint32 dx,uint32 fc,uint32 fy,uint32 fx)
{
    int di= ((blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x; //pdo
    if (di>=blockcount*fc*dy*dx) return;
    int px=di/blockcount; di%=blockcount;
    int py=px/dx; px%=dx;
    int pco=py/dy; py%=dy;
    if (di>=pdo) return;
    float otmp=0.0f;

    py-=fy/2; px-=fx/2;
    for (int fyi=0;fyi<fy;fyi++) if (fyi+py>=0&&fyi+py<dy)
        for (int fxi=0;fxi<fx;fxi++) if (fxi+px>=0&&fxi+px<dx)
            for (int pc=0;pc<dc;pc++) 
                otmp+=devicefilters[((pco*fy+fyi)*fx+fxi)*dc+pc]*devicedata[((pc*dy+fyi+py)*dx+fxi+px)*blockcount+di];
    deviceout[((pco*dy+py+fy/2)*dx+px+fx/2)*blockcount+di]=otmp;
}
__global__ void kgradconvkeep4D(const float*__restrict__ devicedata,const float*__restrict__ devicegrad,float*__restrict__ devicefilters,uint32 pdo,uint32 dy,uint32 dx,uint32 dc,uint32 fc,uint32 fy,uint32 fx)
{
    int oy=dy-fy+1, ox=dx-fx+1;
    int pc=((blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x; //pdo
    if (pc>=fc*fy*fx*dc) return;
    int pfx=pc/dc; pc%=dc;
    int pfy=pfx/fx; pfx%=fx;
    int pco=pfy/fy; pfy%=fy;
    pfx-=fx/2;
    pfy-=fy/2;
    float tmp=0.0f;
    for (int pi=0;pi<pdo;pi++)
        for (int pyo=0;pyo<dy;pyo++) if (((uint32)(pyo+pfy))<dy)
            for (int pxo=0;pxo<dx;pxo++) if (((uint32)(pxo+pfx))<dx)
                tmp+=devicegrad[((pi*fc+pco)*dy+pyo)*dx+pxo]*devicedata[((pi*dy+pyo+pfy)*dx+pxo+pfx)*dc+pc];

    devicefilters[((pco*fy+pfy+fy/2)*fx+pfx+fx/2)*dc+pc]+=tmp;
}
__global__ void kreverseconvkeep4D(float*__restrict__ devicedata,const float*__restrict__ devicefilters,const float*__restrict__ deviceout,uint32 pdo,uint32 blockcount,uint32 dy,uint32 dx,uint32 dc,uint32 fc,uint32 fy,uint32 fx)
{
	int di= ((blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x; //pdo
	if (di>=blockcount*dc*dy*dx) return;
	int pxo=di/blockcount; di%=blockcount; //ox
	int pyo=pxo/dx; pxo%=dx; //oy
	int pco=pyo/dy; pyo%=dy; //fc
	if (di>=pdo) return;
	//float otmp=0.0f;
	for (int py=0;py<fy;py++) if (((uint32)(py+pyo-fy/2))<dy)
		for (int px=0;px<fx;px++) if (((uint32)(px+pxo-fx/2))<dx)
            for (int pc=0;pc<dc;pc++)
			    atomicAdd(&devicedata[(((py+pyo-fy/2)*dx+px+pxo-fx/2)*dc+pc)*blockcount+di],devicefilters[((pco*fy+py)*fx+px)*dc+pc] * deviceout[((pco*dy+pyo)*dx+pxo)*blockcount+di]);
}

__global__ void kconvolution4D(const float*__restrict__ devicedata,const float*__restrict__ devicefilters,float*__restrict__ deviceout,uint32 pdo,uint32 blockcount,uint32 dy,uint32 dx,uint32 dc,uint32 fc,uint32 fy,uint32 fx,uint32 oy,uint32 ox)
{
	int di= ((blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x; //pdo
	if (di>=blockcount*fc*oy*ox) return;
	int pxo=di/blockcount; di%=blockcount; //ox
	int pyo=pxo/ox; pxo%=ox; //oy
	int pco=pyo/oy; pyo%=oy; //fc
	if (di>=pdo) return;
	devicefilters+=pco*fy*fx*dc;
	//const float*filter=&devicefilters[pco*fy*fx*dc];
	float otmp=0.0f;
	for (int py=0;py<fy;py++) {
		const float*dev=&devicedata[((py+pyo)*dx+pxo)*dc*blockcount+di];
		for (int px=fx*dc;px-->0;)
			{otmp+=*devicefilters * *dev; devicefilters++; dev+=blockcount; }
	}
	deviceout[((pco*oy+pyo)*ox+pxo)*blockcount+di]=otmp;
}


__global__ void kreverseconvolution4D_outorder(float*__restrict__ devicedata,const float*__restrict__ devicefilters,const float*__restrict__ deviceout,uint32 pdo,uint32 blockcount,uint32 dc,uint32 fc,uint32 fy,uint32 fx,uint32 oy,uint32 ox)
{
	int di= ((blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x; //pdo
	if (di>=blockcount*dc*(oy+fy-1)*(ox+fx-1)) return;
	int px=di/blockcount; di%=blockcount;
	if (di>=pdo) return;
	int py=px/(ox+fx-1); px%=(ox+fx-1);
	int pc=py/(ox+fy-1); py%=(oy+fy-1);
	float ans=0.0f;
	for (int of=0;of<fc;of++)
		for (int pfy=-fy+1;pfy<=0;pfy++) if (pfy+py>=0&&pfy+py<oy)
			for (int pfx=-fx+1;pfx<=0;pfx++) if (pfx+px>=0&&pfx+px<ox)
				ans+=deviceout[((of*oy+pfy+py)*ox+pfx+px)*blockcount+di]*devicefilters[((pc*fc+of)*fy+fy+pfy-1)*fx+fx+pfx-1];
	devicedata[((py*(ox+fx-1)+px)*dc+pc)*blockcount+di]=ans;
}
__global__ void kgradconvolution4D(const float*__restrict__ devicedata,const float*__restrict__ devicegrad,float*__restrict__ devicefilters,uint32 pdo,uint32 dy,uint32 dx,uint32 dc,uint32 fc,uint32 fy,uint32 fx,uint32 oy,uint32 ox)
{
	int pc=((blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x; //pdo
	if (pc>=fc*fy*fx*dc) return;
	int pfx=pc/dc; pc%=dc;
	int pfy=pfx/fx; pfx%=fx;
	int pco=pfy/fy; pfy%=fy;

	float tmp=0.0f;
	for (int pi=0;pi<pdo;pi++)
		for (int pyo=0;pyo<oy;pyo++)
			for (int pxo=0;pxo<ox;pxo++)
				tmp+=devicegrad[((pi*fc+pco)*oy+pyo)*ox+pxo]*devicedata[((pi*dy+pyo+pfy)*dx+pxo+pfx)*dc+pc];

	devicefilters[((pco*fy+pfy)*fx+pfx)*dc+pc]+=tmp;
}

__global__ void kreverseconvolution4D(float*__restrict__ devicedata,const float*__restrict__ devicefilters,const float*__restrict__ deviceout,uint32 pdo,uint32 blockcount,uint32 dy,uint32 dx,uint32 dc,uint32 fc,uint32 fy,uint32 fx,uint32 oy,uint32 ox)
{
	int di= ((blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x; //pdo
	if (di>=blockcount*fc*oy*ox) return;
	int pxo=di/blockcount; di%=blockcount; //ox
	int pyo=pxo/ox; pxo%=ox; //oy
	int pco=pyo/oy; pyo%=oy; //fc
	if (di>=pdo) return;
	float grad=deviceout[((pco*oy+pyo)*ox+pxo)*blockcount+di];
	const float*filter=&devicefilters[pco*fy*fx*dc];
	//float otmp=0.0f;
	for (int py=0;py<fy;py++) {
		float*dev=&devicedata[((py+pyo)*dx+pxo)*dc*blockcount+di];
		for (int px=fx*dc;px-->0;)
			{atomicAdd(dev,*filter * grad); filter++; dev+=blockcount; }
	}
}

//Assume filter x size < BLOCKTOTAL
#define BLOCKTOTAL 256
//grid: data, dc, thread: BLOCKTOTAL
#define fxd fx
//out: nd fc oy ox
//filters: fc fy fx pc
//rearrange data: nd dc dy dx
__global__ void kreverseconvolution4D_smallfilter(float*__restrict__ devicedata, const float*__restrict__ devicefilters,const float*__restrict__ deviceout,uint32 pdo,uint32 dy,uint32 dx,uint32 dc,uint32 fc,uint32 fy,uint32 fx,uint32 oy,uint32 ox)
{
	__shared__ float out[BLOCKTOTAL];
	__shared__ float filter[BLOCKTOTAL];
	__shared__ float data[BLOCKTOTAL*4];
	int pc=blockIdx.x/pdo;
	int pd=blockIdx.x%pdo;

	__shared__ int fyd;
	__shared__ int BUSEFUL;
	__shared__ int BDATAS;
	if (threadIdx.x==0)
	{
		fyd=BLOCKTOTAL/fxd;
		if (fyd>fy) fyd=fy;
		BUSEFUL=fyd*fxd;
		BDATAS=(fyd*2-1)*(fxd*2-1);
	}
	__syncthreads();

	int tfy=threadIdx.x/fxd;
	int tfx=threadIdx.x%fxd;

	for (int pfy=0;pfy<fy;pfy+=fyd)
		for (int py=0;py<oy;py+=fyd)
			for (int px=0;px<ox;px+=fx)
			{
				//clean data
				data[threadIdx.x]=0;
				data[BLOCKTOTAL+threadIdx.x]=0;
				data[2*BLOCKTOTAL+threadIdx.x]=0;
				data[3*BLOCKTOTAL+threadIdx.x]=0;
				

					for (int pco=0;pco<fc;pco++)
					{
						//fill filter
						if (threadIdx.x<BUSEFUL&&tfy+pfy<fy)
							filter[threadIdx.x]=devicefilters[((pc*fc+pco)*fy+tfy+pfy)*fx+tfx];
						else
							filter[threadIdx.x]=0.0f;
						//fill out
						if (threadIdx.x<BUSEFUL&&py+tfy<oy&&px+tfx<ox)
							out[threadIdx.x]=deviceout[((pd*fc+pco)*oy+py+tfy)*ox+px+tfx];
						else
							out[threadIdx.x]=0.0f;
						__syncthreads();
						//accumulate data
						for (int ly=0;ly<fyd;ly++)
							for (int lx=0;lx<fxd;lx++)
							{
								if (threadIdx.x<BUSEFUL)
									data[(tfy+ly)*(fxd*2-1)+tfx+lx]+=filter[ly*fxd+lx]*out[tfy*fxd+tfx];
								__syncthreads();
							}
					}
				//add out data
				if (threadIdx.x<BUSEFUL)
				{
					int ly=threadIdx.x/(fxd*2-1);
					int lx=threadIdx.x%(fxd*2-1);
					ly+=py+pfy;
					lx+=px;
					if (threadIdx.x<BDATAS&&ly<dy&&lx<dx)
						devicedata[((pd*dc+pc)*dy+ly)*dx+lx]+=data[threadIdx.x];
				
					ly=(threadIdx.x+BUSEFUL)/(fxd*2-1);
					lx=(threadIdx.x+BUSEFUL)%(fxd*2-1);
					ly+=py+pfy;
					lx+=px;
					if (threadIdx.x+BUSEFUL<BDATAS&&ly<dy&&lx<dx)
						devicedata[((pd*dc+pc)*dy+ly)*dx+lx]+=data[BUSEFUL+threadIdx.x];
				
					ly=(threadIdx.x+2*BUSEFUL)/(fxd*2-1);
					lx=(threadIdx.x+2*BUSEFUL)%(fxd*2-1);
					ly+=py+pfy;
					lx+=px;
					if (threadIdx.x+2*BUSEFUL<BDATAS&&ly<dy&&lx<dx)
						devicedata[((pd*dc+pc)*dy+ly)*dx+lx]+=data[2*BUSEFUL+threadIdx.x];
				
					ly=(threadIdx.x+3*BUSEFUL)/(fxd*2-1);
					lx=(threadIdx.x+3*BUSEFUL)%(fxd*2-1);
					ly+=py+pfy;
					lx+=px;
					if (threadIdx.x+3*BUSEFUL<BDATAS&&ly<dy&&lx<dx)
						devicedata[((pd*dc+pc)*dy+ly)*dx+lx]+=data[3*BUSEFUL+threadIdx.x];
				}
				__syncthreads();
			}

}
