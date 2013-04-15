/*
CUDANPLEARN General kernel functions
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

__device__ __constant__ int cy,cx,cc,wc,wyo,wxo,wyp,wxp;
void setconstant(int cy1, int cx1, int cc1, int wc1, int wyo1, int wxo1, int wyp1, int wxp1)
{
	cudaMemcpyToSymbol(cy,&cy1,1*sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(cx,&cx1,1*sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(cc,&cc1,1*sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(wc,&wc1,1*sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(wyo,&wyo1,1*sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(wxo,&wxo1,1*sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(wyp,&wyp1,1*sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(wxp,&wxp1,1*sizeof(int),0,cudaMemcpyHostToDevice);
}

__global__ void tohidden_dataparallel(const float* __restrict__ const data, const float* __restrict__ weight, float* __restrict__ out, int const dalign, int const d) {
	int di = ((blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
	if (di>=d*wc*wyo*wxo) return;
	int pco = di/d; di%=d;
	int pyo = pco/wc; pco%=wc;
	int pxo = pyo/wyo; pyo%=wyo;
	if (di>=d) return;
	const float* __restrict__ pdr;
	weight+=((pco*wyo+pyo)*wxo+pxo)*wyp*wxp*cc;
	{
		float otemp=0;
		for (int py=0;py<wyp;py++)
		{
			pdr = data+( ((py+pyo)*cx+pxo)*cc*dalign+di );
			for (int px=wxp*cc;px--;)
			{
				otemp += *pdr * *weight;
				weight++;
				pdr+=dalign;
			}
		}
		out[((pco*wyo+pyo)*wxo+pxo)*dalign+di] = otemp;
	}
}

__global__ void fromhidden2_dataparallel(const float* __restrict__ const hidden, const float* __restrict__ const weight, float* __restrict__ out, int dalign, int d) {
	int di = ((blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
	if (di>=d*cy*cx*cc) return;
	int pc = di/d; di%=d;
	int px = pc/cc; pc%=cc;
	int py = px/cx; px%=cx;
	float tmp=0;
	for (int pco=0;pco<wc;pco++)
		for (int pyp=0;pyp<wyp;pyp++) { int pyo=py-pyp; if (pyp>py || pyo>=wyo) continue; else
			for (int pxp=0;pxp<wxp;pxp++) { int pxo=px-pxp; if (pxp>px || pxo>=wxo) continue; else
				tmp+=hidden[((pco*wyo+pyo)*wxo+pxo)*dalign+di] * weight[((((pco*wyo+pyo)*wxo+pxo)*wyp+pyp)*wxp+pxp)*cc+pc];
			}
		}
	out[((py*cx+px)*cc+pc)*dalign+di]=tmp;
}

__global__ void extractvalue_dataparallel(const float* __restrict__ const data, const float* __restrict__ const hidden, float* __restrict__ dweight, int d) {
	int pc=((blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
	if (pc>=wc*wyo*wxo*wyp*wxp*cc) return;
	int px=pc/cc; pc%=cc;
	int pxo=px/wxp; px%=wxp;
	int py=pxo/wxo; pxo%=wxo;
	int pyo=py/wyp; py%=wyp;
	int pco=pyo/wyo; pyo%=wyo;
	float otmp=0;
	for (int pd=0;pd<d;pd++)
		otmp+=data[((pd*cy+pyo+py)*cx+pxo+px)*cc+pc]*hidden[((pd*wc+pco)*wyo+pyo)*wxo+pxo];
	dweight[((((pco*wyo+pyo)*wxo+pxo)*wyp+py)*wxp+px)*cc+pc]+=otmp;
}

__global__ void pooling_in_dataparallel(const float*__restrict__ const hiddenin, float*__restrict__ poolingout, int d, int wc, int wyo, int wxo, int poolsize)
{
	int pxo=threadIdx.x+blockDim.x*blockIdx.x;
	int cyo=wyo-poolsize+1;
	int cxo=wxo-poolsize+1;
	if (pxo>=cyo*cxo) return;
	int pyo=pxo/cxo; pxo%=cxo;
	for (int pd=0;pd<d;pd++)
		for (int pc=0;pc<wc;pc++)
		{
			float otemp=0;
			for (int ly=0;ly<poolsize;ly++)
				for (int lx=0;lx<poolsize;lx++)
					otemp+=hiddenin[((pd*wc+pc)*wyo+pyo+ly)*wxo+pxo+lx];
			poolingout[((pd*wc+pc)*cyo+pyo)*cxo+pxo]=otemp;
		}
}

__global__ void pooling_back_dataparallel(const float*__restrict__ const poolingin, float*__restrict__ hiddenout, int d, int wc, int wyo, int wxo, int poolsize)
{
	int pxo=threadIdx.x+blockDim.x*blockIdx.x;
	int cyo=wyo-poolsize+1;
	int cxo=wxo-poolsize+1;
	if (pxo>=cyo*cxo) return;
	int pyo=pxo/cxo; pxo%=cxo;
	for (int pd=0;pd<d;pd++)
		for (int pc=0;pc<wc;pc++)
		{
			float otemp=poolingin[((pd*wc+pc)*cyo+pyo)*cxo+pxo];
			for (int ly=0;ly<poolsize;ly++)
				for (int lx=0;lx<poolsize;lx++)
					atomicAdd(&hiddenout[((pd*wc+pc)*wyo+pyo+ly)*wxo+pxo+lx],otemp);
		}
}
__global__ void kinlayermax(const float*__restrict__ const hiddenin, float*__restrict__ poolingout, int d, int dc, int dy, int dx, int poolsize)
{
	int pxo=threadIdx.x+blockDim.x*blockIdx.x;
	if (pxo>=dy*dx) return;
	int pyo=pxo/dx; pxo%=dx;
	for (int pd=0;pd<d;pd++)
		for (int pc=0;pc<dc;pc++)
		{
			float center=hiddenin[((pd*dc+pc)*dy+pyo)*dx+pxo];
			bool throughflag=true;
			for (int ly=-poolsize;++ly<poolsize;) if (ly+pyo>=0&&ly+pyo<dy)
				for (int lx=-poolsize;++lx<poolsize;) if (lx+pxo>=0&&lx+pxo<dx)
					if ((ly!=0||lx!=0)&&(hiddenin[((pd*dc+pc)*dy+ly+pyo)*dx+lx+pxo]>center))
						throughflag=false;
			poolingout[((pd*dc+pc)*dy+pyo)*dx+pxo]=throughflag?center:0.0f;
		}
}

__global__ void kreverseinlayermax(const float*__restrict__ const hiddenin, const float*__restrict__ const grad, float* hiddengrad, int d, int dc, int dy, int dx, int poolsize)
{
	//TODO
	int pxo=threadIdx.x+blockDim.x*blockIdx.x;
	if (pxo>=dy*dx) return;
	int pyo=pxo/dx; pxo%=dx;
	for (int pd=0;pd<d;pd++)
		for (int pc=0;pc<dc;pc++)
		{
			float center=hiddenin[((pd*dc+pc)*dy+pyo)*dx+pxo];
			bool throughflag=true;
			for (int ly=-poolsize;++ly<poolsize;) if (ly+pyo>=0&&ly+pyo<dy)
				for (int lx=-poolsize;++lx<poolsize;) if (lx+pxo>=0&&lx+pxo<dx)
					if ((ly!=0||lx!=0)&&(hiddenin[((pd*dc+pc)*dy+ly+pyo)*dx+lx+pxo]>center))
						throughflag=false;
			hiddengrad[((pd*dc+pc)*dy+pyo)*dx+pxo]=throughflag?grad[((pd*dc+pc)*dy+pyo)*dx+pxo]:0.0f;
		}
}

__global__ void kalllayermax(const float*__restrict__ const hiddenin, float*__restrict__ poolingout, int d, int dc, int dy, int dx, int poolsize)
{
	int pxo=threadIdx.x+blockDim.x*blockIdx.x;
	if (pxo>=dy*dx*dc) return;
	int pyo=pxo/dx; pxo%=dx;
	int pco=pyo/dy; pyo%=dy;
	for (int pd=0;pd<d;pd++) {
		float center=hiddenin[((pd*dc+pco)*dy+pyo)*dx+pxo];
		bool throughflag=true;
		for (int pc=0;pc<dc;pc++)
			for (int ly=-poolsize;++ly<poolsize;) if (ly+pyo>=0&&ly+pyo<dy)
				for (int lx=-poolsize;++lx<poolsize;) if (lx+pxo>=0&&lx+pxo<dx)
					if ((ly!=0||lx!=0||pc!=pco)&&(hiddenin[((pd*dc+pco)*dy+ly+pyo)*dx+lx+pxo]>center))
						throughflag=false;
		poolingout[((pd*dc+pco)*dy+pyo)*dx+pxo]=throughflag?center:0.0f;
	}
}

__global__ void kreversealllayermax(const float*__restrict__ const hiddenin, const float*__restrict__ const grad, float* hiddengrad, int d, int dc, int dy, int dx, int poolsize)
{
	//TODO
	int pxo=threadIdx.x+blockDim.x*blockIdx.x;
	if (pxo>=dy*dx*dc) return;
	int pyo=pxo/dx; pxo%=dx;
	int pco=pyo/dy; pyo%=dy;
	for (int pd=0;pd<d;pd++) {
		float center=hiddenin[((pd*dc+pco)*dy+pyo)*dx+pxo];
		bool throughflag=true;
		for (int pc=0;pc<dc;pc++)
			for (int ly=-poolsize;++ly<poolsize;) if (ly+pyo>=0&&ly+pyo<dy)
				for (int lx=-poolsize;++lx<poolsize;) if (lx+pxo>=0&&lx+pxo<dx)
					if ((ly!=0||lx!=0||pco!=pc)&&(hiddenin[((pd*dc+pc)*dy+ly+pyo)*dx+lx+pxo]>center))
						throughflag=false;
		hiddengrad[((pd*dc+pco)*dy+pyo)*dx+pxo]=throughflag?grad[((pd*dc+pco)*dy+pyo)*dx+pxo]:0.0f;
	}
}


__global__ void kinlayerabsmax(const float*__restrict__ const hiddenin, float*__restrict__ poolingout, int d, int dc, int dy, int dx, int poolsize)
{
	int pxo=threadIdx.x+blockDim.x*blockIdx.x;
	if (pxo>=dy*dx) return;
	int pyo=pxo/dx; pxo%=dx;
	for (int pd=0;pd<d;pd++)
		for (int pc=0;pc<dc;pc++)
		{
			float center=hiddenin[((pd*dc+pc)*dy+pyo)*dx+pxo];
            float cabs=fabs(center);
			bool throughflag=true;
			for (int ly=-poolsize;++ly<poolsize;) if (ly+pyo>=0&&ly+pyo<dy)
				for (int lx=-poolsize;++lx<poolsize;) if (lx+pxo>=0&&lx+pxo<dx)
					if ((ly!=0||lx!=0)&&(fabs(hiddenin[((pd*dc+pc)*dy+ly+pyo)*dx+lx+pxo])>cabs))
						throughflag=false;
			poolingout[((pd*dc+pc)*dy+pyo)*dx+pxo]=throughflag?center:0.0f;
		}
}

__global__ void kreverseinlayerabsmax(const float*__restrict__ const hiddenin, const float*__restrict__ const grad, float* hiddengrad, int d, int dc, int dy, int dx, int poolsize)
{
	//TODO
	int pxo=threadIdx.x+blockDim.x*blockIdx.x;
	if (pxo>=dy*dx) return;
	int pyo=pxo/dx; pxo%=dx;
	for (int pd=0;pd<d;pd++)
		for (int pc=0;pc<dc;pc++)
		{
			float center=hiddenin[((pd*dc+pc)*dy+pyo)*dx+pxo];
            float cabs=fabs(center);
			bool throughflag=true;
			for (int ly=-poolsize;++ly<poolsize;) if (ly+pyo>=0&&ly+pyo<dy)
				for (int lx=-poolsize;++lx<poolsize;) if (lx+pxo>=0&&lx+pxo<dx)
					if ((ly!=0||lx!=0)&&(fabs(hiddenin[((pd*dc+pc)*dy+ly+pyo)*dx+lx+pxo])>cabs))
						throughflag=false;
			hiddengrad[((pd*dc+pc)*dy+pyo)*dx+pxo]=throughflag?grad[((pd*dc+pc)*dy+pyo)*dx+pxo]:0.0f;
		}
}

__global__ void kalllayerabsmax(const float*__restrict__ const hiddenin, float*__restrict__ poolingout, int d, int dc, int dy, int dx, int poolsize)
{
	int pxo=threadIdx.x+blockDim.x*blockIdx.x;
	if (pxo>=dy*dx*dc) return;
	int pyo=pxo/dx; pxo%=dx;
	int pco=pyo/dy; pyo%=dy;
	for (int pd=0;pd<d;pd++) {
		float center=hiddenin[((pd*dc+pco)*dy+pyo)*dx+pxo];
        float cabs=fabs(center);
		bool throughflag=true;
		for (int pc=0;pc<dc;pc++)
			for (int ly=-poolsize;++ly<poolsize;) if (ly+pyo>=0&&ly+pyo<dy)
				for (int lx=-poolsize;++lx<poolsize;) if (lx+pxo>=0&&lx+pxo<dx)
					if ((ly!=0||lx!=0||pc!=pco)&&(fabs(hiddenin[((pd*dc+pco)*dy+ly+pyo)*dx+lx+pxo])>cabs))
						throughflag=false;
		poolingout[((pd*dc+pco)*dy+pyo)*dx+pxo]=throughflag?center:0.0f;
	}
}

__global__ void kreversealllayerabsmax(const float*__restrict__ const hiddenin, const float*__restrict__ const grad, float* hiddengrad, int d, int dc, int dy, int dx, int poolsize)
{
	//TODO
	int pxo=threadIdx.x+blockDim.x*blockIdx.x;
	if (pxo>=dy*dx*dc) return;
	int pyo=pxo/dx; pxo%=dx;
	int pco=pyo/dy; pyo%=dy;
	for (int pd=0;pd<d;pd++) {
		float center=hiddenin[((pd*dc+pco)*dy+pyo)*dx+pxo];
        float cabs=fabs(center);
		bool throughflag=true;
		for (int pc=0;pc<dc;pc++)
			for (int ly=-poolsize;++ly<poolsize;) if (ly+pyo>=0&&ly+pyo<dy)
				for (int lx=-poolsize;++lx<poolsize;) if (lx+pxo>=0&&lx+pxo<dx)
					if ((ly!=0||lx!=0||pco!=pc)&&(fabs(hiddenin[((pd*dc+pc)*dy+ly+pyo)*dx+lx+pxo])>cabs))
						throughflag=false;
		hiddengrad[((pd*dc+pco)*dy+pyo)*dx+pxo]=throughflag?grad[((pd*dc+pco)*dy+pyo)*dx+pxo]:0.0f;
	}
}


__global__ void kmaxblock2D(const float*__restrict__ devicedata,float*__restrict__ deviceout,uint32 pdo,uint32 blockcount,uint32 dc,uint32 dy,uint32 dx,uint32 oy,uint32 ox,uint32 size)
{
	int pi= ((blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
	if (pi>=blockcount*dc*oy*ox) return;
	int px=pi/blockcount; pi%=blockcount;
	int py=px/ox; px%=ox;
	int pc=py/oy; py%=oy;
	if (pi>=pdo) return;
	px*=size;
	py*=size;

	float data=-1.0/0.0;
	for (int ly=0;ly<size;ly++) if (ly+py<dy)
		for (int lx=0;lx<size;lx++) if (lx+px<dx)
			data=max(data,devicedata[((pc*dy+py+ly)*dx+px+lx)*blockcount+pi]);
	
	deviceout[((pc*oy+py/size)*ox+px/size)*blockcount+pi]=data;
}
__global__ void kreversemaxblock2D(const float*__restrict__ devicedata,const float*__restrict__ devicegrad,float*__restrict__ deviceout,const float*__restrict__ deviceoutdata,uint32 pdo,uint32 blockcount,uint32 dc,uint32 dy,uint32 dx,uint32 oy,uint32 ox,uint32 size)
{
	int pi= ((blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x; //pdo
	if (pi>=blockcount*dc*oy*ox) return;
	int px=pi/blockcount; pi%=blockcount;
	int py=px/ox; px%=ox;
	int pc=py/oy; py%=oy;
	if (pi>=pdo) return;
	float tgrad=devicegrad[((pc*oy+py)*ox+px)*blockcount+pi];
	px*=size;
	py*=size;

	float data;
	if (deviceoutdata==NULL)
	{
		data=-1.0/0.0;
		for (int ly=0;ly<size;ly++) if (ly+py<dy)
			for (int lx=0;lx<size;lx++) if (lx+px<dx)
				data=max(data,devicedata[((pc*dy+py+ly)*dx+px+lx)*blockcount+pi]);
	} else
		data=deviceoutdata[((pc*oy+py/size)*ox+px/size)*blockcount+pi];

	for (int ly=0;ly<size;ly++) if (ly+py<dy)
		for (int lx=0;lx<size;lx++) if (lx+px<dx)
			if (devicedata[((pc*dy+py+ly)*dx+px+lx)*blockcount+pi]==data)
				deviceout[((pc*dy+py+ly)*dx+px+lx)*blockcount+pi]=tgrad;
			else
				deviceout[((pc*dy+py+ly)*dx+px+lx)*blockcount+pi]=0.0f;
}
__global__ void kfollowmaxblock2D(const float*__restrict__ devicedata,const float*__restrict__ devicegrad,float*__restrict__ deviceout,uint32 pdo,uint32 blockcount,uint32 dc,uint32 dy,uint32 dx,uint32 oy,uint32 ox,uint32 size)
{
	int pi= ((blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x; //pdo
	if (pi>=blockcount*dc*oy*ox) return;
	int px=pi/blockcount; pi%=blockcount;
	int py=px/ox; px%=ox;
	int pc=py/oy; py%=oy;
	if (pi>=pdo) return;
	px*=size;
	py*=size;

	float data,out;
	data=-1.0/0.0;
	out=0.0f;
	for (int ly=0;ly<size;ly++) if (ly+py<dy)
		for (int lx=0;lx<size;lx++) if (lx+px<dx)
		{
			float v=devicedata[((pc*dy+py+ly)*dx+px+lx)*blockcount+pi];
			if (v>data)
			{
				data=v;
				out=devicegrad[((pc*dy+py+ly)*dx+px+lx)*blockcount+pi];
			}
		}
	deviceout[((pc*oy+py/size)*ox+px/size)*blockcount+pi]=out;
}
__global__ void ksquareblock2D(const float*__restrict__ devicedata,float*__restrict__ deviceout,uint32 pdo,uint32 blockcount,uint32 dc,uint32 dy,uint32 dx,uint32 oy,uint32 ox,uint32 size)
{
	int pi= ((blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x; //pdo
	if (pi>=blockcount*dc*oy*ox) return;
	int px=pi/blockcount; pi%=blockcount;
	int py=px/ox; px%=ox;
	int pc=py/oy; py%=oy;
	if (pi>=pdo) return;
	px*=size;
	py*=size;

	float data=0.0f;
	for (int ly=0;ly<size;ly++) if (ly+py<dy)
		for (int lx=0;lx<size;lx++) if (lx+px<dx)
		{
			float val=fabs(devicedata[((pc*dy+py+ly)*dx+px+lx)*blockcount+pi])+EPSILON;
			data+=val*val;
		}
	
	deviceout[((pc*oy+py/size)*ox+px/size)*blockcount+pi]=sqrt(data);
}

__global__ void kreversesquareblock2D(const float*__restrict__ devicedata,const float*__restrict__ devicegrad,float*__restrict__ deviceout,const float*__restrict__ deviceoutdata,uint32 pdo,uint32 blockcount,uint32 dc,uint32 dy,uint32 dx,uint32 oy,uint32 ox,uint32 size)
{
	int pi= ((blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x; //pdo
	if (pi>=blockcount*dc*oy*ox) return;
	int px=pi/blockcount; pi%=blockcount;
	int py=px/ox; px%=ox;
	int pc=py/oy; py%=oy;
	if (pi>=pdo) return;
	float tgrad=devicegrad[((pc*oy+py)*ox+px)*blockcount+pi];
	px*=size;
	py*=size;

	float data;
	if (deviceoutdata==NULL)
	{
		data=0.0f;
		for (int ly=0;ly<size;ly++) if (ly+py<dy)
			for (int lx=0;lx<size;lx++) if (lx+px<dx)
			{
				float val=fabs(devicedata[((pc*dy+py+ly)*dx+px+lx)*blockcount+pi])+EPSILON;
				data+=val*val;
			}
		data=sqrt(data);
	} else
		data=deviceoutdata[((pc*oy+py/size)*ox+px/size)*blockcount+pi];
	data=1/data;

	for (int ly=0;ly<size;ly++) if (ly+py<dy)
		for (int lx=0;lx<size;lx++) if (lx+px<dx)
        {
            float val=devicedata[((pc*dy+py+ly)*dx+px+lx)*blockcount+pi];
            if (val>=0) val+=EPSILON; else val-=EPSILON;
			deviceout[((pc*dy+py+ly)*dx+px+lx)*blockcount+pi]=tgrad*data*val;
        }
}
__global__ void kfollowsquareblock2D(const float*__restrict__ devicedata,const float*__restrict__ devicegrad,float*__restrict__ deviceout,const float*__restrict__ deviceoutdata,uint32 pdo,uint32 blockcount,uint32 dc,uint32 dy,uint32 dx,uint32 oy,uint32 ox,uint32 size)
{
	int pi= ((blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x; //pdo
	if (pi>=blockcount*dc*oy*ox) return;
	int px=pi/blockcount; pi%=blockcount;
	int py=px/ox; px%=ox;
	int pc=py/oy; py%=oy;
	if (pi>=pdo) return;
	float tgrad=0.0f;

	float data=1/deviceoutdata[((pc*oy+py)*ox+px)*blockcount+pi];
	px*=size;
	py*=size;

	for (int ly=0;ly<size;ly++) if (ly+py<dy)
		for (int lx=0;lx<size;lx++) if (lx+px<dx)
        {
            float val=devicedata[((pc*dy+py+ly)*dx+px+lx)*blockcount+pi];
            if (val>=0) val+=EPSILON; else val-=EPSILON;
			tgrad+=devicegrad[((pc*dy+py+ly)*dx+px+lx)*blockcount+pi]*data*val;
        }
	deviceout[((pc*oy+py/size)*ox+px/size)*blockcount+pi]=tgrad;
}

// data: [oc [size [len [blockcount(pdo) [ float ]]]]]
// out: [oc [len [blockcount(pdo) [ float ]]]]
__global__ void ksquarelayer(const float*__restrict__ devicedata,float*__restrict__ out,uint32 pdo,uint32 blockcount,uint32 oc,uint32 len,uint32 size)
{
	int id=((blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
	if (id>=blockcount*oc*len) return;
	int lpos=id/blockcount; id%=blockcount;
	int pc=lpos/len; lpos%=len;
	if (id>=pdo) return;
	pc*=size;
	float data=0.0f;
	for (int i=0;i<size;i++)
	{
		float val=fabs(devicedata[((i+pc)*len+lpos)*blockcount+id])+EPSILON;
		data+=val*val;
	}
	out[(pc/size*len+lpos)*blockcount+id]=sqrt(data);
}

__global__ void kreversesquarelayer(const float*__restrict__ devicedata,const float*__restrict__ devicegrad,float*__restrict__ deviceout,const float*__restrict__ deviceoutdata,uint32 pdo,uint32 blockcount,uint32 oc,uint32 len,uint32 size)
{
	int id=((blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
	if (id>=blockcount*oc*len) return;
	int lpos=id/blockcount; id%=blockcount;
	int pc=lpos/len; lpos%=len;
	if (id>=pdo) return;
	float tgrad=devicegrad[(pc*len+lpos)*blockcount+id];
	pc*=size;
	float data;
	if (deviceoutdata==NULL) {
		data=0.0f;
		for (int i=0;i<size;i++)
		{
			float val=fabs(devicedata[((i+pc)*len+lpos)*blockcount+id])+EPSILON;
			data+=val*val;
		}
		data=sqrt(data);
	} else
		data=deviceoutdata[(pc/size*len+lpos)*blockcount+id];
	data=1/data;

	for (int i=0;i<size;i++)
    {
        float val=devicedata[((i+pc)*len+lpos)*blockcount+id];
        if (val>=0) val+=EPSILON; else val-=EPSILON;
		deviceout[((i+pc)*len+lpos)*blockcount+id]=tgrad*data*val;
    }
}
__global__ void kfollowsquarelayer(const float*__restrict__ devicedata,const float*__restrict__ devicegrad,float*__restrict__ deviceout,const float*__restrict__ deviceoutdata,uint32 pdo,uint32 blockcount,uint32 oc,uint32 len,uint32 size)
{
	int id=((blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
	if (id>=blockcount*oc*len) return;
	int lpos=id/blockcount; id%=blockcount;
	int pc=lpos/len; lpos%=len;
	if (id>=pdo) return;
	float tgrad=0.0f;
	pc*=size;
	float data;
	if (deviceoutdata==NULL) {
		data=0.0f;
		for (int i=0;i<size;i++)
		{
			float val=fabs(devicedata[((i+pc)*len+lpos)*blockcount+id])+EPSILON;
			data+=val*val;
		}
		data=sqrt(data);
	} else
		data=deviceoutdata[(pc/size*len+lpos)*blockcount+id];
	data=1/data;

	for (int i=0;i<size;i++)
    {
        float val=devicedata[((i+pc)*len+lpos)*blockcount+id];
        if (val>=0) val+=EPSILON; else val-=EPSILON;
		tgrad+=devicegrad[((i+pc)*len+lpos)*blockcount+id]*data*val;
    }
	deviceout[(pc/size*len+lpos)*blockcount+id]=tgrad;
}

// data: [oc [size [len [blockcount(pdo) [ float ]]]]]
// out: [oc [len [blockcount(pdo) [ float ]]]]
__global__ void kmaxlayer(const float*__restrict__ devicedata,float*__restrict__ out,uint32 pdo,uint32 blockcount,uint32 oc,uint32 len,uint32 size)
{
	int id=((blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
	if (id>=blockcount*oc*len) return;
	int lpos=id/blockcount; id%=blockcount;
	int pc=lpos/len; lpos%=len;
	if (id>=pdo) return;
	pc*=size;
	float data=-1e10f;
	for (int i=0;i<size;i++)
	{
		float val=devicedata[((i+pc)*len+lpos)*blockcount+id];
		if (val>data) data=val;
	}
	out[(pc/size*len+lpos)*blockcount+id]=data;
}

__global__ void kreversemaxlayer(const float*__restrict__ devicedata,const float*__restrict__ devicegrad,float*__restrict__ deviceout,uint32 pdo,uint32 blockcount,uint32 oc,uint32 len,uint32 size)
{
	int id=((blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
	if (id>=blockcount*oc*len) return;
	int lpos=id/blockcount; id%=blockcount;
	int pc=lpos/len; lpos%=len;
	if (id>=pdo) return;
	float tgrad=devicegrad[(pc*len+lpos)*blockcount+id];
	pc*=size;
	float data;
	data=-1.0/0.0;
	for (int i=0;i<size;i++)
	{
		float val=devicedata[((i+pc)*len+lpos)*blockcount+id];
		if (val>data) data=val;
	}

	for (int i=0;i<size;i++)
		if (devicedata[((i+pc)*len+lpos)*blockcount+id]==data)
			deviceout[((i+pc)*len+lpos)*blockcount+id]=tgrad;
		else
			deviceout[((i+pc)*len+lpos)*blockcount+id]=0;
}
__global__ void kfollowmaxlayer(const float*__restrict__ devicedata,const float*__restrict__ devicegrad,float*__restrict__ deviceout,uint32 pdo,uint32 blockcount,uint32 oc,uint32 len,uint32 size)
{
	int id=((blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
	if (id>=blockcount*oc*len) return;
	int lpos=id/blockcount; id%=blockcount;
	int pc=lpos/len; lpos%=len;
	if (id>=pdo) return;
	pc*=size;
	float data;
	data=-1.0/0.0;
	for (int i=0;i<size;i++)
	{
		float val=devicedata[((i+pc)*len+lpos)*blockcount+id];
		if (val>data) data=val;
	}

	for (int i=0;i<size;i++)
		if (devicedata[((i+pc)*len+lpos)*blockcount+id]==data)
			deviceout[(pc/size*len+lpos)*blockcount+id]=devicegrad[((i+pc)*len+lpos)*blockcount+id];
}

__global__ void kaccumDC(float*__restrict__ data,float*__restrict__ counts,int nd,int count)
{
	int id=((blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
	if (id>count) return;
	float tmp=0.0f;
	for (int p=0;p<nd;p++) tmp+=data[p*count+id];
	counts[id]+=tmp;
}
__global__ void kdivi(float*data, float val, int count)
{
	int id=((blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
	if (id>count) return;
	data[id]/=val;
}
__global__ void ksub(float*data,float*sub,int nd,int count)
{
	int id=((blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
	if (id>count) return;
	float tmp=sub[id];
	for (int p=0;p<nd;p++) data[p*count+id]-=tmp;
}
__global__ void normalize_kern(float* __restrict__ weight, int olen, int ilen)
{
	int myid = threadIdx.x+blockIdx.x*blockDim.x;
	if (myid>=olen) return;

	myid*=ilen;
	float sum=0;
	sum=0;
	for (int i=0;i<ilen;i++) sum+=weight[i+myid]*weight[i+myid];
	sum=1.0/sqrt(sum+0.0001);
	for (int i=0;i<ilen;i++) weight[i+myid]*=sum;
}
__device__ unsigned char op_transform[MAX_TRANSFORM];
void __set_op_transform(const unsigned char*operates,int len){cudaMemcpyToSymbol(op_transform,operates,len);}
__global__ void ktransform(const float*__restrict__ d0,const float*__restrict__ d1,const float*__restrict__ d2,const float*__restrict__ d3,float*__restrict__ o,uint64 len,uint32 oplen,uint64 ps)
{
	int id=((blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
	if (id>=len) return;
	float stack[6];
	int stp=-1;
	int chp=0;
	while (chp<oplen)
	{
		switch(op_transform[chp])
		{
		case 0:
			stack[++stp]=1;
			break;
		case 1:
			stack[++stp]=d0[id];
			break;
		case 2:
			stack[++stp]=d1[id];
			break;
		case 3:
			stack[++stp]=d2[id];
			break;
		case 4:
			stack[++stp]=d3[id];
			break;
		case 5:
			stp++;
			((char*)&(stack[stp]))[0]=op_transform[chp+1];
			((char*)&(stack[stp]))[1]=op_transform[chp+2];
			((char*)&(stack[stp]))[2]=op_transform[chp+3];
			((char*)&(stack[stp]))[3]=op_transform[chp+4];
			chp+=4;
			break;
		case 6:
			stack[++stp]=id+ps;
			break;
		case 7:
			stack[stp-1]+=stack[stp];
			stp--;
			break;
		case 8:
			stack[stp-1]=stack[stp]-stack[stp-1];
			stp--;
			break;
		case 9:
			stack[stp]=-stack[stp];
			break;
		case 10:
			stack[stp-1]*=stack[stp];
			stp--;
			break;
		case 11:
			stack[stp-1]=stack[stp]/stack[stp-1];
			stp--;
			break;
		case 12:
			stack[stp-1]=fmod(stack[stp],stack[stp-1]);
			stp--;
			break;
		case 13:
			stack[stp]=sin(stack[stp]);
			break;
		case 14:
			stack[stp]=cos(stack[stp]);
			break;
		case 15:
			stack[stp]=tan(stack[stp]);
			break;
		case 16:
			stack[stp]=1/tan(stack[stp]);
			break;
		case 17:
			stack[stp]=1/cos(stack[stp]);
			break;
		case 18:
			stack[stp]=1/sin(stack[stp]);
			break;
		case 19:
			stack[stp]=1/(stack[stp]+EPSILON);
			break;
		case 20:
			stack[stp]=1+stack[stp];
			break;
		case 21:
			stack[stp]=exp(stack[stp]);
			break;
		case 22:
			stack[stp]=log(stack[stp]);
			break;
		case 23:
			stack[stp]=sinh(stack[stp]);
			break;
		case 24:
			stack[stp]=cosh(stack[stp]);
			break;
		case 25:
			stack[stp]=tanh(stack[stp]);
			break;
		case 26:
			stack[stp]=1/tanh(stack[stp]);
			break;
		case 27:
			stack[stp]=1/cosh(stack[stp]);
			break;
		case 28:
			stack[stp]=1/sinh(stack[stp]);
			break;
		case 29:
			stack[stp]*=stack[stp];
			break;
		case 30:
			stack[stp]=sqrt(stack[stp]);
			break;
		case 31:
			stack[stp-1]=pow(stack[stp],stack[stp-1]);
			stp--;
			break;
		case 32:
			stack[stp]=asin(stack[stp]);
			break;
		case 33:
			stack[stp]=acos(stack[stp]);
			break;
		case 34:
			stack[stp]=atan(stack[stp]);
			break;
		case 35:
			stack[stp]=fabs(stack[stp]);
			break;
		case 36:
			stack[stp]=ceil(stack[stp]);
			break;
		case 37:
			stack[stp]=floor(stack[stp]);
			break;
		case 38:
			stack[stp]=1/(1+exp(-stack[stp]));
			break;
		case 39:
			stack[stp]=stack[stp]*(1-stack[stp]);
			break;
		case 40:
			stack[stp+1]=stack[stp];
			stp++;
			break;
		case 41:
			stack[stp+1]=stack[stp-1];
			stp++;
			break;
		case 42:
			stack[stp+1]=stack[stp-2];
			stp++;
			break;
		case 43:
			stack[stp+1]=stack[stp-3];
			stp++;
			break;
		case 44:
			((int*)stack)[stp-1]=(stack[stp]>stack[stp-1]);
			stp--;
			break;
		case 45:
			((int*)stack)[stp-1]=(stack[stp]<stack[stp-1]);
			stp--;
			break;
		case 46:
			((int*)stack)[stp-1]=(fabs(stack[stp]-stack[stp-1])<1e-6);
			stp--;
			break;
		case 47:
			((int*)stack)[stp-1]=(stack[stp]-stack[stp-1]>-1e-6);
			stp--;
			break;
		case 48:
			((int*)stack)[stp-1]=(stack[stp-1]-stack[stp]>-1e-6);
			stp--;
			break;
		case 49:
			((int*)stack)[stp-1]=(fabs(stack[stp]-stack[stp-1])>1e-6);
			stp--;
			break;
		case 50:
			((int*)stack)[stp-1]=(stack[stp]==stack[stp-1]);
			stp--;
			break;
		case 51:
			((int*)stack)[stp-1]=(((int*)stack)[stp]&&((int*)stack)[stp-1]);
			stp--;
			break;
		case 52:
			((int*)stack)[stp-1]=(((int*)stack)[stp]||((int*)stack)[stp-1]);
			stp--;
			break;
		case 53:
			((int*)stack)[stp-1]=(((int*)stack)[stp]^((int*)stack)[stp-1]);
			stp--;
			break;
		case 54:
			((int*)stack)[stp]=(!((int*)stack)[stp]);
			break;
		case 55:
			stack[stp-2]=(((int*)stack)[stp]?stack[stp-1]:stack[stp-2]);
			stp-=2;
			break;
		case 56:
			if (stack[stp]!=stack[stp-1]) {
				((int*)stack)[stp]^=((int*)stack)[stp-1];
				((int*)stack)[stp-1]^=((int*)stack)[stp];
				((int*)stack)[stp]^=((int*)stack)[stp-1];
			}
			break;
		case 57:
			((int*)stack)[stp]=isnan(stack[stp]);
			break;
		case 58:
			((int*)stack)[stp]=isinf(stack[stp]);
			break;
		}
		chp++;
	}
	o[id]=stack[stp];
}
__global__ void ktransformD(const double*__restrict__ d0,const double*__restrict__ d1,const double*__restrict__ d2,const double*__restrict__ d3,double*__restrict__ o,uint64 len,uint32 oplen,uint64 ps)
{
	int id=((blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
	if (id>=len) return;
	double stack[6];
	int stp=-1;
	int chp=0;
	while (chp<oplen)
	{
		switch(op_transform[chp])
		{
		case 0:
			stack[++stp]=1;
			break;
		case 1:
			stack[++stp]=d0[id];
			break;
		case 2:
			stack[++stp]=d1[id];
			break;
		case 3:
			stack[++stp]=d2[id];
			break;
		case 4:
			stack[++stp]=d3[id];
			break;
		case 5:
			stp++;
			((char*)&(stack[stp]))[0]=op_transform[chp+1];
			((char*)&(stack[stp]))[1]=op_transform[chp+2];
			((char*)&(stack[stp]))[2]=op_transform[chp+3];
			((char*)&(stack[stp]))[3]=op_transform[chp+4];
            ((char*)&(stack[stp]))[4]=op_transform[chp+5];
            ((char*)&(stack[stp]))[5]=op_transform[chp+6];
            ((char*)&(stack[stp]))[6]=op_transform[chp+7];
            ((char*)&(stack[stp]))[7]=op_transform[chp+8];
			chp+=8;
			break;
		case 6:
			stack[++stp]=id+ps;
			break;
		case 7:
			stack[stp-1]+=stack[stp];
			stp--;
			break;
		case 8:
			stack[stp-1]=stack[stp]-stack[stp-1];
			stp--;
			break;
		case 9:
			stack[stp]=-stack[stp];
			break;
		case 10:
			stack[stp-1]*=stack[stp];
			stp--;
			break;
		case 11:
			stack[stp-1]=stack[stp]/stack[stp-1];
			stp--;
			break;
		case 12:
			stack[stp-1]=fmod(stack[stp],stack[stp-1]);
			stp--;
			break;
		case 13:
			stack[stp]=sin(stack[stp]);
			break;
		case 14:
			stack[stp]=cos(stack[stp]);
			break;
		case 15:
			stack[stp]=tan(stack[stp]);
			break;
		case 16:
			stack[stp]=1/tan(stack[stp]);
			break;
		case 17:
			stack[stp]=1/cos(stack[stp]);
			break;
		case 18:
			stack[stp]=1/sin(stack[stp]);
			break;
		case 19:
			stack[stp]=1/(stack[stp]+EPSILON);
			break;
		case 20:
			stack[stp]=1+stack[stp];
			break;
		case 21:
			stack[stp]=exp(stack[stp]);
			break;
		case 22:
			stack[stp]=log(stack[stp]);
			break;
		case 23:
			stack[stp]=sinh(stack[stp]);
			break;
		case 24:
			stack[stp]=cosh(stack[stp]);
			break;
		case 25:
			stack[stp]=tanh(stack[stp]);
			break;
		case 26:
			stack[stp]=1/tanh(stack[stp]);
			break;
		case 27:
			stack[stp]=1/cosh(stack[stp]);
			break;
		case 28:
			stack[stp]=1/sinh(stack[stp]);
			break;
		case 29:
			stack[stp]*=stack[stp];
			break;
		case 30:
			stack[stp]=sqrt(stack[stp]);
			break;
		case 31:
			stack[stp-1]=pow(stack[stp],stack[stp-1]);
			stp--;
			break;
		case 32:
			stack[stp]=asin(stack[stp]);
			break;
		case 33:
			stack[stp]=acos(stack[stp]);
			break;
		case 34:
			stack[stp]=atan(stack[stp]);
			break;
		case 35:
			stack[stp]=fabs(stack[stp]);
			break;
		case 36:
			stack[stp]=ceil(stack[stp]);
			break;
		case 37:
			stack[stp]=floor(stack[stp]);
			break;
		case 38:
			stack[stp]=1/(1+exp(-stack[stp]));
			break;
		case 39:
			stack[stp]=stack[stp]*(1-stack[stp]);
			break;
		case 40:
			stack[stp+1]=stack[stp];
			stp++;
			break;
		case 41:
			stack[stp+1]=stack[stp-1];
			stp++;
			break;
		case 42:
			stack[stp+1]=stack[stp-2];
			stp++;
			break;
		case 43:
			stack[stp+1]=stack[stp-3];
			stp++;
			break;
		case 44:
			((long long*)stack)[stp-1]=(stack[stp]>stack[stp-1]);
			stp--;
			break;
		case 45:
			((long long*)stack)[stp-1]=(stack[stp]<stack[stp-1]);
			stp--;
			break;
		case 46:
			((long long*)stack)[stp-1]=(fabs(stack[stp]-stack[stp-1])<1e-6);
			stp--;
			break;
		case 47:
			((long long*)stack)[stp-1]=(stack[stp]-stack[stp-1]>-1e-6);
			stp--;
			break;
		case 48:
			((long long*)stack)[stp-1]=(stack[stp-1]-stack[stp]>-1e-6);
			stp--;
			break;
		case 49:
			((long long*)stack)[stp-1]=(fabs(stack[stp]-stack[stp-1])>1e-6);
			stp--;
			break;
		case 50:
			((long long*)stack)[stp-1]=(stack[stp]==stack[stp-1]);
			stp--;
			break;
		case 51:
			((long long*)stack)[stp-1]=(((long long*)stack)[stp]&&((long long*)stack)[stp-1]);
			stp--;
			break;
		case 52:
			((long long*)stack)[stp-1]=(((long long*)stack)[stp]||((long long*)stack)[stp-1]);
			stp--;
			break;
		case 53:
			((long long*)stack)[stp-1]=(((long long*)stack)[stp]^((long long*)stack)[stp-1]);
			stp--;
			break;
		case 54:
			((long long*)stack)[stp]=(!((long long*)stack)[stp]);
			break;
		case 55:
			stack[stp-2]=(((long long*)stack)[stp]?stack[stp-1]:stack[stp-2]);
			stp-=2;
			break;
		case 56:
			if (stack[stp]!=stack[stp-1]) {
				((long long*)stack)[stp]^=((long long*)stack)[stp-1];
				((long long*)stack)[stp-1]^=((long long*)stack)[stp];
				((long long*)stack)[stp]^=((long long*)stack)[stp-1];
			}
			break;
		case 57:
			((long long*)stack)[stp]=isnan(stack[stp]);
			break;
		case 58:
			((long long*)stack)[stp]=isinf(stack[stp]);
			break;
		}
		chp++;
	}
	o[id]=stack[stp];
}
__global__ void ktransform2(const float*__restrict__ d0,const float*__restrict__ d1,const float*__restrict__ d2,const float*__restrict__ d3,float*__restrict__ o,float*__restrict__ o2,uint64 len,uint32 oplen,uint64 ps)
{
	int id=((blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
	if (id>=len) return;
	float stack[6];
	int stp=-1;
	int chp=0;
	while (chp<oplen)
	{
		switch(op_transform[chp])
		{
		case 0:
			stack[++stp]=1;
			break;
		case 1:
			stack[++stp]=d0[id];
			break;
		case 2:
			stack[++stp]=d1[id];
			break;
		case 3:
			stack[++stp]=d2[id];
			break;
		case 4:
			stack[++stp]=d3[id];
			break;
		case 5:
			stp++;
			((char*)&(stack[stp]))[0]=op_transform[chp+1];
			((char*)&(stack[stp]))[1]=op_transform[chp+2];
			((char*)&(stack[stp]))[2]=op_transform[chp+3];
			((char*)&(stack[stp]))[3]=op_transform[chp+4];
			chp+=4;
			break;
		case 6:
			stack[++stp]=id+ps;
			break;
		case 7:
			stack[stp-1]+=stack[stp];
			stp--;
			break;
		case 8:
			stack[stp-1]=stack[stp]-stack[stp-1];
			stp--;
			break;
		case 9:
			stack[stp]=-stack[stp];
			break;
		case 10:
			stack[stp-1]*=stack[stp];
			stp--;
			break;
		case 11:
			stack[stp-1]=stack[stp]/stack[stp-1];
			stp--;
			break;
		case 12:
			stack[stp-1]=fmod(stack[stp],stack[stp-1]);
			stp--;
			break;
		case 13:
			stack[stp]=sin(stack[stp]);
			break;
		case 14:
			stack[stp]=cos(stack[stp]);
			break;
		case 15:
			stack[stp]=tan(stack[stp]);
			break;
		case 16:
			stack[stp]=1/tan(stack[stp]);
			break;
		case 17:
			stack[stp]=1/cos(stack[stp]);
			break;
		case 18:
			stack[stp]=1/sin(stack[stp]);
			break;
		case 19:
			stack[stp]=1/(stack[stp]+EPSILON);
			break;
		case 20:
			stack[stp]=1+stack[stp];
			break;
		case 21:
			stack[stp]=exp(stack[stp]);
			break;
		case 22:
			stack[stp]=log(stack[stp]);
			break;
		case 23:
			stack[stp]=sinh(stack[stp]);
			break;
		case 24:
			stack[stp]=cosh(stack[stp]);
			break;
		case 25:
			stack[stp]=tanh(stack[stp]);
			break;
		case 26:
			stack[stp]=1/tanh(stack[stp]);
			break;
		case 27:
			stack[stp]=1/cosh(stack[stp]);
			break;
		case 28:
			stack[stp]=1/sinh(stack[stp]);
			break;
		case 29:
			stack[stp]*=stack[stp];
			break;
		case 30:
			stack[stp]=sqrt(stack[stp]);
			break;
		case 31:
			stack[stp-1]=pow(stack[stp],stack[stp-1]);
			stp--;
			break;
		case 32:
			stack[stp]=asin(stack[stp]);
			break;
		case 33:
			stack[stp]=acos(stack[stp]);
			break;
		case 34:
			stack[stp]=atan(stack[stp]);
			break;
		case 35:
			stack[stp]=fabs(stack[stp]);
			break;
		case 36:
			stack[stp]=ceil(stack[stp]);
			break;
		case 37:
			stack[stp]=floor(stack[stp]);
			break;
		case 38:
			stack[stp]=1/(1+exp(-stack[stp]));
			break;
		case 39:
			stack[stp]=stack[stp]*(1-stack[stp]);
			break;
		case 40:
			stack[stp+1]=stack[stp];
			stp++;
			break;
		case 41:
			stack[stp+1]=stack[stp-1];
			stp++;
			break;
		case 42:
			stack[stp+1]=stack[stp-2];
			stp++;
			break;
		case 43:
			stack[stp+1]=stack[stp-3];
			stp++;
			break;
		case 44:
			((int*)stack)[stp-1]=(stack[stp]>stack[stp-1]);
			stp--;
			break;
		case 45:
			((int*)stack)[stp-1]=(stack[stp]<stack[stp-1]);
			stp--;
			break;
		case 46:
			((int*)stack)[stp-1]=(fabs(stack[stp]-stack[stp-1])<1e-6);
			stp--;
			break;
		case 47:
			((int*)stack)[stp-1]=(stack[stp]-stack[stp-1]>-1e-6);
			stp--;
			break;
		case 48:
			((int*)stack)[stp-1]=(stack[stp-1]-stack[stp]>-1e-6);
			stp--;
			break;
		case 49:
			((int*)stack)[stp-1]=(fabs(stack[stp]-stack[stp-1])>1e-6);
			stp--;
			break;
		case 50:
			((int*)stack)[stp-1]=(stack[stp]==stack[stp-1]);
			stp--;
			break;
		case 51:
			((int*)stack)[stp-1]=(((int*)stack)[stp]&&((int*)stack)[stp-1]);
			stp--;
			break;
		case 52:
			((int*)stack)[stp-1]=(((int*)stack)[stp]||((int*)stack)[stp-1]);
			stp--;
			break;
		case 53:
			((int*)stack)[stp-1]=(((int*)stack)[stp]^((int*)stack)[stp-1]);
			stp--;
			break;
		case 54:
			((int*)stack)[stp]=(!((int*)stack)[stp]);
			break;
		case 55:
			stack[stp-2]=(((int*)stack)[stp]?stack[stp-1]:stack[stp-2]);
			stp-=2;
			break;
		case 56:
			if (stack[stp]!=stack[stp-1]) {
				((int*)stack)[stp]^=((int*)stack)[stp-1];
				((int*)stack)[stp-1]^=((int*)stack)[stp];
				((int*)stack)[stp]^=((int*)stack)[stp-1];
			}
			break;
		case 57:
			((int*)stack)[stp]=isnan(stack[stp]);
			break;
		case 58:
			((int*)stack)[stp]=isinf(stack[stp]);
			break;
		}
		chp++;
	}
	o[id]=stack[stp];
	o2[id]=stack[stp-1];
}
