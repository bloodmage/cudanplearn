/*
CUDANPLEARN Specialized max-scan functions
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
#include "segment2d.h"
#include <algorithm>
#include <utility>
#include <limits>
using namespace std;
extern "C" {
	struct _INLAYER_FILL
	{
		float*data;
		float*in;
		float*out;
		int dy,dx;
		void operator() (int py,int px,float val)
		{
			//printf("(%d,%d): %f\n",px,py,val);
			if (data[py*dx+px]==val) out[py*dx+px]=in[py*dx+px]; else out[py*dx+px]=0.0f;
		}
	};
	typedef pair<int,int> locator;
	struct _INLAYER_SORT
	{
		int dy,dx;
		float*data;
		bool operator() (const locator&a,const locator&b)
		{
			return data[a.first*dx+a.second]<data[b.first*dx+b.second];
		}
	};
	DLL extern int cpu_inlayermax_keepshape(float*hiddenin, float*poolingout, int d, int dc, int dy, int dx, int poolsize)
	{
		_INLAYER_FILL lf;
		_INLAYER_SORT sf;
		sf.dy=lf.dy=dy;
		sf.dx=lf.dx=dx;
		segment2d<float,_INLAYER_FILL> filler;
		filler.setF(&lf);
		int dsize=dy*dx;
		locator*locs=(locator*)malloc(dsize*sizeof(locator));
		for (int py=0;py<dy;py++)
			for (int px=0;px<dx;px++)
				locs[py*dx+px]=make_pair(py,px);
		for (int pd=0;pd<d;pd++)
		{
			for (int pc=0;pc<dc;pc++)
			{
				sf.data=lf.data=lf.in=&hiddenin[(pd*dc+pc)*dsize];
				lf.out=&poolingout[(pd*dc+pc)*dsize];
				sort(locs,locs+dsize,sf);
				filler.init(dy,dx,-numeric_limits<float>::infinity());
				for (int p=0;p<dsize;p++)
					filler.fill(locs[p].first-poolsize+1,locs[p].first+poolsize,locs[p].second-poolsize+1,locs[p].second+poolsize,sf.data[locs[p].first*dx+locs[p].second]);
				filler.scan();
			}
		}
		free(locs);
		filler.destroy();
		return 0;
	}
	DLL extern int cpu_reverseinlayermax_keepshape(float*hiddenin, float*grad, float*outgrad, int d, int dc, int dy, int dx, int poolsize)
	{
		_INLAYER_FILL lf;
		_INLAYER_SORT sf;
		sf.dy=lf.dy=dy;
		sf.dx=lf.dx=dx;
		segment2d<float,_INLAYER_FILL> filler;
		filler.setF(&lf);
		int dsize=dy*dx;
		locator*locs=(locator*)malloc(dsize*sizeof(locator));
		for (int py=0;py<dy;py++)
			for (int px=0;px<dx;px++)
				locs[py*dx+px]=make_pair(py,px);

		for (int pd=0;pd<d;pd++)
		{
			for (int pc=0;pc<dc;pc++)
			{
				sf.data=lf.data=&hiddenin[(pd*dc+pc)*dsize];
				lf.in=&grad[(pd*dc+pc)*dsize];
				lf.out=&outgrad[(pd*dc+pc)*dsize];
				sort(locs,locs+dsize,sf);
				filler.init(dy,dx,-numeric_limits<float>::infinity());
				for (int p=0;p<dsize;p++)
					filler.fill(locs[p].first-poolsize+1,locs[p].first+poolsize,locs[p].second-poolsize+1,locs[p].second+poolsize,sf.data[locs[p].first*dx+locs[p].second]);
				filler.scan();
			}
		}
		free(locs);
		filler.destroy();
		return 0;
	}
	DLL extern int cpu_alllayermax_keepshape(float*hiddenin, float*poolingout, int d, int dc, int dy, int dx, int poolsize)
	{
		_INLAYER_FILL lf;
		_INLAYER_SORT sf;
		sf.dy=lf.dy=dy;
		sf.dx=lf.dx=dx;
		segment2d<float,_INLAYER_FILL> filler;
		filler.setF(&lf);
		int dsize=dy*dx;
		locator*locs=(locator*)malloc(dsize*sizeof(locator));
		for (int py=0;py<dy;py++)
			for (int px=0;px<dx;px++)
				locs[py*dx+px]=make_pair(py,px);
		float*layermax=(float*)malloc(dsize*sizeof(float));
		for (int pd=0;pd<d;pd++)
		{
			//Compress to one layer
			for (int p=0;p<dsize;p++) layermax[p]=-numeric_limits<float>::infinity();
			float*hpx=&hiddenin[pd*dc*dy*dx],*hpz;
			for (int pc=0;pc<dc;pc++)
				for (int py=0;py<dy;py++)
					for (int px=0;px<dx;px++)
					{
						if (*hpx>layermax[py*dx+px])
							layermax[py*dx+px]=*hpx;
						hpx++;
					}

			sf.data=lf.data=lf.in=layermax;
			lf.out=layermax;
			sort(locs,locs+dsize,sf);
			filler.init(dy,dx,-numeric_limits<float>::infinity());
			for (int p=0;p<dsize;p++)
				filler.fill(locs[p].first-poolsize+1,locs[p].first+poolsize,locs[p].second-poolsize+1,locs[p].second+poolsize,sf.data[locs[p].first*dx+locs[p].second]);
			filler.scan();

			//Expand to all layers
			hpx=&hiddenin[pd*dc*dy*dx];
			hpz=&poolingout[pd*dc*dy*dx];
			for (int pc=0;pc<dc;pc++)
				for (int py=0;py<dy;py++)
					for (int px=0;px<dx;px++)
					{
						if (*hpx==layermax[py*dx+px])
							*hpz=*hpx;
						else
							*hpz=0.0f;
						hpx++;
						hpz++;
					}
		}
		free(locs);
		free(layermax);
		filler.destroy();
		return 0;
	}
	DLL extern int cpu_reversealllayermax_keepshape(float*hiddenin, float*grad, float*outgrad, int d, int dc, int dy, int dx, int poolsize)
	{
		_INLAYER_FILL lf;
		_INLAYER_SORT sf;
		sf.dy=lf.dy=dy;
		sf.dx=lf.dx=dx;
		segment2d<float,_INLAYER_FILL> filler;
		filler.setF(&lf);
		int dsize=dy*dx;
		locator*locs=(locator*)malloc(dsize*sizeof(locator));
		for (int py=0;py<dy;py++)
			for (int px=0;px<dx;px++)
				locs[py*dx+px]=make_pair(py,px);
		float*layermax=(float*)malloc(dsize*sizeof(float));
		for (int pd=0;pd<d;pd++)
		{
			//Compress to one layer
			for (int p=0;p<dsize;p++) layermax[p]=-numeric_limits<float>::infinity();
			float*hpx=&hiddenin[pd*dc*dy*dx],*hpy,*hpz;
			for (int pc=0;pc<dc;pc++)
				for (int py=0;py<dy;py++)
					for (int px=0;px<dx;px++)
					{
						if (*hpx>layermax[py*dx+px])
							layermax[py*dx+px]=*hpx;
						hpx++;
					}

			sf.data=lf.data=lf.in=layermax;
			lf.out=layermax;
			sort(locs,locs+dsize,sf);
			filler.init(dy,dx,-numeric_limits<float>::infinity());
			for (int p=0;p<dsize;p++)
				filler.fill(locs[p].first-poolsize+1,locs[p].first+poolsize,locs[p].second-poolsize+1,locs[p].second+poolsize,sf.data[locs[p].first*dx+locs[p].second]);
			filler.scan();

			//Expand to all layers
			hpx=&hiddenin[pd*dc*dy*dx];
			hpy=&grad[pd*dc*dy*dx];
			hpz=&outgrad[pd*dc*dy*dx];
			for (int pc=0;pc<dc;pc++)
				for (int py=0;py<dy;py++)
					for (int px=0;px<dx;px++)
					{
						if (*hpx==layermax[py*dx+px])
							*hpz=*hpy;
						else
							*hpz=0.0f;
						hpx++;
						hpy++;
						hpz++;
					}
		}
		free(locs);
		free(layermax);
		filler.destroy();
		return 0;
	}
}
