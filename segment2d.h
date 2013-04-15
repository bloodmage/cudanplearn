/*
CUDANPLEARN Segment implemenation
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

#ifndef _SEGMENT2D_H_
#define _SEGMENT2D_H_
#include <stdlib.h>
#include <memory.h>
template<typename T>
struct segelem {
	T val;
	int plu,pld,pru,prd;
};

template<typename T, typename F>
struct segment2d
{
	segelem<T>*space;
	int recycptr,alloclen,heaptop;
	int ymax,xmax;
	segment2d(){alloclen=-1; space=NULL; recycptr=0; heaptop=0; fobj=NULL;}
	void destroy(){free(space);}
	F*fobj;
	void setF(F*newf){fobj=newf;}
	void init(int ysize, int xsize,T val){
		if (space==NULL||alloclen<ysize*xsize*5)
		{
			int newlen=ysize*xsize*5;
			segelem<T>*nspace=(segelem<T>*)malloc(newlen*sizeof(segelem<T>));
			if (space!=NULL)
				memcpy(nspace,space,alloclen*sizeof(segelem<T>));
			alloclen=newlen;
			space=nspace;
			//printf("Realloced %d\n",alloclen);
		}
		recycptr=-1; heaptop=0;
		//All elements node
		space[0].val=val;
		space[0].pld=space[0].plu=space[0].prd=space[0].pru=-1;
		ymax=ysize;
		xmax=xsize;
	}
	//Simple reuse mechanism
	int allocelem(T val){
		int nptr;
		if (recycptr!=-1)
		{
			nptr=recycptr;
			recycptr=space[recycptr].plu;
		}
		else
		{
			nptr=++heaptop;
			if (heaptop==alloclen)
			{
				//Expand
				int newlen=alloclen*3/2;
				segelem<T>*nspace=(segelem<T>*)malloc(newlen*sizeof(segelem<T>));
				if (space!=NULL)
					memcpy(nspace,space,alloclen*sizeof(segelem<T>));
				alloclen=newlen;
				space=nspace;

				//printf("Realloced %d\n",alloclen);
			}
		}
		space[nptr].plu=space[nptr].pld=space[nptr].pru=space[nptr].prd=-1;
		space[nptr].val=val;
		return nptr;
	}
	void freeelem(int posi){
		if (recycptr==-1) {recycptr=posi; space[posi].plu=-1;}
		else {
			space[posi].plu=recycptr;
			recycptr=posi;
		}
	}
	//Helper function
	void delposi(int posi){
		if (space[posi].plu!=-1) delposi(space[posi].plu);
		if (space[posi].pld!=-1) delposi(space[posi].pld);
		if (space[posi].pru!=-1) delposi(space[posi].pru);
		if (space[posi].prd!=-1) delposi(space[posi].prd);
		freeelem(posi);
	}
	void fill(int yst,int yed,int xst,int xed,T val){
		fill(0,0,0,ymax,xmax,yst,yed,xst,xed,val);
	}
	void fill(int posi,int ymin,int xmin,int ymax,int xmax,int yst,int yed,int xst,int xed,T val) {
		if (ymin>=yst&&ymax<=yed&&xmin>=xst&&xmax<=xed)
		{
			//printf("FILL (%d,%d)-(%d,%d) in (%d,%d)-(%d,%d) [%d] with %f\n",xmin,ymin,xmax,ymax,xst,yst,xed,yed,posi,val);
			space[posi].val=val;
			if (space[posi].plu!=-1) {delposi(space[posi].plu); space[posi].plu=-1;}
			if (space[posi].pld!=-1) {delposi(space[posi].pld); space[posi].pld=-1;}
			if (space[posi].pru!=-1) {delposi(space[posi].pru); space[posi].pru=-1;}
			if (space[posi].prd!=-1) {delposi(space[posi].prd); space[posi].prd=-1;}
		} else {
			//printf("split (%d,%d)-(%d,%d) in (%d,%d)-(%d,%d) [%d] with %f\n",xmin,ymin,xmax,ymax,xst,yst,xed,yed,posi,val);
			int xsplit=(xmin+xmax)/2;
			int ysplit=(ymin+ymax)/2;
			int txm,txM,tym,tyM;
			txm=xmin; txM=xsplit; tym=ymin; tyM=ysplit;
			if (txm!=txM&&tym!=tyM) {
				if (space[posi].plu==-1) space[posi].plu=allocelem(space[posi].val);
				if (txM>xst&&tyM>yst) fill(space[posi].plu,tym,txm,tyM,txM,yst,yed,xst,xed,val);
			}
			txm=xsplit; txM=xmax;
			if (txm!=txM&&tym!=tyM) {
				if (space[posi].pru==-1) space[posi].pru=allocelem(space[posi].val);
				if (txm<xed&&tyM>yst) fill(space[posi].pru,tym,txm,tyM,txM,yst,yed,xst,xed,val);
			}
			txm=xmin; txM=xsplit; tym=ysplit; tyM=ymax;
			if (txm!=txM&&tym!=tyM) {
				if (space[posi].pld==-1) space[posi].pld=allocelem(space[posi].val);
				if (txM>xst&&tym<yed) fill(space[posi].pld,tym,txm,tyM,txM,yst,yed,xst,xed,val);
			}
			txm=xsplit; txM=xmax;
			if (txm!=txM&&tym!=tyM) {
				if (space[posi].prd==-1) space[posi].prd=allocelem(space[posi].val);
				if (txm<xed&&tym<yed) fill(space[posi].prd,tym,txm,tyM,txM,yst,yed,xst,xed,val);
			}
		}
	}
	void scan(){ scan(0,0,0,xmax,ymax); }
	void scan(int posi,int xmin,int ymin,int xmax,int ymax) {
		bool direct=true;
		if (space[posi].plu!=-1) { scan(space[posi].plu,xmin,ymin,(xmin+xmax)/2,(ymin+ymax)/2); direct=false; }
		if (space[posi].pru!=-1) { scan(space[posi].pru,(xmin+xmax)/2,ymin,xmax,(ymin+ymax)/2); direct=false; }
		if (space[posi].pld!=-1) { scan(space[posi].pld,xmin,(ymin+ymax)/2,(xmin+xmax)/2,ymax); direct=false; }
		if (space[posi].prd!=-1) { scan(space[posi].prd,(xmin+xmax)/2,(ymin+ymax)/2,xmax,ymax); direct=false; }
		if (direct)
		{
			T val=space[posi].val;
			for (int py=ymin;py<ymax;py++)
				for (int px=xmin;px<xmax;px++)
					(*fobj)(py,px,val);
		}
	}
};

#endif
