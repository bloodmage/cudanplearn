/*
CUDANPLEARN Constants
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

#include "npconstant.cuh"
#include "nplearn.cuh"
#include <cuda_runtime.h>
#include <string.h>
unsigned int CUDAMemory=16777216; //16M
unsigned int CUDALines=256; //256 Lines
unsigned int CUDAAligns=64; //Align r/w to 245byte

char errstr[1024];
char custom[1024];

extern "C" {
	DLL extern int get_CUDAMemory(){return CUDAMemory;}
	DLL extern void set_CUDAMemory(int val){CUDAMemory=val;}
	DLL extern int get_CUDALines(){return CUDALines;}
	DLL extern void set_CUDALines(int val){CUDALines=val;}
	DLL extern int get_CUDAAligns(){return CUDAAligns;}
	DLL extern void set_CUDAAligns(int val){CUDAAligns=val;}
	DLL extern char* get_ErrorString(){return errstr;}
	DLL extern void set_CUDACore(int coreid){cudaSetDevice(coreid);}
	DLL extern int get_CUDACores(){int num_devices;cudaGetDeviceCount(&num_devices); return num_devices;}
}
