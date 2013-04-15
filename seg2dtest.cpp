/*
CUDANPLEARN Segment implemenation test file
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

#include "segment2d.h"
#include <stdio.h>
float dat[10][10];
struct OUTRESULT
	{
		void operator() (int dy,int dx,float val)
		{
			dat[dy][dx]=val;
		}
};
int main()
{
	segment2d<float,OUTRESULT> seg;
	seg.init(10,10,0);
	seg.fill(0,5,0,5,1);
	seg.fill(2,3,4,7,2);
	seg.fill(1,8,2,4,3);
	seg.fill(4,100,5,100,4);
	seg.fill(1,2,2,3,5);
	seg.scan();
	for (int i=0;i<10;i++){
		for (int j=0;j<10;j++)
			printf("%d ",(int)dat[i][j]);
		printf("\n");
	}
	return 0;
}
