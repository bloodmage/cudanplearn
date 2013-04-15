__doc__="""
CUDANPLEARN Test scripts
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

import unittest
import numpy as np
import cudanplearn as learn
class Test(unittest.TestCase):
    def test_conv4D(self):
        data=np.array([
            [[[1,-1],[2,-2],[3,-3]],
            [[4,-4],[5,-5],[6,-6]],
            [[7,-7],[8,-8],[9,-9]]],
            [[[10,-10],[20,-20],[30,-30]],
            [[40,-40],[50,-50],[60,-60]],
            [[70,-70],[80,-80],[90,-90]]]
        ],'f')
        filters=np.array([
            [[[1,-1],[1,-1]],[[1,-1],[1,-1]]],
            [[[-1,1],[-1,1]],[[-1,1],[-1,1]]],
            [[[1,-1],[1,-1]],[[-1,1],[-1,1]]],
            [[[1,-1],[-1,1]],[[1,-1],[-1,1]]]],'f')
        out=learn.empty((2,4,2,2))
        learn.convolution4D(data,filters,out)
        self.assertTrue(np.array_equal(out,np.array([
            [[[24,32],[48,56]],[[-24,-32],[-48,-56]],[[-12,-12],[-12,-12]],[[-4,-4],[-4,-4]]],
            [[[240,320],[480,560]],[[-240,-320],[-480,-560]],[[-120,-120],[-120,-120]],[[-40,-40],[-40,-40]]]
        ],'f')))

    def test_gradconv4D(self):
        data=np.array([
            [[[1,-1],[2,-2],[3,-3]],[[4,-4],[5,-5],[6,-6]],[[7,-7],[8,-8],[9,-9]]],
            [[[10,-10],[20,-20],[30,-30]],[[40,-40],[50,-50],[60,-60]],[[70,-70],[80,-80],[90,-90]]]
        ],'f')
        grad=np.array([
            [[[1,-1],[-1,1]],[[-2,-3],[3,2]],[[-1,3],[-2,1]],[[-1,4],[-4,4]]],
            [[[1,-1],[-1,1]],[[-2,-3],[3,2]],[[-1,3],[-2,1]],[[-1,4],[-4,4]]]
        ],'f')
        out=learn.empty((4,2,2,2))
        learn.gradconvolution4D(data,grad,out)
        self.assertTrue(np.array_equal(out,np.array([
            [[[0,0],[0,0]],[[0,0],[0,0]]],
            [[[154,-154],[154,-154]],[[154,-154],[154,-154]]],
            [[[22,-22],[33,-33]],[[55,-55],[66,-66]]],
            [[[121,-121],[154,-154]],[[220,-220],[253,-253]]]],'f')))

    def test_revconv4D(self):
        filters=np.array([
            [[[1,-1],[1,-1]],[[1,-1],[1,-1]]],
            [[[-1,1],[-1,1]],[[-1,1],[-1,1]]],
            [[[1,-1],[1,-1]],[[-1,1],[-1,1]]],
            [[[1,-1],[-1,1]],[[1,-1],[-1,1]]]],'f')
        grad=np.array([
            [[[1,-1],[-1,1]],[[-2,-3],[3,2]],[[-1,3],[-2,1]],[[-1,4],[-4,4]]],
            [[[0,-1],[-1,1]],[[-2,-3],[3,2]],[[-1,3],[-2,1]],[[-1,4],[-4,4]]]
        ],'f')
        data=learn.empty((2,3,3,2))
        learn.reverseconvolution4D(grad,filters,data)
        self.assertTrue(np.array_equal(data,np.array([
            [[[1,-1],[12,-12],[1,-1]],
             [[-7,7],[10,-10],[-9,9]],
             [[-6,6],[4,-4],[-6,6]]],
            [[[0,0],[11,-11],[1,-1]],
             [[-8,8],[9,-9],[-9,9]],
             [[-6,6],[4,-4],[-6,6]]]])))

    def test_tohidden(self):
        data=np.array([
            [[[1,-1],[2,-2],[3,-3]],
            [[4,-4],[5,-5],[6,-6]],
            [[7,-7],[8,-8],[9,-9]]],
            [[[10,-10],[20,-20],[30,-30]],
            [[40,-40],[50,-50],[60,-60]],
            [[70,-70],[80,-80],[90,-90]]]
        ],'f')
        filters=np.array(
            [[[
                [[[1,1],[1,0],[1,-1]],
                 [[0,1],[0,0],[0,-1]],
                 [[-1,1],[-1,0],[-1,-1]]]
            ]]],'f')
        out=learn.empty((2,1,1,1))
        learn.tohidden_noconv4D(data,filters,out)
        self.assertTrue(np.array_equal(out,np.array([[[[-12]]],[[[-120]]]],'f')))

    def test_fromhidden(self):
        hidden=np.array([[[[1]]],[[[10]]]],'f')
        filters=np.array(
            [[[
                [[[1,1],[1,0],[1,-1]],
                 [[0,1],[0,0],[0,-1]],
                 [[-1,1],[-1,0],[-1,-1]]]
            ]]],'f')
        data=learn.empty((2,3,3,2))
        learn.fromhidden_noconv4D(hidden,filters,data)
        self.assertTrue(np.array_equal(data,np.array([
           [[[1,1],[1,0],[1,-1]],
            [[0,1],[0,0],[0,-1]],
            [[-1,1],[-1,0],[-1,-1]]],
           [[[10,10],[10,0],[10,-10]],
            [[0,10],[0,0],[0,-10]],
            [[-10,10],[-10,0],[-10,-10]]]],'f')))

    def test_gradnoconv(self):
        data=np.array([
            [[[1,-1],[2,-2],[3,-3]],
            [[4,-4],[5,-5],[6,-6]],
            [[7,-7],[8,-8],[9,-9]]],
            [[[10,-10],[20,-20],[30,-30]],
            [[40,-40],[50,-50],[60,-60]],
            [[70,-70],[80,-80],[90,-90]]]
        ],'f')
        grad=np.array([[[[1]]],[[[1]]]],'f')
        filters=learn.empty((1,1,1,3,3,2))
        learn.grad_noconv4D(data,grad,filters)
        self.assertTrue(np.array_equal(filters,np.array(
            [[[[[[11,-11],[22,-22],[33,-33]],
                [[44,-44],[55,-55],[66,-66]],
                [[77,-77],[88,-88],[99,-99]]]]]],'f')))                

    def test_linearpoolin(self):
        data=np.array([[
            [[1,2,3,4],
             [5,6,7,8],
             [9,10,11,12],
             [13,14,15,16]]]],'f')
        out=learn.empty((1,1,2,2))
        learn.linearpoolin(data,out)
        self.assertTrue(np.array_equal(out,np.array([[[[54,63],[90,99]]]],'f')))

    def test_linearpoolout(self):
        data=np.array([[
            [[1,2],
             [3,4]]]],'f')
        out=learn.empty((1,1,4,4))
        learn.linearpoolout(data,out)
        self.assertTrue(np.array_equal(out,np.array([[[
            [1,3,3,2],
            [4,10,10,6],
            [4,10,10,6],
            [3,7,7,4]]]],'f')))

    def test_maxblock2D(self):
        data=np.array([[
            [[1,2,3,4],
             [5,6,7,8],
             [9,10,11,12],
             [13,14,15,16]]]],'f')
        out=learn.empty((1,1,2,2))
        learn.maxblock2D(data,out,2)
        self.assertTrue(np.array_equal(out,np.array([[[[6,8],[14,16]]]],'f')))
    
    def test_revmax2D(self):
        data=np.array([[
            [[1,2,3,4],
             [5,6,7,8],
             [9,10,11,12],
             [13,14,15,16]]]],'f')
        grad=np.array([[[[1,2],[3,4]]]],'f')
        out=learn.empty((1,1,4,4))
        learn.reversemaxblock2D(grad,data,out,2)
        self.assertTrue(np.array_equal(out,np.array([[[[0,0,0,0],[0,1,0,2],[0,0,0,0],[0,3,0,4]]]],'f')))

    def test_forwardmax2D(self):
        data=np.array([[
            [[1,2,3,4],
             [5,6,7,8],
             [9,10,11,12],
             [13,14,15,16]]]],'f')
        grad=-data
        out=learn.empty((1,1,2,2))
        learn.followmaxblock2D(grad,data,out,2)
        self.assertTrue(np.array_equal(out,np.array([[[[-6,-8],[-14,-16]]]],'f')))

    def test_squareblock2D(self):
        data=np.array([[
            [[1,2,3,4],
             [5,6,7,8],
             [9,10,11,12],
             [13,14,15,16]]]],'f')
        out=learn.empty((1,1,2,2))
        learn.squareblock2D(data,out,2)
        self.assertTrue(np.allclose(out,np.array([[[[8.1240387,11.74734211],[23.36664391,27.31299973]]]],'f')))
    
    def test_revsqr2D(self):
        data=np.array([[
            [[1,2,3,4],
             [5,6,7,8],
             [9,10,11,12],
             [13,14,15,16]]]],'f')
        outd=np.array([[[[8.1240387,11.74734211],[23.36664391,27.31299973]]]],'f')
        grad=np.array([[[[1,1],[1,1]]]],'f')
        out=learn.empty((1,1,4,4))
        learn.reversesquareblock2D(grad,data,outd,out,2)
        self.assertTrue(np.allclose(out,np.array(
            [[[[ 0.12309149,  0.24618298,  0.25537694,  0.34050256],
               [ 0.61545742,  0.73854893,  0.5958795,   0.68100512],
               [ 0.38516444,  0.42796049,  0.40273863,  0.43935126],
               [ 0.55634862,  0.5991447,   0.54918903,  0.58580166]]]]
                ,'f')))

    def test_forwardsqr2D(self):
        data=np.array([[
            [[1,2,3,4],
             [5,6,7,8],
             [9,10,11,12],
             [13,14,15,16]]]],'f')
        outd=np.array([[[[8.1240387,11.74734211],[23.36664391,27.31299973]]]],'f')
        grad=np.ones((1,1,4,4),'f')
        out=learn.empty((1,1,2,2))
        learn.followsquareblock2D(grad,data,outd,out,2)
        self.assertTrue(np.allclose(out,np.array([[[[1.72328091,1.87276411],[1.96861827,1.97708058]]]],'f')))
    
    def test_sqrlayer(self):
        data=np.array([
           [[[1,-1]],
            [[0,0]]],
           [[[1,-1]],
            [[2,-2]]]],'f')
        out=learn.empty((2,1,1,2))
        learn.squarelayer(data,out,2)
        self.assertTrue(np.allclose(out,np.array([[[[1,1]]],[[[2.23607039,2.23607039]]]],'f')))

    def test_revsqrlayer(self):
        grad=np.array([[[[1,2]]],[[[3,4]]]],'f')
        data=np.array([
           [[[1,-1]],
            [[0,0]]],
           [[[1,-1]],
            [[2,-2]]]],'f')
        outdata=np.array([[[[1,1]]],[[[2.23607039,2.23607039]]]],'f')
        out=learn.empty((2,2,1,2))
        learn.reversesquarelayer(grad,data,outdata,out,2)
        self.assertTrue(np.allclose(out,np.array([[[[1,-2]],[[0,0]]],[[[1.34163928,-1.78885245]],[[2.68327856,-3.57770491]]]],'f')))

    def test_forwardsqr2D(self):
        data=np.array([
           [[[1,-1]],
            [[0,0]]],
           [[[1,-1]],
            [[2,-2]]]],'f')
        outdata=np.array([[[[1,1]]],[[[2.23607039,2.23607039]]]],'f')
        grad=np.ones((2,2,1,2),'f')
        out=learn.empty((2,1,1,2))
        learn.followsquarelayer(grad,data,outdata,out,2)
        self.assertTrue(np.allclose(out,np.array([[[[1,-1]]],[[[1.34163928,-1.34163928]]]],'f')))

    def test_permutation(self):
        a0=np.array([[1,2],[3,4],[5,6],[7,8]],'f')
        learn.blockpermutation(a0,[0,2,1,3])
        self.assertTrue(np.array_equal(a0,np.array([[1,2],[5,6],[3,4],[7,8]],'f')))

    def test_transform(self):
        a0=np.array([-1000,1000],'f')
        o=learn.empty((2,))
        learn.transform('a sigmoid',o,a0)
        self.assertTrue(np.allclose(o,np.array([0,1],'f')))

    def test_setmemory(self):
        learn.setMemory(16777216)

if __name__=='__main__':
    unittest.main()
