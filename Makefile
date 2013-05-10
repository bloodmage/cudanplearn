makecudanplearn:
	nvcc -O3 -use_fast_math --ptxas-options=-v --compiler-options '-fPIC' -o cudanplearn.so --shared *.cu -arch=sm_21 --machine=64 -lcudart

clean:
	rm *.linkinfo *.pyc *.so
