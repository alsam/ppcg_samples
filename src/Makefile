
stage:
	ppcg --target=cuda gemvn_ref_impl_x.c --tile-size=128 --max-shared-memory=32768 --shared-memory
	./fix_host gemvn_ref_impl_x_host.cu

build:
	nvcc -O3 --gpu-architecture=sm_35 gemvn_ref_impl_x_host.cu gemvn_ref_impl_x_kernel.cu

profile:
	nvprof --metrics all ./a.out
