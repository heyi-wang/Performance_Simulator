.PHONY: all kernels kernel-matmul kernel-dwconv kernel-layernorm kernel-pooling kernel-vecops nafnet clean

all: kernels

kernels:
	$(MAKE) -C kernel all

kernel-matmul:
	$(MAKE) -C kernel matmul

kernel-dwconv:
	$(MAKE) -C kernel dw_conv2d

kernel-layernorm:
	$(MAKE) -C kernel layer_norm

kernel-pooling:
	$(MAKE) -C kernel pooling

kernel-vecops:
	$(MAKE) -C kernel vec_ops

nafnet:
	$(MAKE) -C nafnet all

clean:
	$(MAKE) -C kernel clean
