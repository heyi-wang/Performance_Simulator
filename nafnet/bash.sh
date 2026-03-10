#!/bin/bash
# gcc -pthread nafnet_inference.c ./conv_tiling/src/Conv.c ./conv_tiling/src/Conv_para.c ./conv_tiling/src/Conv_para_dynamic.c -o nafnet_inference

# gcc -o3 -pthread nafnet_inference.c ./conv_tiling/src/Conv.c ./conv_tiling/src/conv_worker.c ./conv_tiling/src/thread_pool.c -o nafnet_inference

gcc -lpthread nafnet_inference.c ./conv_tiling/src/Conv.c ./conv_tiling/src/conv_worker.c ./conv_tiling/src/conv_worker_cw.c ./conv_tiling/src/thread_pool.c -o nafnet_inference
