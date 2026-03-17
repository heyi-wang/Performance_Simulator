# CLAUDE.md

## Requirements
- After the matrix multiplication of each worker, the partial results on each thread will be processed again by vector accelerator for quantization to the target precision.
- While accumulating, each request sent to the accelerator also produces scalar overhead
- Verify the correctness and execution logic of the simulator
- The simulator should also hold the backpressure mechanism. When the queues at the accelerators are full, worker stalls until free spot on the queue is available.
