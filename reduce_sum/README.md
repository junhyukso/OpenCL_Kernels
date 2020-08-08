# Reduce sum

## Purpose of kernel
This kernel is used to add a vector made of float32 as a single value.

## What Actually Do
This kernel reduces vectors with N elements to vectors with 224 elements. (in GPU or Other accelerators..)
The remaining reduce operations are performed on the CPU.

## Main Techniques
I applied the following techniques for optimization.
 - Use Local Memory (shared memory)
 - Add during load
 - Unroll loop (Consider Warp)
 - Limit thread

## Performance
reduce sum of 1.0*10^9 length Vector
| method |numpy_sum(CPU)  | my_kernel(GPU) | speed-up 
|--|--|--|--|
| Time (ms) | 48.46 |1.15|42.1 x


Tested on

 - CPU : Intel(R) Xeon(R) CPU E5-2630 0 @ 2.30GHz
 - GPU : NVIDIA GTX 1080ti

## TODO

 - [ ] Clean code
 - [ ] support various warp size
 - [ ] support no-local-memory-gpu (ex.mali)
 - [ ] Experiment results on various GPU

## Reference
https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
