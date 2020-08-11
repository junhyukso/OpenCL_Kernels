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
|  |numpy_sum(CPU)  |CLBlast(GPU)| my_kernel(GPU) | speed-up (my_kernel / numpy_sum)  
|--|--|--|--|--| 
| Time (ms) | 48.46 |4.10+α|1.15|42.1 x

it is even 3.5x(at least) times faster that famous CLBLAS Library ! (CLBlast : https://github.com/CNugteren/CLBlast)

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
