#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pyopencl as cl
import math
import matplotlib.pylab as plt

#
N = 100000000
x_np = np.random.rand(N).astype(np.float32)
print(x_np)
xsum = np.sum(x_np)
print(f'initial cpu sum :  {xsum}')
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx,
        properties = cl.command_queue_properties.PROFILING_ENABLE ) #PROFILING ENABLE for time measure.


mf = cl.mem_flags
x_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x_np) #if hostbuf specificd, size default to its size

prg = cl.Program(ctx, """
#define VS 4
#define PADDING_VAL 0

//This kernel is written for 256 work-group-size
__kernel void sasum(const int N,
                   const __global float* X, const int indexX, 
									 __global float* Y,__local float* local_data) {

  X += indexX;

  const unsigned int gid = get_global_id(0);
  const unsigned int gsz = get_global_size(0);
  const unsigned int loc_id = get_local_id(0);
  const unsigned int lsz = get_local_size(0);
  
  ////////////////LOAD START///////////////////////////////////////
  //load to local memory, first add during load.
  /*
  unsigned int load_id_1 = gid;
  unsigned int load_id_2;  
  while(load_id_1 < N){
    load_id_2 = gid+lsz;

    float load_data_1 = (load_id_1 < N) ? X[load_id_1] : PADDING_VAL;
    float load_data_2 = (load_id_2 < N) ? X[load_id_2] : PADDING_VAL;

    local_data[loc_id] = fabs(load_data_1) + fabs(load_data_2);
    load_id_1 += gsz*2;
  }
  */
  /*
  int load_id_1 = gid;
  while(load_id_1 < N){
    local_data[loc_id] = 0;
    local_data[loc_id] += X[load_id_1];
    load_id_1 += gsz;
  }
  */
  float acc = 0;
  for(int i = gid; i< N; i+=gsz*2){
    acc += X[i];
    acc += (i+gsz < N) ? X[i+gsz] : 0;
  }
  local_data[loc_id] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  //////////////LOAD END///////////////////////////////////////

  /////////////////REDUCTION START//////////////////////////
  if(loc_id<128){
    local_data[loc_id] += local_data[loc_id+128]; 
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if(loc_id<64){
    local_data[loc_id] += local_data[loc_id+64]; 
  }
  barrier(CLK_LOCAL_MEM_FENCE);
   
  //!!!!! Unroll last warp !!!!! 
  if(loc_id < 32){
    volatile __local float *local_data_vol = local_data; // MUST use "volatile" keyword for remove compiler optimization.
    local_data_vol[loc_id] += local_data_vol[loc_id+32];
    local_data_vol[loc_id] += local_data_vol[loc_id+16];
    local_data_vol[loc_id] += local_data_vol[loc_id+8];
    local_data_vol[loc_id] += local_data_vol[loc_id+4];
    local_data_vol[loc_id] += local_data_vol[loc_id+2];
    local_data_vol[loc_id] += local_data_vol[loc_id+1];
  }
  
  ////////////REDUCTION END//////////////////////////

  //STORE local sum to global
  if(loc_id==0) Y[get_group_id(0)] = local_data[0];

}
""").build()
def reduce (queue, reduce_kernel,input_buffer,input_length,local_size,MAX_BLOCK_N):
    global_work_size = (local_size*min( int((N+local_size-1)/local_size) , MAX_BLOCK_N  ),)
    #global_work_size = ( math.ceil(input_length/2/local_size)*local_size,)
    #global_work_size = (32768,)
    local_work_size = (local_size,)
    block_n = int(global_work_size[0]/local_work_size[0])
    print("input vec")
    print(input_buffer)
    print(f'Reduction start. [ {input_length} -> {block_n}  ]')

    y_g = cl.Buffer(ctx,mf.READ_WRITE,block_n*4)
    print("output vec")
    print(y_g)
    mem_loc = cl.LocalMemory(local_size*4)

    evt = reduce_kernel(queue,global_work_size,local_work_size,np.int32(input_length),input_buffer,np.int32(0),y_g,mem_loc)
    evt.wait()
    
    t_start = evt.get_profiling_info(cl.profiling_info.START)
    t_end = evt.get_profiling_info(cl.profiling_info.END)
    elapsed_ns = t_end - t_start
    tot_bytes = input_length*4 + block_n*4

    
    print(f'GPU kernel time : {elapsed_ns *1e-6}ms')
    print(f'Kernel effective bandwidth : {tot_bytes/elapsed_ns} GB/s')
    
    ####for testing
    """
    y_np = np.empty((block_n,)).astype(np.float32) 
    cl.enqueue_copy(queue, y_np, y_g)
    print(y_np)
    print(f'present sum : {np.sum(y_np)}')
    """
    ####
    
    #print("\n\n")
    return [y_g,block_n,tot_bytes/elapsed_ns]
"""
global_work_size = (math.ceil(N/256)*256,)
local_work_size = (256,) 
y_np = np.empty((int(global_work_size[0]/local_work_size[0]),)).astype(np.float32)
y_g = cl.Buffer(ctx, mf.WRITE_ONLY, y_np.nbytes)
evt = prg.sasum(queue, global_work_size,local_work_size, np.int32(N),x_g,np.int32(0), y_g,cl.LocalMemory(256*4))
evt.wait()
"""

input_buff = x_g
"""
bandwidths = []
for MAX_BLOCK_N in range(64,2048) :
    print(f'Currently doing {MAX_BLOCK_N}')
    band = 0
    for i in range(100) :
        y_info = reduce(queue,prg.sasum,x_g,N,256,MAX_BLOCK_N)
        band += y_info[2]
    bandwidths.append(band/100)
"""
while N > 1 :
    y_info = reduce(queue,prg.sasum,input_buff,N,256,224)
    N           = y_info[1]
    input_buff  = y_info[0] 
    if N < 256 :
        break

y_g = y_info[0]
y_N = y_info[1]
y_np = np.empty((y_N,)).astype(np.float32) 
cl.enqueue_copy(queue, y_np, y_g)
print(y_np)
ysum = np.sum(y_np)
print(f'{x_np.shape} reduced to {y_np.shape}')
print(f'after sum : {ysum}')
print(f'Error rate : {(xsum-ysum)/xsum*100} %')
"""
print(f'Maximum bandwidth : {bandwidths}')

f = open('./expr.txt','w')
for i in range(64,2048) :
    f.write(f'{i} {bandwidths[i-64]}\n')
f.close()
plt.title('bandwidth')
plt.plot(range(64,2048),bandwidths)
plt.xlabel("MAX_BLOCK_N")
plt.ylabel("bandwidth (GB/s)")
plt.show()
"""
# Check on CPU with Numpy:
#print(res_np - (a_np + b_np))
#print(np.linalg.norm(res_np - (a_np + b_np)))


