#define VS 4
#define PADDING_VAL 0
//This kernel is written for 256 work-group-size
__kernel void reduce_sum(const int N,
                   const __global float* X, const int indexX, 
									 __global float* Y,__local float* local_data) {
  X += indexX;
  const unsigned int gid = get_global_id(0);
  const unsigned int gsz = get_global_size(0);
  const unsigned int loc_id = get_local_id(0);
  const unsigned int lsz = get_local_size(0);
  
  ////////////////LOAD START///////////////////////////////////////
  //load to local memory, first add during load.
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
