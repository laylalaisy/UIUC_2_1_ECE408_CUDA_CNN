#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define TILE_WIDTH_1 16
#define TILE_WIDTH_2 24 

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

__global__ void matrixMultiplyShared(const float * __restrict__ Kernel, const float * __restrict__ X, float * __restrict__ Y, int M, int C, int H, int W, int K, int H_out, int W_out)
{
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  int numAColumns = C*K*K;
  int numCColumns = H_out*W_out;
  X += blockIdx.z*C*H*W;
  Y += blockIdx.z*M*numCColumns;
  
  __shared__ float tileA[TILE_WIDTH_2][TILE_WIDTH_2];
  __shared__ float tileB[TILE_WIDTH_2][TILE_WIDTH_2];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int rowIx = blockIdx.y * TILE_WIDTH_2 + ty;
  int colIx = blockIdx.x * TILE_WIDTH_2 + tx;

  float result = 0;
  int q, p, c, w, h;

  #pragma unroll
  for (int tileIx = 0; tileIx < 25; ++tileIx){

    int temp       = tileIx*TILE_WIDTH_2;
    int matrix_col = temp + tx;
    if (matrix_col < numAColumns)
      tileA[ty][tx] = Kernel[rowIx*numAColumns+matrix_col];  
    else
      tileA[ty][tx] = 0;

    int matrix_row = temp + ty;
    if (colIx < numCColumns && matrix_row < numAColumns) {
      q = matrix_row % K;
      matrix_row /= K;
      p = matrix_row % K;
      c = matrix_row / K;
      w = colIx % W_out;
      h = colIx / W_out;
      tileB[ty][tx] = X[c * (H * W) + (h+p) * (W) + w+q];
    }
    else 
      tileB[ty][tx] = 0;

    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TILE_WIDTH_2; k+=1){
       result += tileA[ty][k]*tileB[k][tx];
    }
    __syncthreads();   
  }
  
  if (colIx < numCColumns) {
    Y[rowIx*numCColumns+colIx] = result;
  }
}

__global__ void forward_kernel(float * __restrict__ y, const float * __restrict__ x, const float * __restrict__ k, const int B, const int M, const int C, const int H, const int W, const int K, const int H_out, const int W_out, const int W_grid)
{

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    
    int b, m, h, w, c=0, p, q;

    b = blockIdx.x;
    m = blockIdx.y;
    h = blockIdx.z / W_grid * TILE_WIDTH_1 + threadIdx.y;
    w = blockIdx.z % W_grid * TILE_WIDTH_1 + threadIdx.x;

    if (h >= H_out || w >= W_out)
      return;

    float acc = 0;

    #pragma unroll
    for (p = 0; p < 7; p++)
      #pragma unroll 
      for (q = 0; q < 7; q++)
        acc += x4d(b, c, h + p, w + q) * k4d(m, c, p, q);

    y4d(b, m, h, w) = acc;

#undef y4d
#undef x4d
#undef k4d

}

/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &k)
{

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    // CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";

    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...
    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = k.shape_[3];
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    if ( M / 24) {
      int gridDimY = ceil(1.0*M/TILE_WIDTH_2), gridDimX = ceil(1.0*H_out*W_out/TILE_WIDTH_2);
      dim3 gridDim(gridDimX, gridDimY, B), blockDim(TILE_WIDTH_2, TILE_WIDTH_2, 1);
      matrixMultiplyShared<<< gridDim, blockDim >>> (k.dptr_, x.dptr_, y.dptr_, M, C, H, W, K, H_out, W_out);
      // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
      MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
    }
    else {
      const int W_grid = ceil(1.*W_out/TILE_WIDTH_1);
      const int H_grid = ceil(1.*H_out/TILE_WIDTH_1);
      int Z = H_grid * W_grid;
      dim3 blockDim(TILE_WIDTH_1, TILE_WIDTH_1, 1), gridDim(B, M, Z);
      forward_kernel<<< gridDim, blockDim >>> (y.dptr_, x.dptr_, k.dptr_, B, M, C, H, W, K, H_out, W_out, W_grid);
      // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
      MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
    }
}

/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif
