#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define BLOCK_SIZE 1024
#define TILE_WIDTH 32

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

__global__ void matrixMultiplyShared(float *A, float *B, float *C, int numAColumns, int numCRows, int numCColumns) 
{
  // numARows = numCRows
  // numBRows = numAColumns
  // numBColumns = numCColumns

  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];
  int rowIx = blockIdx.y * blockDim.y + threadIdx.y;
  int colIx = blockIdx.x * blockDim.x + threadIdx.x;

  float result = 0;
  
  for (int tileIx = 0; tileIx < ceil(1.0*numAColumns/TILE_WIDTH); tileIx++) {

    int col = tileIx*TILE_WIDTH+threadIdx.x;
    if (col < numAColumns)
      tileA[threadIdx.y][threadIdx.x] = A[rowIx*numAColumns+col];
    else
      tileA[threadIdx.y][threadIdx.x] = 0;

    int row = tileIx*TILE_WIDTH+threadIdx.y;
    if (row < numAColumns)
      tileB[threadIdx.y][threadIdx.x] = B[row*numCColumns + colIx];
    else 
      tileB[threadIdx.y][threadIdx.x] = 0;


    __syncthreads();
    for (int k = 0; k < TILE_WIDTH; k++)
       result += tileA[threadIdx.y][k]*tileB[k][threadIdx.x];
      
    __syncthreads();   
  }
  
  if ((rowIx < numCRows) && (colIx < numCColumns)) {
    C[rowIx*numCColumns+colIx] = result;
  }
  
}


void gemm(float* Kernel, float* X_unrolled,  float* Y, int CKK, int M, int HW) {
    // matrixMultiplyShared(float *A, float *B, float *C,
    //                                  int numAColumns, int numCRows, int numCColumns)
    // W_unroll = K
    int blockDimX = TILE_WIDTH, blockDimY = TILE_WIDTH;
    int gridDimY = ceil(1.0*M/blockDimY), gridDimX = ceil(1.0*HW/blockDimX);
    dim3 gridDim (gridDimX, gridDimY), blockDim (blockDimX, blockDimY);
    matrixMultiplyShared<<<gridDim, blockDim>>>(Kernel, X_unrolled, Y, CKK, M, HW);
}


__global__ void unrollKernel(float* X_unrolled, int size, float* X, int C, int K, int H, int W) {
    int ix = blockDim.x*blockIdx.x + threadIdx.x;
    if (ix >= size)
        return;
    int H_out = H-K+1;
    int W_out = W-K+1;
    int row = ix/(H_out*W_out);
    int col = ix%(H_out*W_out);
    int q = row % K;
    row /= K;
    int p = row % K;
    int c = row / K;
    int w = col % W_out;
    int h = col / W_out;
    X_unrolled[ix] = X[(c) * (H * W) + (h+p) * (W) + w+q];
}

void unroll(float* X_unrolled, int size, float* X, int C, int K, int H, int W) {
    int gridDim = ceil(1.0*size/BLOCK_SIZE);
    unrollKernel<<<gridDim, BLOCK_SIZE>>>(X_unrolled, size, X, C, K, H, W);
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
    float* Y = y.dptr_;
    float* X = x.dptr_;
    float* Kernel = k.dptr_;
    
    float* X_unrolled;
    int size = C*K*K*H_out*W_out;
    cudaMalloc(&X_unrolled, sizeof(float)*size);
    for (int b = B; b--; ) {
        unroll(X_unrolled, size, X+b*C*H*W, C, K, H, W);
        gemm(Kernel,  X_unrolled,  Y+b*M*H_out*W_out,  C*K*K,  M,  H_out*W_out);
    }
    cudaFree(X_unrolled);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

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
