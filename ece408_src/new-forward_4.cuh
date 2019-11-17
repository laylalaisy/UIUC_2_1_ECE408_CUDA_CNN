#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

#define TILE_WIDTH 16
#define BLOCK_SIZE 32

namespace mxnet
{
namespace op
{

__constant__ float constMem[8192];

/* ===== Optimization 1: Shared Memory Convolution==== */
__global__ void forward_kernel_shared(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    int W_grid = ceil((float)W_out/TILE_WIDTH);

    int TILE_WIDTH_K = TILE_WIDTH + K -1;
    extern __shared__ float shared[];
    float* sharedX = &shared[0];
    float* sharedW = &shared[TILE_WIDTH_K*TILE_WIDTH_K];

    int b = bx;
    int m = by;
    int h_base = (bz / W_grid) * TILE_WIDTH;
    int w_base = (bz % W_grid) * TILE_WIDTH;
    int h = h_base + ty;
    int w = w_base + tx;

    float acc = 0.0;

    for (int c=0; c<C; c++) {
      // 1. load the filter W into the shared memory
      if ((tx < K) && (ty < K)) {
        sharedW[ty*K+tx] = k4d(m, c, ty, tx);
      }
      __syncthreads();

      // 2. load tile from X into the shared memory
      for (int i=h; i<h_base+TILE_WIDTH_K; i+=TILE_WIDTH) {
        for (int j=w; j<w_base+TILE_WIDTH_K; j+=TILE_WIDTH) {
          if (i<H && j<W) {
            sharedX[(i-h_base)*TILE_WIDTH_K+(j-w_base)] = x4d(b,c, i, j);
          } else {
            sharedX[(i-h_base)*TILE_WIDTH_K+(j-w_base)] = 0;
          }
          
        }
      }
      __syncthreads();

      // 3. compute partial sum of output Y
      for (int p=0; p<K; p++) {
        for (int q=0; q<K; q++) {
          if (((ty+p)<TILE_WIDTH_K) && ((tx+q)<TILE_WIDTH_K)) {
            acc += sharedX[(ty+p)*TILE_WIDTH_K+(tx+q)]*sharedW[p*K+q];
          }
        }
      }
      __syncthreads();
    }

    if (b<B && m<M && h<H_out && w<W_out) {
      y4d(b, m, h, w) = acc;
    }

#undef y4d
#undef x4d
#undef k4d
}



/* ===== Optimization 2: Shared Memory with Weight matrix (kernel values) in constant memory ==== */  
__global__ void forward_kernel_shared_constMem(float *y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    int W_grid = ceil((float)W_out/TILE_WIDTH);
    int H_grid = ceil((float)H_out/TILE_WIDTH);
    int TILE_WIDTH_K = TILE_WIDTH + K -1;

    extern __shared__ float shared[];
    float* sharedX = &shared[0];

    int b = bx;
    int m = by;
    int h_base = (bz / W_grid) * TILE_WIDTH;
    int w_base = (bz % W_grid) * TILE_WIDTH;
    int h = h_base + ty;
    int w = w_base + tx;

    float acc = 0.0;
    float weight;

    for (int c=0; c<C; c++) {
      // 1. load tile from X into the shared memory
      for (int i=h; i<h_base+TILE_WIDTH_K; i+=TILE_WIDTH) {
        for (int j=w; j<w_base+TILE_WIDTH_K; j+=TILE_WIDTH) {
          if (b<B && i<H && j<W) {
            sharedX[(i-h_base)*TILE_WIDTH_K+(j-w_base)] = x4d(b,c, i, j);
          } else {
            sharedX[(i-h_base)*TILE_WIDTH_K+(j-w_base)] = 0.0;
          }
          
        }
      }
      __syncthreads();

      // 2. compute partial sum of output Y
      for (int p=0; p<K; p++) {
        for (int q=0; q<K; q++) {
      	  if (((ty+p)<TILE_WIDTH_K) && ((tx+q)<TILE_WIDTH_K)) {
            weight = constMem[m*C*K*K+c*K*K+p*K+q];
            acc += sharedX[(ty+p)*TILE_WIDTH_K+(tx+q)]*weight;
          }
        }
      }
      __syncthreads();
    }

    if (b<B && m<M && h<H_out && w<W_out) {
      y4d(b, m, h, w) = acc;
    }

#undef y4d
#undef x4d
#undef k4d
}


/* ===== Optimization 3: Unroll + shared-memory Matrix multiply ==== */
__global__ void unroll(const float *x, float *X_unroll, int b, const int C, const int H, const int W, const int K, int H_out, int W_out, int H_unroll, int W_unroll) {
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (idx < C * W_unroll) {
        int c = idx / W_unroll;
        int h_out = (idx % W_unroll) / W_out;
        int w_out = (idx % W_unroll) % W_out;
        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                int h_unroll = c * K * K + p * K + q;
                int w_unroll = h_out * W_out + w_out;
                X_unroll[h_unroll * W_unroll + w_unroll] = x4d(b, c, h_out + p, w_out + q);
            }
        }
    }
#undef x4d
}

__global__ void forward_kernel_unroll(float *X_unroll, float *y, const float *k, int b, const int B, const int M, const int C, const int H, const int W, const int K, int H_out, int W_out, int H_unroll, int W_unroll) {
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    __shared__ float sharedW[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sharedX[TILE_WIDTH][TILE_WIDTH];

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float result = 0;

    for (int i = 0; i < (H_unroll + TILE_WIDTH - 1) / TILE_WIDTH; i++) {
        // 1. load W into shared memory
        if (i * TILE_WIDTH + tx < H_unroll && row < M) {
            sharedW[ty][tx] = k[row * H_unroll + i * TILE_WIDTH + tx];
        } else {
            sharedW[ty][tx] = 0.0;
        }

        // 2. load X_unroll into shared memory
        if (i * TILE_WIDTH + ty < H_unroll && col < W_unroll) {
            sharedX[ty][tx] = X_unroll[(i * TILE_WIDTH + ty) * W_unroll + col];
        } else {
            sharedX[ty][tx] = 0.0;
        }
        
        __syncthreads();

        // 3. matrix multiple
        for (int j = 0; j < TILE_WIDTH; j++) {
            result += (sharedW[ty][j] * sharedX[j][tx]);
        }
        __syncthreads();

        if (row < M && col < W_unroll) {
            y[b * M * W_unroll + row * W_unroll + col] = result;
        }
    }

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
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    // CHECK_EQ(0, 1) << "Missing GPU implementation!";

    cudaStream_t s = y.stream_->stream_;

    // Extract the tensor dimensions into B,M,C,H,W,K
    const int B = x.shape_[0];  // num of batch
    const int M = y.shape_[1];  // num of output feature map
    const int C = x.shape_[1];  // num of channel
    const int H = x.shape_[2];  // height
    const int W = x.shape_[3];  // width
    const int K = w.shape_[3];  // size of filter

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int TILE_WIDTH_K = TILE_WIDTH + K - 1;
    int H_grid = ceil((float)H_out/TILE_WIDTH);
    int W_grid = ceil((float)W_out/TILE_WIDTH);
    int Z = H_grid * W_grid;

    // ===== Optimization 1: Shared Memory Convolution==== 
    // Set the kernel dimensions
    dim3 gridDim(B, M, Z);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);

    // Call the kernel
    size_t shared = sizeof(float)*(TILE_WIDTH_K*TILE_WIDTH_K+K*K);
    forward_kernel_shared<<<gridDim, blockDim, shared, s>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

    // ===== Optimization 2: Shared Memory with Weight matrix (kernel values) in constant memory ====     
    // Load W into constant memory
    cudaMemcpyToSymbol(constMem, w.dptr_, M*C*K*K*sizeof(float));

    // Set the kernel dimensions
    dim3 gridDim(B, M, Z);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);

    // Call the kernel: shared memory with constant memory
    size_t shared = sizeof(float)*(TILE_WIDTH_K*TILE_WIDTH_K);
    forward_kernel_shared_constMem<<<gridDim, blockDim, shared, s>>>(y.dptr_, x.dptr_, B, M, C, H, W, K);

    // ===== Optimization 3: Unroll + shared-memory Matrix multiply ==== 
    // Set the unroll kernel
    float *X_unroll;
    int W_unroll = H_out * W_out;
    int H_unroll = C * K * K;
    cudaMalloc((void**) &X_unroll, H_unroll * W_unroll * sizeof(float));

    // Set the kernel dimensions
    dim3 gridDim1(ceil((float)C * H_out * W_out / BLOCK_SIZE), 1, 1);
    dim3 blockDim1(BLOCK_SIZE, 1, 1);
    dim3 gridDim2(ceil((float)H_unroll / TILE_WIDTH), ceil((float)M / TILE_WIDTH), 1);
    dim3 blockDim2(TILE_WIDTH, TILE_WIDTH, 1);

    // Call the kernel
    for (int b = 0; b < B; b++) {
        unroll<<<gridDim1, blockDim1>>>(x.dptr_, X_unroll, b, C, H, W, K, H_out, W_out, H_unroll, W_unroll);
        forward_kernel_unroll<<<gridDim2, blockDim2>>>(X_unroll, y.dptr_, w.dptr_, b, B, M, C, H, W, K, H_out, W_out, H_unroll, W_unroll);
    }


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


