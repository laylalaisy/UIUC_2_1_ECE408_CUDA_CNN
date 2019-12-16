#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define CUDA_MAX_NUM_THREADS 1024

#define TILE_WIDTH_1 16
#define TILE_WIDTH_2 24

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

// Kernel code.
__global__ void ConvLayerForward1(int C, int K, int M, int H, int W, int W_out, int H_out, float* x, float* k, float* y) {

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    __shared__ float tileMatA[TILE_WIDTH_1][TILE_WIDTH_1];
    __shared__ float tileMatB[TILE_WIDTH_1][TILE_WIDTH_1];

    int b = blockIdx.z;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH_1 + ty;
    int column = blockIdx.x * TILE_WIDTH_1 + tx;
    int kk = K*K;
    int numMatAColumns = C*kk; // This is the same as numMatBRows.
    float acc = 0.0;
    int num_iterations = ceil(numMatAColumns/(1.0*TILE_WIDTH_1));
    
    for (int i = 0; i < num_iterations; ++i) {
        int temp_col = i*TILE_WIDTH_1 + tx, temp_row = i*TILE_WIDTH_1 + ty;
        tileMatA[ty][tx] = 0;
        tileMatB[ty][tx] = 0;
    
        // Original indices in the filter tensor.
        int W_c = temp_col/kk;
        // int tmp = temp_col%(K*K);
        int tmp = temp_col - W_c * kk;
        int W_h = tmp/K;
        // int W_w = tmp%K;
        int W_w = tmp - W_h * K;
        if (temp_col < numMatAColumns && row < M) {
            tileMatA[ty][tx] = k4d(row, W_c, W_h, W_w);
        }
        else {
            tileMatA[ty][tx] = 0;
        }    

        // Original indices in the input tensor.
        int X_c = temp_row/kk;
        // tmp = temp_row%(K*K);
        tmp = temp_row - X_c * kk;
        int X_p = tmp/K;
        // int X_q = tmp%K;
        int X_q = tmp - X_p * K;
        int X_h = column/W_out;
        // int X_w = column%W_out;
        int X_w = column - X_h * W_out;

        if (temp_row < numMatAColumns && column < H_out*W_out) {
            tileMatB[ty][tx] = x4d(b, X_c, X_h + X_p, X_w + X_q);
        }
        else {
            tileMatB[ty][tx] = 0;
        }
        __syncthreads();

        #pragma unroll
        for (int q = 0; q < TILE_WIDTH_1; ++q) {
            acc += tileMatA[ty][q] * tileMatB[q][tx];
        }
        __syncthreads();
    }

    // Original indices in the output tensor.
    int Y_h = column / W_out;
    // int Y_w = column % W_out;
    int Y_w = column - Y_h * W_out;

    if (row < M && column < W_out*H_out) {
        y4d(b, row, Y_h, Y_w) = acc;
    }
}

__global__ void ConvLayerForward2(int C, int K, int M, int H, int W, int W_out, int H_out, float* x, float* k, float* y) {

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    __shared__ float tileMatA[TILE_WIDTH_2][TILE_WIDTH_2];
    __shared__ float tileMatB[TILE_WIDTH_2][TILE_WIDTH_2];

    int b = blockIdx.z;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH_2 + ty;
    int column = blockIdx.x * TILE_WIDTH_2 + tx;
    int kk = K*K;
    int numMatAColumns = C*kk; // This is the same as numMatBRows.
    float acc = 0.0;
    int num_iterations = ceil(numMatAColumns/(1.0*TILE_WIDTH_2));
    
    for (int i = 0; i < num_iterations; ++i) {
        int temp_col = i*TILE_WIDTH_2 + tx, temp_row = i*TILE_WIDTH_2 + ty;
        tileMatA[ty][tx] = 0;
        tileMatB[ty][tx] = 0;
    
        // Original indices in the filter tensor.
        int W_c = temp_col/kk;
        // int tmp = temp_col%kk;
        int tmp = temp_col - W_c * kk;
        int W_h = tmp/K;
        // int W_w = tmp%K;
        int W_w = tmp - W_h * K;
        if (temp_col < numMatAColumns && row < M) {
            tileMatA[ty][tx] = k4d(row, W_c, W_h, W_w);
        }
        else {
            tileMatA[ty][tx] = 0;
        }    

        // Original indices in the input tensor.
        int X_c = temp_row/kk;
        // tmp = temp_row%kk;
        tmp = temp_row - X_c * kk;
        int X_p = tmp/K;
        // int X_q = tmp%K;
        int X_q = tmp - X_p * K;
        int X_h = column/W_out;
        // int X_w = column%W_out;
        int X_w = column - X_h * W_out;

        if (temp_row < numMatAColumns && column < H_out*W_out) {
            tileMatB[ty][tx] = x4d(b, X_c, X_h + X_p, X_w + X_q);
        }
        else {
            tileMatB[ty][tx] = 0;
        }
        __syncthreads();

        #pragma unroll
        for (int q = 0; q < TILE_WIDTH_2; ++q) {
            acc += tileMatA[ty][q] * tileMatB[q][tx];
        }
        __syncthreads();
    }

    // Original indices in the output tensor.
    int Y_h = column / W_out;
    // int Y_w = column % W_out;
    int Y_w = column - Y_h * W_out;

    if (row < M && column < W_out*H_out) {
        y4d(b, row, Y_h, Y_w) = acc;
    }
}

/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template<>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w) {
    

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    //CHECK_EQ(0, 1) << "Missing an ECE408 GPU implementation!";

    // You'll probably need to launch kernels against the right stream to keep MXNet happy
    cudaStream_t s = y.stream_->stream_;

    // Extract the tensor dimensions into B,M,C,H,W,K
    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    if (M == 24) {
    	dim3 gridDim(ceil(H_out*W_out/(1.0*TILE_WIDTH_2)),ceil(M/(1.0*TILE_WIDTH_2)),B);
    	dim3 blockDim(TILE_WIDTH_2,TILE_WIDTH_2,1);
    
    	ConvLayerForward2<<<gridDim, blockDim, 0, s>>>(C, K, M, H, W, W_out, H_out, x.dptr_, w.dptr_, y.dptr_);

    	MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
    } else {
    	dim3 gridDim(ceil(H_out*W_out/(1.0*TILE_WIDTH_1)),ceil(M/(1.0*TILE_WIDTH_1)),B);
    	dim3 blockDim(TILE_WIDTH_1,TILE_WIDTH_1,1);
    
    	ConvLayerForward1<<<gridDim, blockDim, 0, s>>>(C, K, M, H, W, W_out, H_out, x.dptr_, w.dptr_, y.dptr_);

    	MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
    }
}


/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template<typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w) {
    assert( 0 && "No forward implementation for other datatypes needed for ECE408");
}

}
}

#endif
