#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#define TILE_WIDTH 16
#define CUDA_MAX_NUM_THREADS 512

namespace mxnet {
    namespace op {

        __constant__ float constK[15000];
        __global__ void forward_kernel_atomicAdd(float *__restrict__ y, const float *__restrict__ x,
                                                 const int B, const int M, const int C, const int H,
                                                 const int W, const int K, const int W_grid) {

            const int H_out = H - K + 1;
            const int W_out = W - K + 1;
            const int h = ((blockIdx.z / W_grid) * TILE_WIDTH + threadIdx.y);
            const int w = ((blockIdx.z % W_grid) * TILE_WIDTH + threadIdx.x);

            const bool flag1 = h < H_out && w < W_out;

            int iter = K / 2;
            if (iter * 2 < K)
                iter += 1;

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) constK[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

            int b = blockIdx.x;
            int m = blockIdx.y % M;
            int c = blockIdx.y / M;

            float acc1 = 0.0;

#pragma unroll

            for (int p = 0; p < iter; ++p) {
                for (int q = 0; q < iter; ++q) {
                    int p1 = 2 * p;
                    int p2 = 2 * p + 1;

                    int q1 = 2 * q;
                    int q2 = 2 * q + 1;

                    if (flag1) acc1 += x4d(b, c, h + p1, w + q1) * k4d(m, c, p1, q1);

                    if (q2 < K && p2 < K) {
                        if (flag1)
                            acc1 += x4d(b, c, h + p1, w + q2) * k4d(m, c, p1, q2) +
                                    x4d(b, c, h + p2, w + q2) * k4d(m, c, p2, q2) +
                                    x4d(b, c, h + p2, w + q1) * k4d(m, c, p2, q1);
                    } else if (q2 < K) {
                        if (flag1) acc1 += x4d(b, c, h + p1, w + q2) * k4d(m, c, p1, q2);
                    } else if (p2 < K) {
                        if (flag1) acc1 += x4d(b, c, h + p2, w + q1) * k4d(m, c, p2, q1);
                    }
                }
            }

            if (flag1) atomicAdd(&y[b * M * H_out * W_out + m * H_out * W_out + h * W_out + w], acc1);

#undef y4d
#undef x4d
#undef k4d
        }
      template<>
        void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y,
                                 const mshadow::Tensor<gpu, 4, float> &x,
                                 const mshadow::Tensor<gpu, 4, float> &k) {

            // Extract the tensor dimensions into B,M,C,H,W,K
            const int B = x.shape_[0];
            const int M = y.shape_[1];
            const int C = x.shape_[1];
            const int H = x.shape_[2];
            const int W = x.shape_[3];
            const int K = k.shape_[3];

            // the kernel with optimization: shared memory convolution &&
            // loop unrolling && make k a constant memory
            const int H_out = H - K + 1;
            const int W_out = W - K + 1;

            int W_grid = ceil(float(W_out) / TILE_WIDTH);
            int H_grid = ceil(float(H_out) / TILE_WIDTH);
            // int Y = M * C;
            int Wstd = ceil(W_grid / 2.0);
            int Z = Wstd * ceil(H_grid / 2.0);
 // the kernel with restricted and const memory and atomicAdd
            dim3 block2(TILE_WIDTH, TILE_WIDTH, 1);
            dim3 grid2(B, M * C, W_grid * H_grid);
            cudaMemcpyToSymbol(constK, k.dptr_, M * C * K * K * sizeof(float));
            forward_kernel_atomicAdd << < grid2, block2 >> >
            (y.dptr_, x.dptr_, B, M, C, H, W, K, W_grid);

            // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
            // MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
        }

/*
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
        template<typename gpu, typename DType>
        void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x,
                     const mshadow::Tensor<gpu, 4, DType> &w) {
            CHECK_EQ(0, 1) << "Remove this line and replace it with your implementation.";
        }
    }
}

#endif
