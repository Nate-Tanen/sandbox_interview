import os

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


CPP_SOURCE = r"""
#include <torch/extension.h>

torch::Tensor naive_gemm_cpu(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(!A.is_cuda(), "A must be a CPU tensor");
    TORCH_CHECK(!B.is_cuda(), "B must be a CPU tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be rank-2 tensors");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions must match");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.scalar_type() == torch::kFloat32, "B must be float32");

    auto A_contig = A.contiguous();
    auto B_contig = B.contiguous();
    const auto N = static_cast<int64_t>(A_contig.size(0));

    auto C = torch::zeros({N, N}, A_contig.options());

    const float* a_ptr = A_contig.data_ptr<float>();
    const float* b_ptr = B_contig.data_ptr<float>();
    float* c_ptr = C.data_ptr<float>();

    // Outer loop over rows of C, middle loop over columns of C,
    // inner loop over the product dimension.
    for (int64_t row = 0; row < N; ++row) {
        for (int64_t col = 0; col < N; ++col) {
            float acc = 0.0f;
            int64_t k;
            // Loop unrolling for enhanced performance
            for (k = 0; k <= N - 4; k += 4) {
                acc += a_ptr[row * N + k] * b_ptr[k * N + col];
                acc += a_ptr[row * N + k + 1] * b_ptr[(k + 1) * N + col];
                acc += a_ptr[row * N + k + 2] * b_ptr[(k + 2) * N + col];
                acc += a_ptr[row * N + k + 3] * b_ptr[(k + 3) * N + col];
            }
            // Process remaining elements if N is not a multiple of 4
            for (; k < N; ++k) {
                acc += a_ptr[row * N + k] * b_ptr[k * N + col];
            }
            c_ptr[row * N + col] = acc;
        }
    }

    return C;
}
"""


BEST_GEMM = load_inline(
    name="best_gemm_extension",
    cpp_sources=[CPP_SOURCE],
    functions=["naive_gemm_cpu"],
    extra_cflags=["-O0"],
    with_cuda=False,
    verbose=False,
    build_directory=os.environ.get("TORCH_EXTENSIONS_DIR"),
)


class ModelNew(nn.Module):
    """
    Same naive C++ as `task/base_kernel.py`. Promotion to this file requires beating the
    profiled naive baseline (~202 ms mean for this implementation), not torch.matmul.
    """

    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return BEST_GEMM.naive_gemm_cpu(A, B)
