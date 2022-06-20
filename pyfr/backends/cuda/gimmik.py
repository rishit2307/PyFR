# -*- coding: utf-8 -*-

from gimmik import generate_mm
import numpy as np

from pyfr.backends.base import NotSuitableError
from pyfr.backends.cuda.provider import (CUDAKernel, CUDAKernelProvider,
                                         get_grid_for_block)


class CUDAGiMMiKKernels(CUDAKernelProvider):
    def __init__(self, backend):
        super().__init__(backend)

    def mul(self, a, b, out, alpha=1.0, beta=0.0):
        # Ensure the matrices are compatible
        if a.nrow != out.nrow or a.ncol != b.nrow or b.ncol != out.ncol:
            raise ValueError('Incompatible matrices for out = a*b')

        # Check that A is constant
        if 'const' not in a.tags:
            raise NotSuitableError('GiMMiK requires a constant a matrix')

        # Fetch the matrix and tally up the number of non-zeros
        arr = a.get()
        nnz, nuq = np.count_nonzero(arr), len(np.unique(np.abs(arr)))

        # Check that A is suitable
        if nuq > 28 and nnz / arr.size > 0.15:
            raise NotSuitableError('Matrix is inappropriate for GiMMiK')

        # Generate
        src = generate_mm(arr, a.dtype, 'cuda', alpha=alpha, beta=beta,
                          n=b.ncol, ldb=b.leaddim, ldc=out.leaddim)

        # Build
        fun = self._build_kernel('gimmik_mm', src, 'PP')
        fun.set_cache_pref(prefer_l1=True)

        # Determine the grid/block
        block = (128, 1, 1)
        grid = get_grid_for_block(block, b.ncol)

        # Set the parameters
        params = fun.make_params(grid, block)
        params.set_args(b, out)

        class MulKernel(CUDAKernel):
            def add_to_graph(self, graph, deps):
                return graph.graph.add_kernel(params, deps)

            def run(self, stream):
                fun.exec_async(stream, params)

        return MulKernel(mats=[b, out])
