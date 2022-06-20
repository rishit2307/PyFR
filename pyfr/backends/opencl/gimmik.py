# -*- coding: utf-8 -*-

from gimmik import generate_mm
import numpy as np

from pyfr.backends.base import NotSuitableError
from pyfr.backends.opencl.provider import OpenCLKernel, OpenCLKernelProvider


class OpenCLGiMMiKKernels(OpenCLKernelProvider):
    def __init__(self, backend):
        super().__init__(backend)

        self.max_nnz = backend.cfg.getint('backend-opencl', 'gimmik-max-nnz',
                                          2048)

    def mul(self, a, b, out, alpha=1.0, beta=0.0):
        # Ensure the matrices are compatible
        if a.nrow != out.nrow or a.ncol != b.nrow or b.ncol != out.ncol:
            raise ValueError('Incompatible matrices for out = a*b')

        # Check that A is constant
        if 'const' not in a.tags:
            raise NotSuitableError('GiMMiK requires a constant a matrix')

        # Check that A is reasonably sparse
        if np.count_nonzero(a.get()) > self.max_nnz:
            raise NotSuitableError('Matrix too dense for GiMMiK')

        # Generate
        src = generate_mm(a.get(), a.dtype, 'opencl', alpha=alpha, beta=beta,
                          n=b.ncol, ldb=b.leaddim, ldc=out.leaddim)

        # Build
        fun = self._build_kernel('gimmik_mm', src, 'PP')
        fun.set_dims((b.ncol,))
        fun.set_args(b, out)

        class MulKernel(OpenCLKernel):
            def run(self, queue, wait_for=None, ret_evt=False):
                return fun.exec_async(queue, wait_for, ret_evt)

        return MulKernel(mats=[b, out])
