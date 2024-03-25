from ctypes import POINTER, c_int, c_double, c_float, c_void_p, c_uint64

import numpy as np

from pyfr.backends.cuda.provider import CUDAKernel, CUDAKernelProvider
from pyfr.ctypesutil import LibWrapper


# Possible CUBLAS exception types
class CUBLASError(Exception): pass
class CUBLASNotInitialized(CUBLASError): pass
class CUBLASAllocFailed(CUBLASError): pass
class CUBLASInvalidValue(CUBLASError): pass
class CUBLASArchMismatch(CUBLASError): pass
class CUBLASMappingError(CUBLASError): pass
class CUBLASExecutionFailed(CUBLASError): pass
class CUBLASInternalError(CUBLASError): pass


class CUBLASWrappers(LibWrapper):
    _libname = 'cublas'

    # Error codes
    _statuses = {
        0x1: CUBLASNotInitialized,
        0x3: CUBLASAllocFailed,
        0x7: CUBLASInvalidValue,
        0x8: CUBLASArchMismatch,
        0xb: CUBLASMappingError,
        0xd: CUBLASExecutionFailed,
        0xe: CUBLASInternalError,
        '*': CUBLASError
    }

    # Constants
    OP_N = 0
    OP_T = 1
    OP_C = 2
    CUBLAS_POINTER_MODE_DEVICE = 1

    # Functions
    _functions = [
        (c_int, 'cublasCreate_v2', POINTER(c_void_p)),
        (c_int, 'cublasDestroy_v2', c_void_p),
        (c_int, 'cublasSetStream_v2', c_void_p, c_void_p),
        (c_int, 'cublasDgemm_v2', c_void_p, c_int, c_int, c_int, c_int, c_int,
         POINTER(c_double), c_void_p, c_int, c_void_p, c_int,
         POINTER(c_double), c_void_p, c_int),
        (c_int, 'cublasSgemm_v2', c_void_p, c_int, c_int, c_int, c_int, c_int,
         POINTER(c_float), c_void_p, c_int, c_void_p, c_int,
         POINTER(c_float), c_void_p, c_int),
        (c_int, 'cublasDdot_v2', c_void_p, c_int, c_void_p, c_int, c_void_p, 
        c_int, c_uint64),
        (c_int, 'cublasSetPointerMode_v2', c_void_p, c_int), 
        (c_int, 'cublasDaxpy_v2', c_void_p, c_int, POINTER(c_double), c_void_p, 
         c_int, c_void_p, c_int)
    ]

    def _transname(self, name):
        return name[:-3]


class CUDACUBLASKernels(CUDAKernelProvider):
    def __init__(self, backend):
        super().__init__(backend)

        self.handle = c_void_p()

        # Load and wrap CUBLAS
        self.lib = CUBLASWrappers()

        # Init
        self.lib.cublasCreate(self.handle)

        # Timing data cache
        self._mul_timing = {}

    def __del__(self):
        if self.handle:
            self.lib.cublasDestroy(self.handle)
    
    def add(self, a, b, out):
        w, h = self.lib, self.handle
        cublasadd = w.cublasDaxpy_v2

    def dot(self, a, b, out=None):
        w, h = self.lib, self.handle
        cublasdot = w.cublasDdot
        n = a.nrow*a.ncol
        x, y, z = a, b, out
        w.cublasSetPointerMode(h, w.CUBLAS_POINTER_MODE_DEVICE)

        # ckey = (n)

        def ddd(stream):
            w.cublasSetStream(h, stream)
            cublasdot(h, n, x, 1, y, 1, z)

        class DotKernel(CUDAKernel):
            def run(self, stream):
                return ddd(stream)
        
        return DotKernel(mats=[a, b, out])

    def mul(self, a, b, out, alpha=1.0, beta=0.0, gmres=False):
        cuda = self.backend.cuda
        w, h = self.lib, self.handle
        # print(f'a.nrow is {a.nrow}')
        # print(f'out.nrow is {out.nrow}')
        # print(f'a.ncol is {a.ncol}')
        # print(f'b.nrow is {b.nrow}')
        # print(f'b.ncol is {b.ncol}')
        # print(f'out.ncol is {out.ncol}')
        # print(f'A leaddim is {a.leaddim}')
        # print(f'B leaddim is {b.leaddim}')
        # print(f'C leaddim is {out.leaddim}')
        # print(f'gmres is {gmres}')

        # Ensure the matrices are compatible
        # if a.nrow != out.nrow or a.ncol != b.nrow or b.ncol != out.ncol:
        #     raise ValueError('Incompatible matrices for out = a*b')

        # cuBLAS expects inputs to be column-major (or Fortran order in
        # numpy parlance).  However as C = A*B => C^T = (A*B)^T
        # = (B^T)*(A^T) with a little trickery we can multiply our
        # row-major matrices directly.
        m =  b.ncol if not gmres else 1
        n, k =  a.nrow, a.ncol


        A, B, C = b, a, out
        # Cache key
        if gmres:
            ckey = (A.dtype, alpha, beta, m, n, k, 1, B.leaddim, C.leaddim)
        else:
            ckey = (A.dtype, alpha, beta, m, n, k, A.leaddim, B.leaddim, C.leaddim)

        # Size checks
        if any(sz > 2**31 - 1 for sz in ckey[3:]):
            raise ValueError('Matrices too large for cuBLAS')

        # α and β factors for C = α*(A*B) + β*C
        if a.dtype == np.float64:
            cublasgemm = w.cublasDgemm
            alpha_ct, beta_ct = c_double(alpha), c_double(beta)
        else:
            cublasgemm = w.cublasSgemm
            alpha_ct, beta_ct = c_float(alpha), c_float(beta)

        # Convenience wrapper
        def gemm(stream):
            w.cublasSetStream(h, stream)
            if gmres:
                # print(f'h is {h}')
                # print(f'm is {m}')
                # print(f'n is {n}')
                # print(f'k is {k}')
                # print(f'alpha_ct is {alpha_ct}')
                # print(f'A is {A}')
                # print(f'A.leaddim is {1}')
                # print(f'B is {B}')
                # print(f'B.leaddim is {B.leaddim}')
                # print(f'beta is {beta_ct}')
                # print(f'C is {C}')
                # print(f'C.leaddim is {C.leaddim}')

                cublasgemm(h, w.OP_N, w.OP_N, m, n, k, alpha_ct, A, 1,
                       B, B.leaddim, beta_ct, C, C.leaddim)
            else:
                cublasgemm(h, w.OP_N, w.OP_N, m, n, k, alpha_ct, A, A.leaddim,
                       B, B.leaddim, beta_ct, C, C.leaddim)


        # Obtain the performance of the kernel
        try:
            dt = self._mul_timing[ckey]
        except KeyError:
            # Save a copy of the contents of the output matrix
            out_np = getattr(out, 'parent', out).get()

            # Benchmark the kernel and update the cache
            self._mul_timing[ckey] = dt = self._benchmark(gemm)

            # Restore the output matrix
            getattr(out, 'parent', out).set(out_np)

        class MulKernel(CUDAKernel):
            def add_to_graph(self, graph, deps):
                stream = cuda.create_stream()

                # Capture the execution of cuBLAS to obtain a graph
                stream.begin_capture()
                gemm(stream)
                gnode = stream.end_capture()

                # Embed this graph in our main graph
                return graph.graph.add_graph(gnode, deps)

            def run(self, stream):
                gemm(stream)

        return MulKernel(mats=[a, b, out], dt=dt)