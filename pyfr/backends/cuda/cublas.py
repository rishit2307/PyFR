from ctypes import POINTER, c_int, c_double, c_float, c_void_p, c_uint64, byref, pointer

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
        c_int, c_void_p),
        (c_int, 'cublasSetPointerMode_v2', c_void_p, c_int), 
        (c_int, 'cublasDaxpy_v2', c_void_p, c_int, POINTER(c_double), c_void_p, 
         c_int, c_void_p, c_int), 
        (c_int, 'cublasDnrm2_v2', c_void_p, c_int, c_void_p, c_int, c_void_p),
        (c_int, 'cublasDgetrfBatched', c_void_p, c_int, c_void_p, c_int, 
         c_void_p, c_void_p, c_int),
        (c_int, 'cublasDgetriBatched', c_void_p, c_int, c_void_p, c_int, 
         c_void_p, c_void_p, c_int, c_void_p, c_int),
         (c_int, 'cublasGetVersion_v2',c_void_p,  c_void_p), 
         (c_int, 'cublasIdamin_v2', c_void_p, c_int, c_void_p, c_int, c_void_p), 
         (c_int, 'cublasDasum_v2', c_void_p, c_int, c_void_p, c_int, c_void_p),


         (c_int, 'cublasSnrm2_v2', c_void_p, c_int, c_void_p, c_int, c_void_p),
        (c_int, 'cublasSgetrfBatched', c_void_p, c_int, c_void_p, c_int, 
         c_void_p, c_void_p, c_int),
        (c_int, 'cublasSgetriBatched', c_void_p, c_int, c_void_p, c_int, 
         c_void_p, c_void_p, c_int, c_void_p, c_int),
         (c_int, 'cublasSasum_v2', c_void_p, c_int, c_void_p, c_int, c_void_p),
         (c_int, 'cublasSdot_v2', c_void_p, c_int, c_void_p, c_int, c_void_p, 
        c_int, c_void_p),
        (c_int, 'cublasSaxpy_v2', c_void_p, c_int, POINTER(c_double), c_void_p, 
         c_int, c_void_p, c_int),
         (c_int, 'cublasDgemv_v2', c_void_p, c_int, c_int, c_int, POINTER(c_double), 
          c_void_p, c_int, c_void_p, c_int, POINTER(c_double), c_void_p, c_int)
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

    
    def norm1(self, a):
        cuda = self.backend.cuda
        w, h = self.lib, self.handle
        fpdtype = a.traits[-1]
        cublasnrm2 = w.cublasDasum if fpdtype == np.float64 else w.cublasSasum

        n = a.nrow*a.ncol
        x = a
        w.cublasSetPointerMode(h, w.CUBLAS_POINTER_MODE_DEVICE)
        rdev = cuda.mem_alloc(np.dtype(fpdtype).itemsize)
        rhost = cuda.pagelocked_empty((), fpdtype)

        def l1(stream):
            w.cublasSetStream(h, stream)
            cublasnrm2(h, n, x, 1, rdev)

        class NormKernel(CUDAKernel):
            def run(self, stream):
                l1(stream)
                cuda.memcpy(rhost, rdev, rdev.nbytes, stream)
            
            @property
            def retval(self):
                return rhost

        return NormKernel(mats=[a])

    def gemv(self, a, b, c):
        cuda = self.backend.cuda
        w, h = self.lib, self.handle

        cublasgemv = w.cublasDgemv
        alpha = 1
        m, n = a.nrow, a.ncol
        lda = n
        beta=  1.0

        def mv(stream):
            w.cublasSetStream(h, stream)
            cublasgemv(h, w.OP_N, m, n, c_double(alpha), a, lda, b, 1, c_double(beta), c, 1)
        
        class GEMVKernel(CUDAKernel):
            def run(self, stream):
                mv(stream)
        
        return GEMVKernel(mats=[a, b, c])



    def norm2(self, a):
        cuda = self.backend.cuda
        w, h = self.lib, self.handle

        fpdtype = a.traits[-1]
        cublasnrm2 = w.cublasDnrm2 if fpdtype == np.float64 else w.cublasSnrm2
        n = a.nrow*a.ncol
        x = a
        w.cublasSetPointerMode(h, w.CUBLAS_POINTER_MODE_DEVICE)
        rdev = cuda.mem_alloc(np.dtype(fpdtype).itemsize)

        rhost = cuda.pagelocked_empty((), fpdtype)

        def l2(stream):
            w.cublasSetStream(h, stream)
            cublasnrm2(h, n, x, 1, rdev)

        class NormKernel(CUDAKernel):
            def run(self, stream):
                l2(stream)
                cuda.memcpy(rhost, rdev, rdev.nbytes, stream)
            
            @property
            def retval(self):
                return rhost

        return NormKernel(mats=[a])

    def dot(self, a, b):
        cuda = self.backend.cuda
        fpdtype = a.traits[-1]
        w, h = self.lib, self.handle
        cublasdot = w.cublasDdot if fpdtype == np.float64 else w.cublasSdot
        n = a.nrow*a.ncol
        x, y = a, b
        w.cublasSetPointerMode(h, w.CUBLAS_POINTER_MODE_DEVICE)
        rdev = cuda.mem_alloc(np.dtype(float).itemsize)

        rhost = cuda.pagelocked_empty((), fpdtype)

        # ckey = (n)

        def ddd(stream):
            w.cublasSetStream(h, stream)
            cublasdot(h, n, x, 1, y, 1, rdev)

        class DotKernel(CUDAKernel):
            def run(self, stream):
                ddd(stream)
                cuda.memcpy(rhost, rdev, rdev.nbytes, stream)

            @property
            def retval(self):
                return rhost
        
        return DotKernel(mats=[a, b])

    def lu(self, *arr):
        cuda = self.backend.cuda
        w, h = self.lib, self.handle
        a = arr[0]
        fpdtype = a.traits[-1]
        P = arr[1]
        sz = np.dtype(a.traits[-1]).itemsize
        batchsize = a.nrow
        n = int(np.sqrt(a.ncol))

        cdptr = cuda.mem_alloc(np.dtype(np.uintp).itemsize)
        adptr = cuda.mem_alloc(batchsize*np.dtype(np.uintp).itemsize)
        ahptr = np.ascontiguousarray([a.data + i*sz*a.leaddim for i in range(batchsize)], dtype=np.uintp)

        cuda.memcpy(adptr, ahptr, adptr.nbytes)
        cublasgetrf = w.cublasDgetrfBatc if fpdtype == np.float64 else w.cublasSgetrfBatc
        # cublasgetrf = w.cublasSgetrfBatc
        print(fpdtype)

        def lu(stream):
            w.cublasSetStream(h, stream)
            # cublasgetrf(h, n, adptr, n,  cdptr, batchsize)
            cublasgetrf(h, n, adptr, n, P, cdptr, batchsize)
        class LUKernel(CUDAKernel):
            def run(self, stream):
                lu(stream)

        return LUKernel(mats=[a])
    
    def inv(self, *arr):
        cuda = self.backend.cuda
        a, b, P = arr
        w, h = self.lib, self.handle
        sz = np.dtype(a.traits[-1]).itemsize    
        fpdtype = a.traits[-1]

        

        
        batchsize = a.nrow
        n = int(np.sqrt(a.ncol))

        cdptr = cuda.mem_alloc(np.dtype(np.uint32).itemsize)
        bdptr = cuda.mem_alloc(batchsize*np.dtype(np.uintp).itemsize)
        adptr = cuda.mem_alloc(batchsize*np.dtype(np.uintp).itemsize)
        bhptr = np.ascontiguousarray([b.data + i*sz*a.leaddim for i in range(batchsize)], dtype=np.uintp)
        ahptr = np.ascontiguousarray([a.data + i*sz*a.leaddim for i in range(batchsize)], dtype=np.uintp)

        cuda.memcpy(bdptr, bhptr, bdptr.nbytes)
        cuda.memcpy(adptr, ahptr, adptr.nbytes)

        cublasgetri = w.cublasDgetriBatc if fpdtype == np.float64 else w.cublasSgetriBatc
        # cublasgetri=  w.cublasSgetriBatc

        def getinv(stream):
            w.cublasSetStream(h, stream)
            cublasgetri(h, n, adptr, n, P, bdptr, n, cdptr, batchsize)
        class InvKernel(CUDAKernel):
            def run(self, stream):
                getinv(stream)

        return InvKernel(mats=[a, b])
            
    def amin(self, a, b):
        cuda = self.backend.cuda
        w, h = self.lib, self.handle
        n = a.ncol*a.nrow

        cublasamin = w.cublasIdamin
        w.cublasSetPointerMode(h, w.CUBLAS_POINTER_MODE_HOST)

        # rdev = cuda.mem_alloc(np.dtype(np.int32).itemsize)
        # rhost = cuda.pagelocked_empty((), np.int32)
        


        def ver(stream):
            w.cublasSetStream(h, stream)
            cublasamin(h, n, a, 1, b)
        
        class VKernel(CUDAKernel):
            def run(self, stream):
                ver(stream)
                # cuda.memcpy(rhost, rdev, rdev.nbytes, stream)
            
            # @property
            # def retval(self):
            #     return rhost
        
        return VKernel()


        




    # def mul(self, a, b, out, alpha=1.0, beta=0.0, gmres=False):
    #     cuda = self.backend.cuda
    #     w, h = self.lib, self.handle
    #     # print(f'a.nrow is {a.nrow}')
    #     # print(f'out.nrow is {out.nrow}')
    #     # print(f'a.ncol is {a.ncol}')
    #     # print(f'b.nrow is {b.nrow}')
    #     # print(f'b.ncol is {b.ncol}')
    #     # print(f'out.ncol is {out.ncol}')
    #     # print(f'A leaddim is {a.leaddim}')
    #     # print(f'B leaddim is {b.leaddim}')
    #     # print(f'C leaddim is {out.leaddim}')
    #     # print(f'gmres is {gmres}')

    #     # Ensure the matrices are compatible
    #     # if a.nrow != out.nrow or a.ncol != b.nrow or b.ncol != out.ncol:
    #     #     raise ValueError('Incompatible matrices for out = a*b')

    #     # cuBLAS expects inputs to be column-major (or Fortran order in
    #     # numpy parlance).  However as C = A*B => C^T = (A*B)^T
    #     # = (B^T)*(A^T) with a little trickery we can multiply our
    #     # row-major matrices directly.
    #     m =  b.ncol if not gmres else 1
    #     n, k =  a.nrow, a.ncol


    #     A, B, C = b, a, out
    #     # Cache key
    #     if gmres:
    #         ckey = (A.dtype, alpha, beta, m, n, k, 1, B.leaddim, C.leaddim)
    #     else:
    #         ckey = (A.dtype, alpha, beta, m, n, k, A.leaddim, B.leaddim, C.leaddim)

    #     # Size checks
    #     if any(sz > 2**31 - 1 for sz in ckey[3:]):
    #         raise ValueError('Matrices too large for cuBLAS')

    #     # α and β factors for C = α*(A*B) + β*C
    #     if a.dtype == np.float64:
    #         cublasgemm = w.cublasDgemm
    #         alpha_ct, beta_ct = c_double(alpha), c_double(beta)
    #     else:
    #         cublasgemm = w.cublasSgemm
    #         alpha_ct, beta_ct = c_float(alpha), c_float(beta)

    #     # Convenience wrapper
    #     def gemm(stream):
    #         w.cublasSetStream(h, stream)
    #         if gmres:
    #             # print(f'h is {h}')
    #             # print(f'm is {m}')
    #             # print(f'n is {n}')
    #             # print(f'k is {k}')
    #             # print(f'alpha_ct is {alpha_ct}')
    #             # print(f'A is {A}')
    #             # print(f'A.leaddim is {1}')
    #             # print(f'B is {B}')
    #             # print(f'B.leaddim is {B.leaddim}')
    #             # print(f'beta is {beta_ct}')
    #             # print(f'C is {C}')
    #             # print(f'C.leaddim is {C.leaddim}')

    #             cublasgemm(h, w.OP_N, w.OP_N, m, n, k, alpha_ct, A, 1,
    #                    B, B.leaddim, beta_ct, C, C.leaddim)
    #         else:
    #             cublasgemm(h, w.OP_N, w.OP_N, m, n, k, alpha_ct, A, A.leaddim,
    #                    B, B.leaddim, beta_ct, C, C.leaddim)


    #     # Obtain the performance of the kernel
    #     try:
    #         dt = self._mul_timing[ckey]
    #     except KeyError:
    #         # Save a copy of the contents of the output matrix
    #         out_np = getattr(out, 'parent', out).get()

    #         # Benchmark the kernel and update the cache
    #         self._mul_timing[ckey] = dt = self._benchmark(gemm)

    #         # Restore the output matrix
    #         getattr(out, 'parent', out).set(out_np)

    #     class MulKernel(CUDAKernel):
    #         def add_to_graph(self, graph, deps):
    #             stream = cuda.create_stream()

    #             # Capture the execution of cuBLAS to obtain a graph
    #             stream.begin_capture()
    #             gemm(stream)
    #             gnode = stream.end_capture()

    #             # Embed this graph in our main graph
    #             return graph.graph.add_graph(gnode, deps)

    #         def run(self, stream):
    #             gemm(stream)

    #     return MulKernel(mats=[a, b, out], dt=dt)