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
        (c_int, 'cublasSetPointerMode_v2', c_void_p, c_int), 
        (c_int, 'cublasDgetrfBatched', c_void_p, c_int, c_void_p, c_int, 
         c_void_p, c_void_p, c_int),
        (c_int, 'cublasDgetriBatched', c_void_p, c_int, c_void_p, c_int, 
         c_void_p, c_void_p, c_int, c_void_p, c_int),
         (c_int, 'cublasGetVersion_v2',c_void_p,  c_void_p), 
         (c_int, 'cublasDasum_v2', c_void_p, c_int, c_void_p, c_int, POINTER(c_double)),
        (c_int, 'cublasSasum_v2', c_void_p, c_int, c_void_p, c_int, POINTER(c_float)),
         (c_int, 'cublasSnrm2_v2', c_void_p, c_int, c_void_p, c_int, POINTER(c_float)),
        (c_int, 'cublasDnrm2_v2', c_void_p, c_int, c_void_p, c_int, POINTER(c_double)),
        (c_int, 'cublasSgetrfBatched', c_void_p, c_int, c_void_p, c_int, 
         c_void_p, c_void_p, c_int),
        (c_int, 'cublasSgetriBatched', c_void_p, c_int, c_void_p, c_int, 
         c_void_p, c_void_p, c_int, c_void_p, c_int),
         (c_int, 'cublasSdot_v2', c_void_p, c_int, c_void_p, c_int, c_void_p, 
        c_int, POINTER(c_float)),
        (c_int, 'cublasDdot_v2', c_void_p, c_int, c_void_p, c_int, c_void_p, 
        c_int, POINTER(c_double)),
        (c_int, 'cublasDgemmBatched_v2', c_void_p, c_int, c_int, c_int, c_int, c_int,
         POINTER(c_float), POINTER(c_void_p), c_int, POINTER(c_void_p), c_int,
         POINTER(c_float), POINTER(c_void_p), c_int, c_int)
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
    
    def norm1(self, a):
        w, h = self.lib, self.handle
        fpdtype = a.traits[-1]
        cublasnrm2 = w.cublasDasum if fpdtype == np.float64 else w.cublasSasum

        n = a.nrow*a.ncol
        x = a

        if fpdtype == np.float64:
            result = c_double(0.0)
        else:
            result = c_float(0.0)

        def l1(stream):
            w.cublasSetStream(h, stream)
            cublasnrm2(h, n, x, 1, result)

        class NormKernel(CUDAKernel):
            def run(self, stream):
                l1(stream)

            @property
            def retval(self):
                return fpdtype(result)

        return NormKernel(mats=[a])

    def norm2(self, a):
        w, h = self.lib, self.handle

        fpdtype = a.traits[-1]
        cublasnrm2 = w.cublasDnrm2 if fpdtype == np.float64 else w.cublasSnrm2
        n = a.nrow*a.ncol
        x = a
        if fpdtype == np.float64:
            result = c_double(0.0)
        else:
            result = c_float(0.0)

        def l2(stream):
            w.cublasSetStream(h, stream)
            cublasnrm2(h, n, x, 1, byref(result))

        class NormKernel(CUDAKernel):
            def run(self, stream):
                l2(stream)

            @property
            def retval(self):
                return fpdtype(result)

        return NormKernel(mats=[a])

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
    


    def mul(self, a, b, out, alpha=1.0, beta=0.0):
        cuda = self.backend.cuda
        w, h = self.lib, self.handle

        m = 256
        n = 160
        k = 160

        # α and β factors for C = α*(A*op(B)) + β*C
        if a.dtype == np.float64:
            cublasgemm = w.cublasDgemmBatched
            alpha_ct, beta_ct = c_double(alpha), c_double(beta)
        else:
            cublasgemm = w.cublasSgemmBatched
            alpha_ct, beta_ct = c_float(alpha), c_float(beta)

        class MulKernel(CUDAKernel):
            def add_to_graph(self, graph, deps):
                stream = cuda.create_stream()

                # Capture the execution of cuBLAS to obtain a graph
                stream.begin_capture()
                self.run(stream)
                gnode = stream.end_capture()

                # Embed this graph in our main graph
                return graph.graph.add_graph(gnode, deps)

            def run(self, stream):
                w.cublasSetStream(h, stream)
                cublasgemm(h, w.OP_N, w.OP_N, m, n, k,
                           alpha_ct, b, 128, a, 160,
                           beta_ct, out, 128, 50)

        return MulKernel(mats=[a, b, out])