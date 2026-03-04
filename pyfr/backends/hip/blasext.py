import numpy as np
import math

from pyfr.backends.hip.provider import (HIPKernel, HIPKernelProvider,
                                        get_grid_for_block)


class HIPBlasExtKernels(HIPKernelProvider):
    def axnpby(self, *arr, subdims=None):
        if any(arr[0].traits != x.traits for x in arr[1:]):
            raise ValueError('Incompatible matrix types')

        nv = len(arr)
        ixdtype = self.backend.ixdtype
        nrow, ncol, ldim, fpdtype = arr[0].traits[1:]
        ncola, ncolb = arr[0].ioshape[1:]

        # Determine the grid/block
        block = (128, 1, 1)
        grid = get_grid_for_block(block, ncolb, nrow)

        # Render the kernel template
        src = self.backend.lookup.get_template('axnpby').render(
            block=block, subdims=subdims or range(ncola), ncola=ncola, nv=nv
        )

        # Build the kernel
        kern = self._build_kernel('axnpby', src,
                                  [ixdtype]*3 + [np.uintp]*nv + [fpdtype]*nv)

        # Set the parameters
        params = kern.make_params(grid, block)
        params.set_args(nrow, ncolb, ldim, *arr)

        class AxnpbyKernel(HIPKernel):
            def bind(self, *consts):
                params.set_args(*consts, start=3 + nv)

            def run(self, stream):
                kern.exec_async(stream, params)

        return AxnpbyKernel(mats=arr)

    def copy(self, dst, src):
        hip = self.backend.hip

        if dst.traits != src.traits:
            raise ValueError('Incompatible matrix types')

        class CopyKernel(HIPKernel):
            def add_to_graph(self, graph, deps):
                pass

            def run(self, stream):
                hip.memcpy(dst, src, dst.nbytes, stream)

        return CopyKernel(mats=[dst, src])

    def reduction(self, *rs, method, norm=None, dt_mat=None):
        if any(r.traits != rs[0].traits for r in rs[1:]):
            raise ValueError('Incompatible matrix types')

        hip = self.backend.hip
        ixdtype = self.backend.ixdtype
        nrow, ncol, ldim, fpdtype = rs[0].traits[1:]
        ncola, ncolb = rs[0].ioshape[1:]

        # Reduction block dimensions
        block = (128, 1, 1)

        # Determine the grid size
        grid = get_grid_for_block(block, ncolb, ncola)

        # Empty result buffer on the device
        reduced_dev = hip.mem_alloc(ncola*grid[0]*rs[0].itemsize)

        # Empty result buffer on the host
        reduced_host = hip.pagelocked_empty((ncola, grid[0]), fpdtype)

        tplargs = dict(norm=norm, blocksz=block[0], method=method)

        if method == 'resid':
            tplargs['dt_type'] = 'matrix' if dt_mat else 'scalar'

        # Get the kernel template
        src = self.backend.lookup.get_template('reduction').render(**tplargs)

        regs = list(rs) + [dt_mat] if dt_mat else rs

        # Argument types for reduction kernel
        if method == 'errest':
            argt = [ixdtype]*3 + [np.uintp]*4 + [fpdtype]*2
        elif method == 'errest_imp':
            argt = [ixdtype]*3 + [np.uintp]*3 + [fpdtype]*2
        elif method == 'resid' and dt_mat:
            argt = [ixdtype]*3 + [np.uintp]*4 + [fpdtype]
        elif method == 'dot':
            argt = [ixdtype]*3 + [np.uintp]*3
        elif method == 'norm':
            argt = [ixdtype]*3 + [np.uintp]*2
        else:
            argt = [ixdtype]*3 + [np.uintp]*3 + [fpdtype]

        # Build the reduction kernel
        rkern = self._build_kernel('reduction', src, argt)

        # Set the parameters
        params = rkern.make_params(grid, block)
        params.set_args(nrow, ncolb, ldim, reduced_dev, *regs)

        # Runtime argument offset
        if fpdtype in argt:
            facoff = argt.index(fpdtype)

        # Norm type
        reducer = np.max if norm == 'uniform' else np.sum

        class ReductionKernel(HIPKernel):
            @property
            def retval(self):
                return reducer(reduced_host, axis=1)

            def bind(self, *facs):
                params.set_args(*facs, start=facoff)

            def run(self, stream):
                rkern.exec_async(stream, params)
                hip.memcpy(reduced_host, reduced_dev, reduced_dev.nbytes,
                           stream)

        return ReductionKernel(mats=regs)
    
    def dot(self, *rs):
        hip = self.backend.hip
        ixdtype = self.backend.ixdtype
        nrow, ncol, ldim, fpdtype = rs[0].traits[1:]
        ncola, ncolb = rs[0].ioshape[1:]

        # Reduction block dimensions
        block = (128, 1, 1)

        # Determine the grid size
        grid = get_grid_for_block(block, ncolb, ncola)

        # Empty result buffer on the device
        reduced_dev = hip.mem_alloc(ncola*grid[0]*rs[0].itemsize)

        # Empty result buffer on the host
        reduced_host = hip.pagelocked_empty((ncola, grid[0]), fpdtype)

        tplargs = dict(blocksz=block[0])

        # Get the kernel template
        src = self.backend.lookup.get_template('dot').render(**tplargs)

        regs = list(rs)

        argt = [ixdtype]*3 + [np.uintp]*3

        # Build the reduction kernel
        rkern = self._build_kernel('dot', src, argt)

        # Set the parameters
        params = rkern.make_params(grid, block)
        params.set_args(nrow, ncolb, ldim, reduced_dev, *regs)


        class DOTKernel(HIPKernel):
            @property
            def retval(self):
                return np.sum(reduced_host, axis=1)

            def bind(self, *facs):
                pass

            def run(self, stream):
                rkern.exec_async(stream, params)
                hip.memcpy(reduced_host, reduced_dev, reduced_dev.nbytes,
                           stream)

        return DOTKernel(mats=regs)

    def addidx(self, *arr, subdims=None):
        nrow, ncol, ldim, fpdtype = arr[0].traits[1:]
        ncola, ncolb = arr[0].ioshape[1:]

        block = (128, 1, 1)
        grid = get_grid_for_block(block, ncolb, nrow)

        ixdtype = self.backend.ixdtype
        fpdtype = self.backend.fpdtype

        # Render the kernel template
        src = self.backend.lookup.get_template('addidx').render(
            ncola=ncola, subdims=subdims or range(ncola), block=block)

        # Build the kernel
        kern = self._build_kernel('addidx', src,
                                  [ixdtype]*2 + [np.uintp]*3 + [ixdtype]*3
                                  + [fpdtype])

        # Set the parameters
        params = kern.make_params(grid, block)
        params.set_args(ncolb, ldim, *arr)

        class AddidxKernel(HIPKernel):
            def bind(self, *consts):
                params.set_args(*consts, start=5)

            def run(self, stream):
                kern.exec_async(stream, params)

        return AddidxKernel(mats=arr)    

    def jacinit(self, *arr):
        nrow, ncol, ldim, fpdtype = arr[0].traits[1:]
        ncola, ncolb = arr[0].ioshape[1:]

        # nrowj, ncolj, ldimj, fpdtypej = arr[1].traits[1:]
        ldimj = arr[2].ioshape[0]

        block = (128, 1, 1)
        grid = get_grid_for_block(block, ncolb, ldimj)
        ldimj  = ldimj**2


        ixdtype = self.backend.ixdtype
        fpdtype = self.backend.fpdtype

        # Render the kernel template
        src = self.backend.lookup.get_template('jacinit').render(
            ncola=ncola, block=block)

        # Build the kernel
        kern = self._build_kernel('jacinit', src,
               [ixdtype]*4 +[np.uintp]*4 + 
               [ixdtype]*3 + [fpdtype]*2)

        # Set the parameters
        params = kern.make_params(grid, block)
        params.set_args(nrow, ncolb, ldim, ldimj, *arr)

        class JacInitKernel(HIPKernel):
            def bind(self, *consts):
                params.set_args(*consts, start=8)

            def run(self, stream):
                kern.exec_async(stream, params)

        return JacInitKernel(mats=arr)
    
    def jacshuffle(self, *arr):
        nrow, ncol, ldim0, fpdtype = arr[0].traits[1:]
        nrow, ncol, ldim1, fpdtype = arr[1].traits[1:]
        nupts, ncola, ncolb = arr[1].ioshape[1:]


        block = (128, 1, 1)
        grid = get_grid_for_block(block, ncolb, nupts*ncola, nupts*ncola)

        ixdtype = self.backend.ixdtype
        fpdtype = self.backend.fpdtype

        # Render the kernel template
        src = self.backend.lookup.get_template('jacshuffle').render(
            ncola=ncola, block=block)

        # Build the kernel
        kern = self._build_kernel('jacshuffle', src,
                                  [ixdtype]*3 + [np.uintp]*2)

        # Set the parameters
        params = kern.make_params(grid, block)
        params.set_args(ncolb, ldim0, ldim1, *arr)

        class JacShuffleKernel(HIPKernel):
            def bind(self, *consts):
                pass

            def run(self, stream):
                kern.exec_async(stream, params)

        return JacShuffleKernel(mats=arr)
    
    def jacmul_clust(self, *arr):
        ixdtype = self.backend.ixdtype

        nclust = arr[2].traits[1]
        ncol = np.prod(arr[0].ioshape[:-1])
        neles = arr[0].ioshape[-1]
        nrow, ncola, ncolb = arr[0].ioshape
        nrow, ldim2 = arr[0].traits[1:3]

        eleclust = neles // nclust

        block = (256, 1, 1)

        bm, bn = 160, 128
        bk = 16
        grid = (nclust, -(-ncol //bm), -(-eleclust//bn))
        tplargs = dict()
        tplargs['_macros'] = {}
        # grid = (nclust, 1, 1)
        # src = self.backend.lookup.get_template('jacmulclustv6').render(
        # ncol=ncol, ncola=ncola, ldim2=ldim2, blkx=block[0], tot_neles=neles, 
        # bm=bm, bn=bn, bk=bk, ldim=ncol**2, 
        # wm=32, wn=64, wmiter=2, wniter=2, tm=4, tn=4)

        src = self.backend.lookup.get_template('jacmulclustv7').render(
        ncol=ncol, ncola=ncola, ldim2=ldim2, blkx=block[0], tot_neles=neles, 
        bm=bm, bn=bn, bk=bk, ldim=ncol**2, **tplargs)

        with open("jacmul.cu", 'w') as f:
            print(src, file=f)
        # Build the kernel
        kern = self._build_kernel('jacmulclustv7', src, [np.uintp]*5)

        # Set the parameters 

        params = kern.make_params(grid, block)
        params.set_args(*arr)

        class JacMulClustKernel(HIPKernel):
            def bind(self, *consts):
                pass

            def run(self, stream):
                kern.exec_async(stream, params)

        return JacMulClustKernel(mats=arr)
    

    def jacmul(self, *arr):
        ixdtype = self.backend.ixdtype
        nrow, ncol, ldim, fpdtype = arr[0].traits[1:]
        ncola, ncolb = arr[0].ioshape[1:]

        jac_fpdtype = arr[-1].traits[-1]

        # Determine the grid/block
        block = (64, 16, 1)
        blksz = 256 if fpdtype == np.float32 else 128
        blky = block[1]

        grid = get_grid_for_block(block, ncolb)
        blkx = block[0]

        # Render the kernel template
        src = self.backend.lookup.get_template('jacmul').render(
             ncola=ncola, jac_fpdtype=jac_fpdtype, blkx=blkx, 
             blky=blky, blksz=blksz, ndof=nrow*ncola)

        # Build the kernel
        kern = self._build_kernel('jacmul', src,
                                [ixdtype]*3 + [np.uintp]*3)

        # Set the parameters
        params = kern.make_params(grid, block)
        params.set_args(nrow, ncolb, ldim, *arr)

        class JacMulKernel(HIPKernel):
            def bind(self, *consts):
                pass

            def run(self, stream):
                kern.exec_async(stream, params)

        return JacMulKernel(mats=arr)

    def getf3(self, *arr):
        ixdtype = self.backend.ixdtype
        neles, nrow, ldim, fpdtype = arr[1].traits[1:]
        nrow = math.isqrt(nrow)
        ncol = 1
        # neles, ncol, ldim, fpdytype = arr[0].traits[1:]
        # nrow = ldim // ncol


        # Determine the grid/block
        block = (512, 1, 1)


        # grid = get_grid_for_block(block, ncolb, nrow*ncola)
        grid = (neles, 1, 1)
        tplargs = {'nrow':nrow}

        
        tplargs['nby'] = block[0]//32
        tplargs['nb'] = 32
        tplargs['blksz'] = block[0]
        tplargs['_macros'] = {}

        # Render the kernel template
        src = self.backend.lookup.get_template('getf3').render(
            **tplargs
        )

         # Build the kernel
        kern = self._build_kernel('getf3', src,
                                [ixdtype]*3 + [np.uintp]*3)

        # Set the parameters
        params = kern.make_params(grid, block)
        params.set_args(nrow, ncol, ldim, *arr)

        class GetF3Kernel(HIPKernel):
            def bind(self, *consts):
                pass

            def run(self, stream):
                kern.exec_async(stream, params)

        return GetF3Kernel(mats=arr)