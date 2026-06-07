import numpy as np
import math

from pyfr.backends.cuda.provider import (CUDAKernel, CUDAKernelProvider,
                                         get_grid_for_block)
from pyfr.mpiutil import get_comm_rank_root

class CUDABlasExtKernels(CUDAKernelProvider):
    def axnpby(self, *arr, subdims=None, inscales, in_idx,
               outscales):
        if any(arr[0].traits != x.traits for x in arr[1:]):
            raise ValueError('Incompatible matrix types')

        nv = len(arr)
        ixdtype = self.backend.ixdtype
        nrow, ncol, ldim, fpdtype = arr[0].traits[1:]
        ncola, ncolb = arr[0].ioshape[1:]

        # Render the kernel template
        src = self.backend.lookup.get_template('axnpby').render(
            subdims=subdims or range(ncola), ncola=ncola, nv=nv,
            inscales=inscales, outscales=outscales, in_idx=in_idx)

        # Build the kernel
        kern = self._build_kernel('axnpby', src,
                                  [ixdtype]*3 + [np.uintp]*nv + [fpdtype]*nv)

        # Determine the grid/block
        block = (128, 1, 1)
        grid = get_grid_for_block(block, ncolb, nrow)

        # Set the parameters
        params = kern.make_params(grid, block)
        params.set_args(nrow, ncolb, ldim, *arr)

        class AxnpbyKernel(CUDAKernel):
            def bind(self, *consts):
                params.set_args(*consts, start=3 + nv)

            def run(self, stream):
                kern.exec_async(stream, params)

        return AxnpbyKernel(mats=arr)

    def copy(self, dst, src):
        cuda = self.backend.cuda

        if dst.traits != src.traits:
            raise ValueError('Incompatible matrix types')

        class CopyKernel(CUDAKernel):
            def add_to_graph(self, graph, deps):
                return graph.graph.add_memcpy(dst, src, dst.nbytes, deps)

            def run(self, stream):
                cuda.memcpy(dst, src, dst.nbytes, stream)

        return CopyKernel(mats=[dst, src])

    def reduction(self, *rs, method, norm=None, dt_mat=None):
        if any(r.traits != rs[0].traits for r in rs[1:]):
            raise ValueError('Incompatible matrix types')

        cuda = self.backend.cuda
        ixdtype = self.backend.ixdtype
        nrow, ncol, ldim, fpdtype = rs[0].traits[1:]
        ncola, ncolb = rs[0].ioshape[1:]

        # Reduction block dimensions
        block = (256, 1, 1)

        # Determine the grid size
        grid = get_grid_for_block(block, ncolb, ncola)

        # Empty result buffer on the device
        reduced_dev = cuda.mem_alloc(ncola*grid[0]*rs[0].itemsize)

        # Empty result buffer on the host
        reduced_host = cuda.pagelocked_empty((ncola, grid[0]), fpdtype)

        tplargs = dict(norm=norm, method=method)
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

        class ReductionKernel(CUDAKernel):
            @property
            def retval(self):
                return reducer(reduced_host, axis=1)

            def bind(self, *facs):
                params.set_args(*facs, start=facoff)

            def run(self, stream):
                rkern.exec_async(stream, params)
                cuda.memcpy(reduced_host, reduced_dev, reduced_dev.nbytes,
                            stream)

        return ReductionKernel(mats=regs)

    def dot(self, *rs):
        cuda = self.backend.cuda
        ixdtype = self.backend.ixdtype
        nrow, ncol, ldim, fpdtype = rs[1].traits[1:]
        ncola, ncolb = rs[1].ioshape[1:]
        comm, rank, root = get_comm_rank_root()
        # Reduction block dimensions
        block = (256, 1, 1)
        nv = len(rs)
        # Determine the grid size
        grid = get_grid_for_block(block, ncolb, ncola)

        # Empty result buffer on the device
        reduced_dev = cuda.mem_alloc(ncola*grid[0]*(nv-1)*rs[1].itemsize)

        # Empty result buffer on the host
        reduced_host = cuda.pagelocked_empty((ncola, grid[0], nv-1), fpdtype)

        tplargs = dict(blocksz=block[0])

        # Get the kernel template
        src = self.backend.lookup.get_template('dot').render(nv = nv,
                                                             **tplargs)
        regs = list(rs)
        argt = [ixdtype]*3 + [np.uintp]*(nv+1)

        # Build the reduction kernel
        rkern = self._build_kernel('dot', src, argt)

        # Set the parameters
        params = rkern.make_params(grid, block)
        params.set_args(nrow, ncolb, ldim, reduced_dev, *regs)

        class DOTKernel(CUDAKernel):
            @property
            def retval(self):
                reduced = np.sum(reduced_host, axis=(0, 1))

                return reduced

            def bind(self):
                pass

            def run(self, stream):
                rkern.exec_async(stream, params)
                cuda.memcpy(reduced_host, reduced_dev, 
                                reduced_dev.nbytes, stream)

        return DOTKernel(mats=regs)

    def addidx(self, *arr, subdims=None):
        nrow, ncol, ldim, fpdtype = arr[0].traits[1:]
        ncola, ncolb = arr[0].ioshape[1:]
        ecount = arr[-1].ioshape[-1]

        block = (128, 1, 1)
        grid = get_grid_for_block(block, ecount)

        ixdtype = self.backend.ixdtype
        fpdtype = self.backend.fpdtype

        # Render the kernel template
        src = self.backend.lookup.get_template('addidx').render(
            ncola=ncola, subdims=subdims or range(ncola), 
            ecount=ecount)

        # Build the kernel
        kern = self._build_kernel('addidx', src,
                                  [ixdtype]*2 + [np.uintp]*3
                                  + [ixdtype]*3 + [fpdtype])

        # Set the parameters
        params = kern.make_params(grid, block)
        params.set_args(ncolb, ldim, *arr)

        class AddidxKernel(CUDAKernel):
            def bind(self, *consts):
                params.set_args(*consts, start=5)

            def run(self, stream):
                kern.exec_async(stream, params)

        return AddidxKernel(mats=arr)

    def jacinit(self, *arr):
        nrow, ncol, ldim, fpdtype = arr[0].traits[1:]
        ncola, ncolb = arr[0].ioshape[1:]
        ecount = arr[-1].ioshape[-1]
        ldimj = np.prod(arr[0].ioshape[:2])

        # nrowj, ncolj, ldimj, fpdtypej = arr[1].traits[1:]
        # ldimj = arr[2].ioshape[0]

        block = (128, 1, 1)
        grid = get_grid_for_block(block, ecount, ldimj)
        ldimj  = ldimj**2


        ixdtype = self.backend.ixdtype
        fpdtype = self.backend.fpdtype

        # Render the kernel template
        src = self.backend.lookup.get_template('jacinit').render(
            ncola=ncola, ecount=ecount)

        # Build the kernel
        kern = self._build_kernel('jacinit', src,
               [ixdtype]*4 +[np.uintp]*4 + 
               [ixdtype]*3 + [fpdtype]*2)

        # Set the parameters
        params = kern.make_params(grid, block)
        params.set_args(nrow, ncolb, ldim, ldimj, *arr)

        class JacInitKernel(CUDAKernel):
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
            ncola=ncola)

        # Build the kernel
        kern = self._build_kernel('jacshuffle', src,
                                  [ixdtype]*3 + [np.uintp]*2)

        # Set the parameters
        params = kern.make_params(grid, block)
        params.set_args(ncolb, ldim0, ldim1, *arr)

        class JacShuffleKernel(CUDAKernel):
            def bind(self, *consts):
                pass

            def run(self, stream):
                kern.exec_async(stream, params)

        return JacShuffleKernel(mats=arr)
    
    def reshuff_clust(self, *arr):
        ixdtype = self.backend.ixdtype

        nclust = arr[1].traits[1]
        ncol = np.prod(arr[0].ioshape[:-1])
        neles = arr[0].ioshape[-1]
        nrow, ncola, ncolb = arr[0].ioshape
        nrow, ldim2 = arr[0].traits[1:3]

        eleclust = neles // nclust

        block = (256, 1, 1)

        bm, bn = 160, 128
        bk = 16

        grid = (nclust, -(-ncol //bm), -(-eleclust//bn))

        src = self.backend.lookup.get_template('reshuffclust').render(
            ncol=ncol, ncola=ncola, ldim2=ldim2, blkx=block[0], tot_neles=neles,
            bm=bm, bn=bn, bk=bk, ldim=ncol**2
        )

        kern = self._build_kernel('reshuffclust', src, [ixdtype]+ [np.uintp]*4)

        # Set the parameters
        params = kern.make_params(grid, block)
        params.set_args(nclust,  *arr)

        class ReshuffClust(CUDAKernel):
            def bind(self, *consts):
                pass

            def run(self, stream):
                kern.exec_async(stream, params)


        return ReshuffClust(mats=arr)

    def jacmul_clust(self, *arr, emap, clust_neles_pad,
                     clust_neles, stidx, stidx_pad,
                     inscales, outscales):

        ncol = np.prod(arr[1].ioshape[:-1])
        nrow, ncola = arr[1].ioshape[:-1]
        nrow, ldim2 = arr[1].traits[1:3]
        ncolb = arr[0].ioshape[-1]

        # ncol = math.isqrt(arr[2].ioshape[-1])
        # ncola=5
        # ncolb=arr[1].ioshape[-1]
        # ldim2=100

        jac_fpdtype = arr[2].traits[-1]

        celes = clust_neles_pad.get()
        cdiff = np.diff(celes)
        nnzix = np.flatnonzero(cdiff)+1
        csplit = np.split(celes, nnzix, axis=-1)

        block = (128, 1, 1)
        K, bm = 16, 128
        bk = 16


        tplargs = dict()
        tplargs['_macros'] = {}

        kerns, params = [], []
        compclust = 0
        tpl = self.backend.lookup.get_template('jacmulclustv9')

        for i, group in enumerate(csplit):
            val = group.flat[0]
            bn = min(val, 64)
            nclust = group.size

            grid = (-(-ncol // bm), -(-val //bn), nclust)
            src = tpl.render(
                ncol=ncol, ncola=ncola, ldim2=ldim2, blkx=block[0],
                ncolb=ncolb, bm=bm, bn=bn, bk=bk, ldim=ncol**2, 
                jac_fpdtype=jac_fpdtype, K=K, inscales=inscales, 
                outscales=outscales, eleclust=val, compclust=compclust,
                **tplargs
            )
            compclust += nclust

            with open(f"jacmul_{i}.cu", 'w') as f:
                print(src, file=f)
            # Build the kernel
            kern = self._build_kernel('jacmulclustv9', src, [np.uintp]*8)
            kerns.append(kern)


            # Set the parameters 
            param = kern.make_params(grid, block)
            param.set_args(*arr, emap, clust_neles, 
                           clust_neles_pad, stidx, stidx_pad)
            params.append(param)

        class JacMulClustKernel(CUDAKernel):
            def bind(self, *consts):
                pass

            def run(self, stream):
                for k, p in zip(kerns, params):
                    k.exec_async(stream, p)

        return JacMulClustKernel(mats=arr)

    def jacmul(self, *arr, inscales, outscales
               ):
        ixdtype = self.backend.ixdtype
        nrow, ncol, ldim, fpdtype = arr[0].traits[1:]
        ncola, ncolb = arr[-1].ioshape[2:]
        ldimj = arr[-1].traits[-2]
        comm, rank, root = get_comm_rank_root()
        jac_fpdtype = arr[-1].traits[-1]

        # Determine the grid/block
        block = (32, 16, 1)
        blksz = 320 if fpdtype == np.float32 else 160
        blky = block[1]

        grid = get_grid_for_block(block, ncolb)
        blkx = block[0]

        # Render the kernel template
        src = self.backend.lookup.get_template('jacmul').render(
             ncola=ncola, jac_fpdtype=jac_fpdtype, blkx=blkx, 
             blky=blky, blksz=blksz, ndof=nrow*ncola, 
             inscales=inscales, outscales=outscales)

        # Build the kernel
        kern = self._build_kernel('jacmul', src,
                                [ixdtype]*4 + [np.uintp]*3)

        # Set the parameters
        params = kern.make_params(grid, block)
        params.set_args(nrow, ncolb, ldim, ldimj, *arr)

        class JacMulKernel(CUDAKernel):
            def bind(self, *consts):
                pass

            def run(self, stream):
                kern.exec_async(stream, params)

        return JacMulKernel(mats=arr)

    def getf3(self, *arr, kmeans=False):
        ixdtype = self.backend.ixdtype
        jac_fpdtype = arr[1].traits[-1]

        ldim = arr[0].ioshape[1]
        nrow = math.isqrt(ldim)
        # neles, ncol, ldim, fpdytype = arr[0].traits[1:]
        # nrow = ldim // ncol
        ecount =  arr[-1].ioshape[-1]

        ncola = arr[1].ioshape[0] // arr[1].ioshape[1]
        ldim2 = arr[1].traits[2]
        ldimj = arr[1].traits[2]*arr[1].ioshape[1]

        # Determine the grid/block
        block = (512, 1, 1)

        grid = (ecount, 1, 1)
        tplargs = dict()
        tplargs['_macros'] = {}

        # Render the kernel template
        src = self.backend.lookup.get_template('getf3').render(
              nrow=nrow, blksz=block[0],ldim2=ldim2, ldimj=ldimj,
              ncola=ncola, kmeans=kmeans, jac_fpdtype=jac_fpdtype,
              ecount=ecount, 
              **tplargs
        )
        # Build the kernel
        kern = self._build_kernel('getf3', src,
                                [ixdtype]*2 + [np.uintp]*4
                                )

        # Set the parameters
        params = kern.make_params(grid, block)
        params.set_args(nrow, ldim, *arr)

        class GetF3Kernel(CUDAKernel):
            def run(self, stream):
                kern.exec_async(stream, params)

        return GetF3Kernel(mats=arr)

    def shuffle(self, *arr):
        nrow, ncol, ldim, fpdtype = arr[0].traits[1:]
        nrow, nvars, ncolb = arr[0].ioshape
        ldimout = arr[1].ioshape[-1]

        # Determine the grid/block
        block = (256, 1, 1)
        grid = get_grid_for_block(block, ncolb, nrow*nvars)

        # Render the kernel template
        src = self.backend.lookup.get_template('shuffle').render(
            ncola=nvars,
            ldimout=ldimout, blkx=block[0], ldim=ldim, 
            ncolb=ncolb)

        # Build the kernel
        kern = self._build_kernel('shuffle', src,
                                  [np.uintp]*6)

        # Set the parameters
        params = kern.make_params(grid, block)
        params.set_args(*arr)

        class ShuffleKernel(CUDAKernel):
            def bind(self, *consts):
                pass

            def run(self, stream):
                kern.exec_async(stream, params)

        return ShuffleKernel(mats=arr)

    def kmeans(self, *arr):
        ixdtype = self.backend.ixdtype
        nsamp, nfeat = arr[0].ioshape
        nclust = arr[1].ioshape[0]
        
        # Determine the grid/block
        block = (512, 1, 1)

        grid = (nsamp, 1, 1)
        tplargs = dict()
        tplargs['_macros'] = {}

        # Render the kernel template
        src = self.backend.lookup.get_template('kmeans').render(
              **tplargs
        )
        # Build the kernel
        kern = self._build_kernel('kmeans', src,
                                [ixdtype]*2 + [np.uintp]*4
                                + [ixdtype]*3)

        # Set the parameters
        params = kern.make_params(grid, block)
        params.set_args(*arr)

        class KmeansKernel(CUDAKernel):
            def bind(self, *consts):
                params.set_args(*consts, start=6)

            def run(self, stream):
                kern.exec_async(stream, params)

        return KmeansKernel(mats=arr)
    
    def sumfactor(self, *arr):
        nupts_in, nvars, neles = arr[0].ioshape
        nupts_out = arr[-1].ioshape[0]

        nupt1d_out, nupt1d_in= arr[1].ioshape
        lag1dsz = nupt1d_out*nupt1d_in

        block = (512, 1, 1)
        print(block)
        # grid = get_grid_for_block(block, neles, nvars)
        nrow, ncol, ldim, fpdtype = arr[0].traits[1:]
        nrow2, ncol2, ldim2, fpdtype = arr[-1].traits[1:]

        wsz = arr[0].itemsize
        s1sz = wsz*(nupts_out + nupt1d_out*nupt1d_in
                    + nupt1d_out*nupt1d_out*nupt1d_in)

        s2sz = lag1dsz*wsz
        maxshared = self.backend.cuda.max_shared_mem()
        clsblk = 16 if fpdtype==np.float64 else 32
        eblkls = [4, 8, 16, 32]


        eblk = (maxshared - s2sz) // s1sz

        ix = np.searchsorted(eblkls, eblk) - 1

        # tidrix = np.searchsorted(eblkls, nupt1d_out)+1
        # tidcix = np.searchsorted(eblkls, nupt1d_in)+1

        # tidr = eblkls[tidrix]
        # tidc = eblkls[tidcix]


        eblk = eblkls[ix]
        print(f'eblk is {eblk}')

        grid = (-(-neles // eblk), nvars, 1)

        tplargs = dict()
        tplargs['_macros'] = {}


        # Render the kernel template
        src = self.backend.lookup.get_template('sumfactorv2').render(
              nrowin=nupts_in, ncolb=neles, ncola=nvars,
              ldim=ldim, eblk=eblk, nrowout=nupts_out,
              lagsz=lag1dsz, n1din=nupt1d_in, n1dout=nupt1d_out,
              blkx=block[0], ldim2=ldim2, nby=max(block[0]//eblk,nupt1d_in**2),
              **tplargs
        )

        with open("factor.cu", 'w') as f:
            print(src, file=f)
        # Build the kernel
        kern = self._build_kernel('sumfactor', src, [np.uintp]*3
                                 )
    


        # Set the parameters
        params = kern.make_params(grid, block)
        params.set_args(*arr)

        class SumfactorKernel(CUDAKernel):
            def bind(self, *consts):
                pass

            def run(self, stream):
                kern.exec_async(stream, params)

        return SumfactorKernel(mats=arr)

