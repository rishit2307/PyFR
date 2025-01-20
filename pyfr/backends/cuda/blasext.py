import numpy as np

from pyfr.backends.cuda.provider import (CUDAKernel, CUDAKernelProvider,
                                         get_grid_for_block)


class CUDABlasExtKernels(CUDAKernelProvider):
    def axnpby(self, *arr, subdims=None):
        if any(arr[0].traits != x.traits for x in arr[1:]):
            raise ValueError('Incompatible matrix types')

        nv = len(arr)
        ixdtype = self.backend.ixdtype
        nrow, ncol, ldim, fpdtype = arr[0].traits[1:]
        ncola, ncolb = arr[0].ioshape[1:]

        # Render the kernel template
        src = self.backend.lookup.get_template('axnpby').render(
            subdims=subdims or range(ncola), ncola=ncola, nv=nv
        )

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

    def reduction(self, *rs, method, norm, dt_mat=None):
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
        elif method == 'resid' and dt_mat:
            argt = [ixdtype]*3 + [np.uintp]*4 + [fpdtype]
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

    def addidx(self, *arr):
        nrow, ncol, ldim, fpdtype = arr[0].traits[1:]
        ncola, ncolb = arr[0].ioshape[1:]

        block = (128, 1, 1)
        grid = get_grid_for_block(block, ncolb)

        ixdtype = self.backend.ixdtype
        fpdtype = self.backend.fpdtype

        # Render the kernel template
        src = self.backend.lookup.get_template('addidx').render(
            ncola=ncola)

        # Build the kernel
        kern = self._build_kernel('addidx', src,
                                  [ixdtype]*2 + [np.uintp]*2 + [ixdtype]*2)

        # Set the parameters
        params = kern.make_params(grid, block)
        params.set_args(ncolb, ldim, *arr)

        class AddidxKernel(CUDAKernel):
            def bind(self, *consts):
                params.set_args(*consts, start=4)

            def run(self, stream):
                kern.exec_async(stream, params)

        return AddidxKernel(mats=arr)
    
    def jacinit(self, *arr):
        nrow, ncol, ldim, fpdtype = arr[0].traits[1:]
        ncola, ncolb = arr[0].ioshape[1:]

        nrowj, ncolj, ldimj, fpdtypej = arr[1].traits[1:]

        block = (128, 1, 1)
        grid = get_grid_for_block(block, ncolb, int(np.sqrt(ldimj)))

        ixdtype = self.backend.ixdtype
        fpdtype = self.backend.fpdtype

        # Render the kernel template
        src = self.backend.lookup.get_template('jacinit').render(
            ncola=ncola)

        # Build the kernel
        kern = self._build_kernel('jacinit', src,
                                  [ixdtype]*4 + [np.uintp]*2 + [ixdtype]*2 + [fpdtype])

        # Set the parameters
        params = kern.make_params(grid, block)
        params.set_args(nrow, ncolb, ldim, ldimj, *arr)

        class JacInitKernel(CUDAKernel):
            def bind(self, *consts):
                params.set_args(*consts, start=6)

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
    

    def jacmul(self, *arr):
        ixdtype = self.backend.ixdtype
        nrow, ncol, ldim, fpdtype = arr[0].traits[1:]
        ncola, ncolb = arr[0].ioshape[1:]

        # Determine the grid/block
        block = (128, 1, 1)
        grid = get_grid_for_block(block, ncolb, nrow*ncola)

        # Render the kernel template
        src = self.backend.lookup.get_template('jacmul').render(
             ncola=ncola)

        # Build the kernel
        kern = self._build_kernel('jacmul', src,
                                [ixdtype]*3 + [np.uintp]*3)

        # Set the parameters
        params = kern.make_params(grid, block)
        params.set_args(nrow, ncolb, ldim, *arr)

        class JacMulKernel(CUDAKernel):
            def bind(self, *consts):
                pass

            def run(self, stream):
                kern.exec_async(stream, params)

        return JacMulKernel(mats=arr)


