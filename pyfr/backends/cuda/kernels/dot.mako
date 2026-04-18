<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

__global__ __launch_bounds__(${blocksz}) void
dot(ixdtype_t nrow, ixdtype_t ncolb, ixdtype_t ldim,
    fpdtype_t *__restrict__ reduced,
    fpdtype_t *__restrict__ r1, fpdtype_t *__restrict__ r2)


{
    int tid = threadIdx.x;
    ixdtype_t i = ixdtype_t(blockIdx.x)*blockDim.x + tid;

    __shared__ fpdtype_t sdata[32];
    fpdtype_t r, acc = 0;

    if (i < ncolb)
    {
        for (ixdtype_t j = 0; j < nrow; j++)
        {
            ixdtype_t idx = j*ldim + SOA_IX(i, blockIdx.y, gridDim.y);
            % if not scaling:
                r = r1[idx]*r2[idx];
            % else:
                r = r1[blockIdx.y]*fabsf(r2[idx]);
            % endif
            acc += r;
        }
    }

    // Reduce within each warp
    for (int off = warpSize / 2; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFFU, acc, off);

    // Have the first thread in each warp write out to shared memory
    if (tid % warpSize == 0)
        sdata[tid / warpSize] = acc;

    __syncthreads();

    // Have the first warp perform the final reduction
    if (tid / warpSize == 0)
    {
        acc = (tid < blockDim.x / warpSize) ? sdata[tid] : 0;

        for (int off = warpSize / 2; off > 0; off >>= 1)
            acc += __shfl_down_sync(0xFFFFFFFFU, acc, off);
        
        if (tid == 0)
            reduced[ixdtype_t(blockIdx.y)*gridDim.x + blockIdx.x] = acc;
    }
}