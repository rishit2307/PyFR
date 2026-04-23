<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

__global__ __launch_bounds__(${blocksz}) void
dot(ixdtype_t nrow, ixdtype_t ncolb, ixdtype_t ldim,
    fpdtype_t* __restrict__ reduced,
    const fpdtype_t *__restrict__ r0,
    ${', '.join(f'const fpdtype_t* __restrict__ r{i}'
                   for i in range(1, nv))})

{
    int tid = threadIdx.x;
    ixdtype_t i = ixdtype_t(blockIdx.x)*blockDim.x + tid;

    __shared__ fpdtype_t sdata[${nv-1}][32];
    fpdtype_t acc[${nv-1}] = {0.0};
    fpdtype_t r = 0;

    if (i < ncolb)
    {
        for (ixdtype_t j = 0; j < nrow; j++)
        {
            ixdtype_t idx = j*ldim + SOA_IX(i, blockIdx.y, gridDim.y);
            % if not scaling:
                r = r0[idx];
                % for i in range(1, nv):
                    acc[${i-1}] += r*${f'r{i}[idx]'};
                % endfor
            % else:
                acc[0] += r0[blockIdx.y]*fabs(r1[idx]);
            % endif
        }
    }

    // Reduce within each warp
    for (int off = warpSize / 2; off > 0; off >>= 1){
        % for i in range(1, nv):
            acc[${i-1}] += __shfl_down_sync(0xFFFFFFFFU, acc[${i-1}], off);
        % endfor
    }
    // Have the first thread in each warp write out to shared memory
    if (tid % warpSize == 0){
        % for i in range(1, nv):
            sdata[${i-1}][tid / warpSize] = acc[${i-1}];
        % endfor
        }

    __syncthreads();

    // Have the first warp perform the final reduction
    if (tid / warpSize == 0)
    {
        % for i in range(1, nv):
            acc[${i-1}] = (tid < blockDim.x / warpSize) ? sdata[${i-1}][tid] : 0;
        % endfor

        for (int off = warpSize / 2; off > 0; off >>= 1){
            % for i in range(1, nv):
                acc[${i-1}] += __shfl_down_sync(0xFFFFFFFFU, acc[${i-1}], off);
            % endfor
        }
        
        if (tid == 0){
            % for i in range(1, nv):
                reduced[ixdtype_t(blockIdx.y)*gridDim.x*${nv-1} + blockIdx.x*${nv-1} + ${i-1}] = acc[${i-1}];
            % endfor
        }
    }
}