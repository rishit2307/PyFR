<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

__global__ __launch_bounds__(${blocksz}) void
reduction(ixdtype_t nrow, ixdtype_t ncolb, ixdtype_t ldim,
          fpdtype_t *__restrict__ reduced,
          fpdtype_t *__restrict__ rcurr
% if method == 'errest':
          ,fpdtype_t *__restrict__ rold, fpdtype_t *__restrict__ rerr, 
          fpdtype_t atol, fpdtype_t rtol)
% elif method == 'errest_imp':
          ,fpdtype_t * __restrict__ rerr, fpdtype_t atol, fpdtype_t rtol)
% elif method == 'resid' and dt_type == 'matrix':
          ,fpdtype_t *__restrict__ rold,
          fpdtype_t *__restrict__ dt_mat, fpdtype_t dt_fac)
% elif method == 'resid':
            ,fpdtype_t *__restrict__ rold, fpdtype_t dt_fac)
% elif method == 'norm':
            )
% endif
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
        % if method == 'errest':
            r = rerr[idx]/(atol + rtol*max(fabs(rcurr[idx]), fabs(rold[idx])));
        % elif method == 'errest_imp':
            r = rerr[idx]/(atol + rtol*fabs(rcurr[idx]));
        % elif method == 'resid':
            r = (rcurr[idx] - rold[idx])/(dt_fac${'*dt_mat[idx]' if dt_type == 'matrix' else ''});
        % elif method == 'norm':
            r = rcurr[idx];
        % endif

        % if norm == 'uniform':
            acc = max(r*r, acc);
        % elif norm == 'l1':
            acc += fabs(r);
        % else:
            acc += r*r;
        % endif
        }
    }

    // Reduce within each warp
    for (int off = warpSize / 2; off > 0; off >>= 1)
% if norm == 'uniform':
        acc = max(__shfl_down(acc, off), acc);
% else:
        acc += __shfl_down(acc, off);
% endif

    // Have the first thread in each warp write out to shared memory
    if (tid % warpSize == 0)
        sdata[tid / warpSize] = acc;

    __syncthreads();

    // Have the first warp perform the final reduction
    if (tid / warpSize == 0)
    {
        acc = (tid < blockDim.x / warpSize) ? sdata[tid] : 0;

        for (int off = warpSize / 2; off > 0; off >>= 1)
    % if norm == 'uniform':
            acc = max(__shfl_down(acc, off), acc);
    % else:
            acc += __shfl_down(acc, off);
    % endif

        if (tid == 0)
            reduced[ixdtype_t(blockIdx.y)*gridDim.x + blockIdx.x] = acc;
    }
}
