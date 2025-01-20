<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

__global__ __launch_bounds__(${block[0]*block[1]}) void
addidx(ixdtype_t ncolb, ixdtype_t ldim,
       ixdtype_t npt, ixdtype_t vi, ixdtype_t* __restrict__ eid,
       fpdtype_t* __restrict__ x0)
{
    ixdtype_t j = ixdtype_t(blockIdx.x)*blockDim.x + threadIdx.x;
    ixdtype_t idx;

   if (j < ncolb)
    {
        idx = npt*ldim + SOA_IX(j, vi, ${ncola});
        x0[idx] += 1e-8*eid[j];
    }
}
