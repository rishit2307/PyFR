<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

__global__ void
addidx(ixdtype_t ncolb, ixdtype_t ldim,
       fpdtype_t* __restrict__ x0, 
       fpdtype_t* __restrict__ eid,
       ixdtype_t npt, ixdtype_t vi)
{
    ixdtype_t j = ixdtype_t(blockIdx.x)*blockDim.x + threadIdx.x;
    ixdtype_t idx;

   if (j < ncolb)
    {
        idx = npt*ldim + SOA_IX(j, vi, ${ncola});
        x0[idx] += 1.0E-8f*eid[j];
    }
}
