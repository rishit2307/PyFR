<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

__global__ void
addidx(ixdtype_t ncolb, ixdtype_t ldim,
       fpdtype_t* __restrict__ x0, 
       fpdtype_t* __restrict__ x1, 
       ixdtype_t* __restrict__ eid,
       ixdtype_t npt, ixdtype_t vi)
{
    ixdtype_t j = ixdtype_t(blockIdx.x)*blockDim.x + threadIdx.x;
    int i = blockIdx.y*blockDim.y + threadIdx.y;
    ixdtype_t idx, idx1;

   if (j < ncolb)
    {
        % for k in subdims:
            idx = i*ldim + SOA_IX(j, ${k}, ${ncola});
            x1[idx] = x0[idx];
        % endfor

        idx1 = npt*ldim + SOA_IX(j, vi, ${ncola});
        x1[idx1] = x0[idx1] + 1.0E-8f*eid[j];
    }
}
