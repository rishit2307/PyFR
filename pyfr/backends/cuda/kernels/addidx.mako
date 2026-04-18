<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

__global__ void
addidx(ixdtype_t ncolb, ixdtype_t ldim,
       fpdtype_t* __restrict__ x0, 
       fpdtype_t* __restrict__ x1, 
       ixdtype_t* __restrict__ eid,
       ixdtype_t npt, ixdtype_t vi, ixdtype_t col,
       ixdtype_t stidx, ixdtype_t bsize, ixdtype_t neles,
       fpdtype_t h)
{
    ixdtype_t j = ixdtype_t(blockIdx.x)*blockDim.x + threadIdx.x;
    ixdtype_t idx, idx1;

   if (j < bsize && j + stidx < neles){
        if (eid[j+stidx] == col){
            idx1 = npt*ldim + SOA_IX(j+stidx, vi, ${ncola});
            x1[idx1] = x0[idx1] + sqrt(1 + fabs(x0[idx1]))*h;
        }
    }
}
