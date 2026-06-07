<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

__global__ void
shuffle(const fpdtype_t* __restrict__ xin, 
       fpdtype_t* __restrict__ xout,
       const ixdtype_t* __restrict__ emap,
       const ixdtype_t* __restrict__ stidx,
       const ixdtype_t* __restrict__ stidx_pad,
       const ixdtype_t* __restrict__ cluster_id)
{
    ixdtype_t i = blockIdx.x*blockDim.x + threadIdx.x;
    int idx, idx1;

    if ( i < ${ncolb}){
        int ci = cluster_id[i];
        idx1 = (blockIdx.y/${ncola})*${ldim} + SOA_IX(emap[i], blockIdx.y % ${ncola}, ${ncola});
        idx = blockIdx.y*${ldimout} + stidx_pad[ci] + (i - stidx[ci]);
        xout[idx] = __ldg(&xin[idx1]);
    }
}
