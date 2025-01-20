<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

__global__ void
jacinit(ixdtype_t nrow, ixdtype_t ncolb, ixdtype_t ldim, ixdtype_t ldimj,
            fpdtype_t* __restrict__ r0, fpdtype_t* __restrict__ jac, 
            ixdtype_t npt, ixdtype_t vi, fpdtype_t dtfac)

{
    ixdtype_t j = ixdtype_t(blockIdx.x)*blockDim.x + threadIdx.x;
    ixdtype_t idx, idxj;

    ixdtype_t cidj = vi*nrow + npt;
    ixdtype_t uid = blockIdx.y % nrow;
    ixdtype_t vid = blockIdx.y / nrow;

    if (blockIdx.y != cidj)
        dtfac = 0.0f;

    if (j < ncolb){
        idx = uid*ldim + SOA_IX(j, vid, ${ncola});
        idxj = j*ldimj + cidj + gridDim.y*blockIdx.y;
        jac[idxj] = r0[idx] + dtfac;
       
    }
}