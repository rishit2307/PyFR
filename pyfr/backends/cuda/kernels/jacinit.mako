<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

__global__ void
jacinit(ixdtype_t nrow, ixdtype_t ncolb, 
        ixdtype_t ldim, ixdtype_t ldimj, 
        fpdtype_t* __restrict__ r0, fpdtype_t* __restrict__ r1, 
        fpdtype_t* __restrict__ jac, ixdtype_t* __restrict__ eid,
        ixdtype_t npt, ixdtype_t vi, ixdtype_t col, 
        fpdtype_t dtfac, fpdtype_t eps)

{
    ixdtype_t j = ixdtype_t(blockIdx.x)*blockDim.x + threadIdx.x;
    ixdtype_t idx, idxj, idx1;

    ixdtype_t cidj = npt*${ncola} + vi;
    ixdtype_t vid = blockIdx.y % ${ncola};
    ixdtype_t uid = blockIdx.y / ${ncola};

    if (blockIdx.y != cidj)
        dtfac = 0.0f;

    if (j < ${ecount}){
        idx = uid*ldim + SOA_IX(eid[j], vid, ${ncola});
        idx1 = npt*ldim + SOA_IX(eid[j], vi, ${ncola});
        idxj = j*ldimj + cidj + gridDim.y*blockIdx.y;
        jac[idxj] = r0[idx]/(sqrt(1 + fabs(r1[idx1]))*eps) + dtfac;
    }
}