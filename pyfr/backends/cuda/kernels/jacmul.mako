<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

__global__ void
jacmul(ixdtype_t nrow, ixdtype_t ncolb, ixdtype_t ldim, 
        fpdtype_t *__restrict__ r0, fpdtype_t *__restrict__ r1,
        fpdtype_t *__restrict__ jac)

{
    ixdtype_t tid = threadIdx.x + blockIdx.x*blockDim.x;
    ixdtype_t jidx, r0idx, r1idx;
    int nx = ldim * nrow;

    ixdtype_t uid = blockIdx.y / ${ncola};
    ixdtype_t vid = blockIdx.y % ${ncola};

    if (tid < ncolb){
        r1idx = uid*ldim + SOA_IX(tid, vid, ${ncola});
        r1[r1idx] = 0.0;
        for (ixdtype_t i = 0; i < ${ncola}; ++i){
            for (ixdtype_t k=0; k < nrow; ++k){
                r0idx = k*ldim + SOA_IX(tid, i, ${ncola});
                jidx = r0idx + blockIdx.y * nx;
                r1[r1idx] += r0[r0idx] * jac[jidx];
            }
        }
    }
}