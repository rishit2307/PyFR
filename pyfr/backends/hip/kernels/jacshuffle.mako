<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

__global__ __launch_bounds__(${block[0]*block[1]*block[2]}) void
jacshuffle(ixdtype_t ncolb, ixdtype_t ldim0, ixdtype_t ldim1,
            fpdtype_t* __restrict__ jac0, fpdtype_t* __restrict__ jac1
)
{
    ixdtype_t j = ixdtype_t(blockIdx.x)*blockDim.x + threadIdx.x;
    ixdtype_t idxj0, idxj1;
    int nupts = gridDim.y / ${ncola};

    ixdtype_t vid = blockIdx.z % ${ncola};
    ixdtype_t uid = blockIdx.z / ${ncola};
    
    ixdtype_t vidy = blockIdx.y % ${ncola};
    ixdtype_t uidy = blockIdx.y / ${ncola};

    if (j < ncolb){
        idxj0 = blockIdx.z + gridDim.y*blockIdx.y + j*ldim0;
        idxj1 = (uidy*${ncola} + vidy)*(nupts*ldim1) + uid*ldim1 + SOA_IX(j, vid, ${ncola});
        jac1[idxj1] = jac0[idxj0];
    }

}