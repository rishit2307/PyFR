<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

__global__ void
jacmul(ixdtype_t nrow, ixdtype_t ncolb, ixdtype_t ldim, ixdtype_t ncola
          fpdtype_t *__restrict__ jac, fpdtype_t *__restrict__ rm)

ixdtype_t j = (threadIdx.x + blockIdx.x*blockDim.x) + blockIdx.y*(blockDim.x*gridDim.x);
ixdtype_t ridx, jidx;
idxtype_t sidx = blockIdx.y * ncola * ncolb * nrow;

ixdtype_t uid = blockIdx.y / ncola;
ixdtype_t vid = blockIdx.y % ncola;

int eidx = j % ncolb;
fpdtype_t r, acc = 0;


if (eidx < ncolb){
    for (ixdtype_t i = 0; i < ncola; ++i){
        for (ixdtype_t k=0; k < nrow; ++k){
            ridx = k*ldim + SOA_IX(eidx, i, ncola);
            jidx = ridx + sidx;
            acc += rm[ridx] * jac[jidx];
        }
    }
    ridx = uid*ldim + SOA_IX(eidx, vid, ncola);
    rm[ridx] = acc;
}
