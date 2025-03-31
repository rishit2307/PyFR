<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%pyfr:macro name='utrtri' params='U, Uinv, ns'>

for (ixdtype_t i=zero; i < nb ; i+=nby)
    Uinv[(tidrow + i)*nb + tidcol] = fzero;

__syncthreads();

if (tid < ns)
    Uinv[tid*nb + tid] = fone/U[tid*nb + tid];

__syncthreads();

if (tid < ns){
    for (ixdtype_t i=ns-two; i >=zero ; i--){
        for (ixdtype_t j=i+one; j <= tid; j++){
            Uinv[i*nb + tid] += U[i*nb + j]*Uinv[j*nb + tid];
        }
    if (tid >i)
        Uinv[i*nb + tid] /= -U[i*nb + i];
    }
}
__syncthreads();
</%pyfr:macro>