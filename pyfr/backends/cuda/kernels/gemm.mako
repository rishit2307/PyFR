<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%pyfr:macro name='gemm' params='B, A, C'>

for (ixdtype_t i=0; i < nb; i+=nby)
    C[(tidrow + i)*nb + tidcol] = 0.0;

// B*A = C
for (ixdtype_t i=0; i < nb; i+=nby){
    idx0 = (tidrow+i)*nb + tidcol;

    for(ixdtype_t j =0; j < nb; j+=nby){
        for(ixdtype_t k=0; k < nby; k++){
            idx1 = (tidrow + j)*nb + k + i;
            C[(tidrow + j)*nb + tidcol] += A[(k+i)*nb + tidcol]*B[idx1];
        }
    }
    __syncthreads();
}

</%pyfr:macro>

<%pyfr:macro name='read' params='r, c, S, mat'>
for (ixdtype_t i=0; i < nb; i+=nby){
    idx = blockIdx.x*ldim + (tidrow + i + r)*nrow + (tidcol + c);

    if (tidcol + c < nrow && tidrow + i + r < nrow)
        S[(tidrow + i)*nb + tidcol] = mat[idx];

}
__syncthreads();
</%pyfr:macro>



<%pyfr:macro name='write' params='r, c, S, mat'>
for (ixdtype_t k=0; k < nb; k+=nby){
    idx = blockIdx.x*ldim + (tidrow + k + r)*nrow + (tidcol + c);
    if (tidcol + c < nrow && tidrow + k + r < nrow)
        mat[idx] = S[(tidrow + k)*nb + tidcol];
}
__syncthreads();
</%pyfr:macro>

