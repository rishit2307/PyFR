<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%pyfr:macro name='gemm' params='B, A, C'>

for (ixdtype_t i=zero; i < nb; i+=nby)
    C[(tidrow + i)*nb + tidcol] = fzero;

// B*A = C
for (ixdtype_t i=zero; i < nb; i+=nby){
    idx0 = (tidrow+i)*nb + tidcol;

    for(ixdtype_t j =zero; j < nb; j+=nby){
        for(ixdtype_t k=zero; k < nby; k++){
            idx1 = (tidrow + j)*nb + k + i;
            C[(tidrow + j)*nb + tidcol] += A[(k+i)*nb + tidcol]*B[idx1];
        }
    }
    __syncthreads();
}

</%pyfr:macro>

<%pyfr:macro name='read' params='r, c, S, mat'>
for (ixdtype_t i=zero; i < nb; i+=nby){
    idx = blockIdx.x*ldim + (tidrow + i + r)*nrow + (tidcol + c);

    if (tidcol + c < nrow && tidrow + i + r < nrow){
        S[(tidrow + i)*nb + tidcol] = mat[idx];

    }
}
__syncthreads();
</%pyfr:macro>



<%pyfr:macro name='write' params='r, c, S, mat'>
for (ixdtype_t k=zero; k < nb; k+=nby){
    idx = blockIdx.x*ldim + (tidrow + k + r)*nrow + (tidcol + c);
    if (tidcol + c < nrow && tidrow + k + r < nrow){
        mat[idx] = S[(tidrow + k)*nb + tidcol];

    }
}
__syncthreads();
</%pyfr:macro>

<%pyfr:macro name='write_jacinv' params='r, c, S, mat'>
for (ixdtype_t k=zero; k < nb; k+=nby){
    upt = (tidcol + c) / ${ncola};
    vpt = (tidcol + c) % ${ncola};
    idx = (tidrow + k + r) *${ldimj} + upt*${ldim2} + SOA_IX(blockIdx.x + stidx, vpt, ${ncola});
    if (tidcol + c < nrow && tidrow + k + r < nrow){
            mat[idx] = S[(tidrow + k)*nb + tidcol];
    }
        
}
__syncthreads();
</%pyfr:macro>


<%pyfr:macro name='write_jacinv_kmeans' params='r, c, S, mat'>
for (ixdtype_t k=zero; k < nb; k+=nby){
    idx = blockIdx.x*ldim + (tidrow + k + r)*nrow + (tidcol + c);
    if (tidcol + c < nrow && tidrow + k + r < nrow)
        mat[idx] = S[(tidrow + k)*nb + tidcol];
}
__syncthreads();
</%pyfr:macro>



<%pyfr:macro name='read_jacinv' params='r, c, S, mat'>
for (ixdtype_t i=zero; i < nb; i+=nby){
    upt = (tidcol + c) / ${ncola};
    vpt = (tidcol + c) % ${ncola};

    idx = (tidrow + i + r) *${ldimj} + upt*${ldim2} + SOA_IX(blockIdx.x +stidx, vpt, ${ncola});


    if (tidcol + c < nrow && tidrow + i + r < nrow){
            S[(tidrow + i)*nb + tidcol] = mat[idx];
    }
        
}
__syncthreads();
</%pyfr:macro>



<%pyfr:macro name='read_jacinv_kmeans' params='r, c, S, mat'>
for (ixdtype_t i=zero; i < nb; i+=nby){
    idx = blockIdx.x*ldim + (tidrow + i + r)*nrow + (tidcol + c);
   

    if (tidcol + c < nrow && tidrow + i + r < nrow)
        S[(tidrow + i)*nb + tidcol] = mat[idx];

}
__syncthreads();
</%pyfr:macro>