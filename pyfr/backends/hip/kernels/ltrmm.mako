<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%pyfr:macro name='l1trmm' params='L, P, Q'>

for (ixdtype_t i=zero; i < nb; i+=nby)
    Q[(tidrow + i)*nb + tidcol] = P[(tidrow + i)*nb + tidcol];

// L*P = Q
for (ixdtype_t i=zero; i < nb; i+=nby){
    idx0 = (tidrow+i)*nb + tidcol;

    for (ixdtype_t j=zero; j< tidrow; j++)
        Q[(tidrow + i)*nb + tidcol] += L[(tidrow+i)*nb + i + j]*P[(j+i)*nb + tidcol];

    for (ixdtype_t j=nby+i; j< nb; j += nby){
        for (ixdtype_t k=zero; k < nby; k++){
            idx1 = (tidrow + j)*nb + k + i;
            Q[(tidrow +j)*nb + tidcol] += P[(k+i)*nb + tidcol]*L[idx1];
        }
    }
    __syncthreads();
}

</%pyfr:macro>

<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%pyfr:macro name='l2trmm' params='P, L, Q'>

//P*L = Q
for (ixdtype_t i=zero; i < nb; i+=nby){
     Q[(tidrow + i)*nb + tidcol] = P[(tidrow+i)*nb + tidcol];
}

__syncthreads();


for (ixdtype_t i=zero; i < nb; i+=nby){
    for (ixdtype_t j=zero; j < nb; j+=nby){
        for (ixdtype_t k=zero; k < nby; k++){
            idx1 = (tidrow + j)*nb + k + i;
            if (tidcol < k + i)
                Q[(tidrow + j)*nb + tidcol] += L[(k+i)*nb + tidcol]*P[idx1];

        }
    }
}
__syncthreads();

</%pyfr:macro>


