<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

typedef ${pyfr.npdtype_to_ctype(jac_fpdtype)} jac_fpdtype_t;

%if pyfr.npdtype_to_ctype(jac_fpdtype) == '__half':
    #define JAC_HALF_PREC
%endif

#ifdef JAC_HALF_PREC
    #define cast(x) __half2float(x)
#else
    #define cast(x) x
#endif

__global__  __launch_bounds__(${blkx}) void
jacmulclustv9(const fpdtype_t* __restrict__ r0, fpdtype_t *__restrict__ r1,
            const jac_fpdtype_t *__restrict__ jac,
            const ixdtype_t *__restrict__ elemap, 
            const ixdtype_t *__restrict__ cluster_neles, 
            const ixdtype_t *__restrict__ cluster_neles_pad,
            const ixdtype_t *__restrict__ stidx,
            const ixdtype_t *__restrict__ stidx_pad)

{   
    int blockz = blockIdx.z + ${compclust};
    const ixdtype_t neles = cluster_neles[blockz];

    ixdtype_t idx0, idx1;
    int sidx;

    int rowA, colA;
    int row, col, batch;
    int upt, upt2, vpt, vpt2;

    __shared__ jac_fpdtype_t sA[${(bm+2)*bk}];
    __shared__ fpdtype_t sB[${(K*(bn//K + 2))*bk}];

    jac_fpdtype_t regM[${bm*K//blkx}] = {0.0};
    fpdtype_t regN[${bn//K}] = {0.0};

    jac_fpdtype_t regAtmp[${bm*bk//blkx}]= {0.0};
    fpdtype_t regBtmp[${bn*bk//blkx}]= {0.0};

    // Accumulation register
    fpdtype_t res[${bm*K//blkx * bn//K}] = {0.0};

    const ixdtype_t stix = stidx[blockz];
    const ixdtype_t stix_pad = stidx_pad[blockz];

    int curr=0;
    int prev = 0;

    ixdtype_t tidrowA = threadIdx.x / ${bk};
    ixdtype_t tidcolA = threadIdx.x % ${bk};

    ixdtype_t tidrow = threadIdx.x / ${K};
    ixdtype_t tidcol = threadIdx.x % ${K};

    rowA = (blockIdx.x*${bm} + tidrowA);
    colA = tidcolA;
    batch = blockz*${ldim};

    sidx = tidcolA*${bm+2} + tidrowA *${bm*K//blkx};
    idx0 = batch + rowA*${ncol} + colA;

    for (int l=0; l < ${bm}; l+= ${blkx//bk}){
        idx1 = idx0 + l*${ncol};
        if (rowA + l < ${ncol} && colA < ${ncol})
            sA[sidx + (l % ${blkx//K}) * ${bm*K//blkx} + (l*${K})/${blkx}] = jac[idx1];
    }

    sidx = tidcol*${bn//K+2} + tidrow*${K*(bn//K + 2)};
    for (int l=0; l <${bn}; l+=${K}){
        for (int m=0; m < ${bk}; m+=${blkx//K}){
            idx1 =  stix_pad + blockIdx.y*${bn} + tidcol + l + (tidrow + m)*${ncolb};
            if (tidrow + m < ${ncol} && tidcol + l + blockIdx.y*${bn} < ${eleclust}){
                sB[(tidcol)*${bn//K+2} + (tidrow+m)*${K*(bn//K+ 2)} + l / ${K}] = r0[idx1];
            }
        }
    }

    __syncthreads();

    if (blockIdx.x < ${ncol // bm}){

        for (int k=${bk}; k < ${-(ncol % -bk) + ncol - bk}; k+=${bk}){

            for (int l=0; l < ${bn*bk//blkx}; l++){
                regBtmp[l] = 0.0;
            }

            idx0 = batch + rowA*${ncol} + colA + k;
            for (int l=0; l < ${bm}; l+=${blkx//bk}){
                idx1 = idx0 + l*${ncol};
                regAtmp[(l*${bk})/ ${blkx}] = jac[idx1];
            }

            for (int l=0; l < ${bn}; l+=${K}){
                for (int m=0; m < ${bk}; m+=${blkx//K}){
                    idx1 = stix_pad + blockIdx.y*${bn} + tidcol + l + (tidrow + m + k)*${ncolb};
                    regBtmp[(l/${K})*${bk*K//blkx} + (m*${K})/${blkx}] = r0[idx1];
                }
            }

            for (int l=0; l < ${bk}; l++){
                for (int m=0; m < ${bm*K//blkx}; m++){
                    regM[m] = sA[l*${(bm+2)} + tidrow*${bm*K//blkx} + m];
                }

                for (int n=0; n < ${bn//K}; n++){
                    regN[n] = sB[l*${K*(bn//K + 2)} + tidcol*${bn//K + 2} + n];
                }

                % for m in range(bm*bk//blkx):
                    % for n in range(bn//K):
                        res[${m*bn//K + n}] += fpdtype_t(cast(regM[${m}]))*regN[${n}];
                    % endfor
                % endfor
            }
            __syncthreads();
            curr ^=1;

            sidx = tidcolA*${bm+2} + tidrowA*${bm*K//blkx};
            for (int l=0; l < ${bm}; l+=${blkx//bk}){
                sA[sidx + (l % ${blkx//K}) * ${bm*K//blkx} + (l*${K})/${blkx}] = regAtmp[(l*${bk})/${blkx}];
            }
            sidx = tidcol*${bn//K+2} + tidrow*${K*(bn//K + 2)};
            for (int l=0; l < ${bn}; l+=${K}){
                for (int m=0; m < ${bk}; m+=${blkx//K}){
                    if (tidrow + m < ${ncol} && tidcol + l + blockIdx.y*${bn} < ${eleclust}){
                        sB[(tidcol)*${bn//K+2} + (tidrow+m)*${K*(bn//K+ 2)} + l / ${K}] = regBtmp[(l/${K})*${bk*K//blkx} + (m*${K})/${blkx}];
                    }
                }
            }
            __syncthreads();
            prev ^=1;
        }
    }

    else{

        for (int k=${bk}; k < ${-(ncol % -bk) + ncol - bk}; k+=${bk}){
            for (int l=0; l < ${bn*bk//blkx}; l++){
                regBtmp[l] = 0.0;
            }

            idx0 = batch + rowA*${ncol} + colA + k;
            for (int l=0; l < ${bm}; l+=${blkx//bk}){
                idx1 = idx0 + l*${ncol};
                if (rowA + l < ${ncol})
                    regAtmp[(l*${bk})/ ${blkx}] = jac[idx1];
            }

            for (int l=0; l < ${bn}; l+=${K}){
                for (int m=0; m < ${bk}; m+=${blkx//K}){
                    idx1 = stix_pad + blockIdx.y*${bn} + tidcol + l + (tidrow + m + k)*${ncolb};
                    if (tidcol + l + blockIdx.y*${bn} < ${eleclust})
                        regBtmp[(l/${K})*${bk*K//blkx} + (m*${K})/${blkx}] = r0[idx1];
                }
            }


            for (int l=0; l < ${bk}; l++){
                for (int m=0; m < ${bm*K//blkx}; m++){
                    regM[m] = sA[l*${(bm+2)} + tidrow*${bm*K//blkx} + m];
                }

                for (int n=0; n < ${bn//K}; n++){
                    regN[n] = sB[l*${K*(bn//K + 2)} + tidcol*${bn//K + 2} + n];
                }

                % for m in range(bm*bk//blkx):
                    % for n in range(bn//K):
                        res[${m*bn//K + n}] += fpdtype_t(cast(regM[${m}]))*regN[${n}];
                    % endfor
                % endfor
            }
            __syncthreads();
            curr ^=1;

            sidx = tidcolA*${bm+2} + tidrowA*${bm*K//blkx};
            for (int l=0; l < ${bm}; l+=${blkx//bk}){
                sA[sidx + (l % ${blkx//K}) * ${bm*K//blkx} + (l*${K})/${blkx}] = regAtmp[(l*${bk})/${blkx}];
            }
            sidx = tidcol*${bn//K+2} + tidrow*${K*(bn//K + 2)};
            for (int l=0; l < ${bn}; l+=${K}){
                for (int m=0; m < ${bk}; m+=${blkx//K}){
                    if (tidrow + m < ${ncol} && tidcol + l + blockIdx.y*${bn} < ${eleclust}){
                        sB[(tidcol)*${bn//K+2} + (tidrow+m)*${K*(bn//K+ 2)} + l / ${K}] = regBtmp[(l/${K})*${bk*K//blkx} + (m*${K})/${blkx}];
                    }
                }
            }
            __syncthreads();
            prev^=1;
        }

    }

    for (int k=${-(ncol % -bk) + ncol - bk}; k < ${ncol}; k+=${bk}){
        for (int l=0; l < ${bn*bk//blkx}; l++){
            regBtmp[l] = 0.0;
        }

        // Async loads
        idx0 = batch + rowA*${ncol} + colA + k;
        for (int l=0; l < ${bm}; l+=${blkx//bk}){
            idx1 = idx0 + l*${ncol};
            if (rowA + l < ${ncol} && colA + k < ${ncol})
                regAtmp[(l*${bk})/ ${blkx}] = jac[idx1];
        }

        for (int l=0; l < ${bn}; l+=${K}){
            for (int m=0; m < ${bk}; m+=${blkx//K}){
                idx1 =  stix_pad + blockIdx.y*${bn} + tidcol + l + (tidrow + m + k)*${ncolb};
                if (tidcol + l + blockIdx.y*${bn} < ${eleclust} && tidrow + m + k < ${ncol})
                    regBtmp[(l/${K})*${bk*K//blkx} + (m*${K})/${blkx}] = r0[idx1];

            }
        }

        // Compute
        for (int l=0; l < ${bk}; l++){
            for (int m=0; m < ${bm*K//blkx}; m++){
                regM[m] = sA[l*${(bm+2)} + tidrow*${bm*K//blkx} + m];
            }

            for (int n=0; n < ${bn//K}; n++){
                regN[n] = sB[l*${K*(bn//K + 2)} + tidcol*${bn//K + 2} + n];
            }

            % for m in range(bm*bk//blkx):
                % for n in range(bn//K):
                    res[${m*bn//K + n}] += fpdtype_t(cast(regM[${m}]))*regN[${n}];
                % endfor
            % endfor
        }
        __syncthreads();
        curr ^=1;

        // Shared Write
        sidx = tidcolA*${bm+2} + tidrowA*${bm*K//blkx};
        for (int l=0; l < ${bm}; l+=${blkx//bk}){
            sA[sidx + (l % ${blkx//K}) * ${bm*K//blkx} + (l*${K})/${blkx}] = regAtmp[(l*${bk})/${blkx}];
        }
        sidx = tidcol*${bn//K+2} + tidrow*${K*(bn//K + 2)};
        for (int l=0; l < ${bn}; l+=${K}){
            for (int m=0; m < ${bk}; m+=${blkx//K}){
                if (tidrow + m < ${ncol} && tidcol + l + blockIdx.y*${bn} < ${eleclust}){
                    sB[(tidcol)*${bn//K+2} + (tidrow+m)*${K*(bn//K+ 2)} + l / ${K}] = regBtmp[(l/${K})*${bk*K//blkx} + (m*${K})/${blkx}];
                }
            }
        }
        __syncthreads();
        prev^=1;
    }

    for (int l=0; l < ${ncol - (-(ncol % -bk) + ncol - bk)}; l++){

        for (int m=0; m < ${bm*K//blkx}; m++){
            regM[m] = sA[l*${(bm+2)} + tidrow*${bm*K//blkx} + m];
        }

        for (int n=0; n < ${bn//K}; n++){
            regN[n] = sB[l*${K*(bn//K + 2)} + tidcol*${bn//K + 2} + n];
        }

        for (int m=0; m < ${bm*K//blkx}; m++){
            for (int n=0; n < ${bn//K}; n++){
                res[(m*${bn//K}) + n] += fpdtype_t(cast(regM[m]))*regN[n];
            }
        }
    }

    idx0 = stix + tidcol + blockIdx.y*${bn};
    for (int l=0; l < ${bm}; l+=${blkx//K}){
        for (int m=0; m < ${bn}; m+=${K}){
            upt = (tidrow + l + blockIdx.x*${bm}) / ${ncola};
            vpt = (tidrow + l + blockIdx.x*${bm}) % ${ncola};
            idx1 = upt*${ldim2} + SOA_IX(elemap[idx0 + m], vpt, ${ncola});
            if (tidcol + m + blockIdx.y*${bn} < neles && tidrow + l + blockIdx.x*${bm} < ${ncol}){
                r1[idx1] =  ${'_out[vpt]*' if inscales else ''}res[(l*${bn})/${blkx} + m/${K}];
            }
        }
    }
}
