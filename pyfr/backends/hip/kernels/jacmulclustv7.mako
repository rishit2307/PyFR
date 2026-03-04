<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.backends.hip.kernels.compute'/>
<%include file='pyfr.backends.hip.kernels.sharedwrite'/>
__global__ __launch_bounds__(${blkx}) void
jacmulclustv7(fpdtype_t *__restrict__ r1,
        fpdtype_t *__restrict__ jac, ixdtype_t *__restrict__ elemap, 
        ixdtype_t *__restrict__ cluster_neles, fpdtype_t* __restrict__ r0)

{
    constexpr ixdtype_t nby = ${blkx}/${bk};

    ixdtype_t idx0, idx1;
    ixdtype_t sidx;
    ixdtype_t upt, vpt;
    ixdtype_t upt2, vpt2;
    ixdtype_t row, col;

    __shared__ fpdtype_t sA[${(bm+2)*bk}];
    __shared__ fpdtype_t sB[${(bn+2)*bk}];

    fpdtype_t regM[${bm*bk//blkx}] = {0.0};
    fpdtype_t regN[${bn//bk}] = {0.0};

    fpdtype_t regAtmp[${bm*bk//blkx}];
    fpdtype_t regBtmp[${bn*bk//blkx}];

    // Accumulation register
    fpdtype_t res[${bm*bk//blkx * bn//bk}] = {0.0};

    ixdtype_t neles = cluster_neles[blockIdx.x];
    ixdtype_t stidx = 0;

    for (ixdtype_t i=0; i < blockIdx.x; i++){
        stidx += cluster_neles[i];
    }

    ixdtype_t tidrow = threadIdx.x / ${bk};
    ixdtype_t tidcol = threadIdx.x % ${bk};

    ixdtype_t tidcolB = threadIdx.x % ${bn};
    ixdtype_t tidrowB = threadIdx.x / ${bn};

    ixdtype_t tidrowsubB = tidcolB / ${bk};
    ixdtype_t tidcolsubB = tidcolB % ${bk};

    for (ixdtype_t k=0; k < ${bm*bn//blkx}; k++){
        res[k] = 0.0;
    }

    ixdtype_t regemap = elemap[stidx + blockIdx.z*${bn} + tidcolB];
    regemap = SOA_IX2(regemap, ${ncola});

    % for l in range(0, bk, blkx//bn):
        sB[(tidrowB + ${l})*${bn + 2} + tidcolB] = 0.0;
    % endfor
    __syncthreads();

    idx0 = blockIdx.x*${ldim} + (blockIdx.y*${bm} + tidrow)*${ncol} + tidcol;
    sidx = tidcol*${bm+2} + tidrow*${bm*bk//blkx};
    % for l in range(0,  bm, blkx//bk):
        idx1 =idx0 + ${l*ncol};
        if (blockIdx.y*${bm} + tidrow + ${l} < ${ncol} && tidcol < ${ncol}){
            sA[sidx + ${l*bk//blkx}] = jac[idx1];
        }
    % endfor

    idx0 = stidx + blockIdx.z*${bn} + tidcolB;
    sidx = tidcolsubB*${bn//bk} + tidrowsubB;
    % for l in range(0, bk, blkx//bn):
        upt = (tidrowB + ${l}) / ${ncola};
        vpt = (tidrowB + ${l}) % ${ncola};

        idx1 = upt*${ldim2} + SOA_IX(elemap[idx0], vpt , ${ncola});
        if (tidcolB + blockIdx.z*${bn} < neles && tidrowB + ${l} < ${ncol}){
            sB[(tidrowB + ${l})*${bn+2} + sidx] = r0[idx1];
        }
    % endfor
    __syncthreads();

    for (ixdtype_t k=${bk}; k < ${-(ncol % -bk) + ncol - bk}; k+=${bk}){
    ## % for k in range(bk, -(ncol % -bk) + ncol - bk, bk):
        % for l in range(bn*bk//blkx):
            regBtmp[${l}] = 0.0;
        % endfor


        if (blockIdx.y < ${ncol // bm}){
            idx0 = blockIdx.x*${ldim} + (blockIdx.y*${bm} + tidrow)*${ncol} + tidcol + k;
            % for l in range(0, bm, blkx//bk):
                idx1 = idx0 + ${l*ncol};
                regAtmp[${l*bk//blkx}] = jac[idx1];
            % endfor
        }
        else{
            idx0 = blockIdx.x*${ldim} + (blockIdx.y*${bm} + tidrow)*${ncol} + tidcol + k;
            % for l in range(0, bm, blkx//bk):
                idx1 = idx0 + ${l*ncol};
                if (blockIdx.y*${bm} + tidrow + ${l} < ${ncol}){
                    regAtmp[${l*bk//blkx}] = jac[idx1];
                }
            % endfor
        }
        if (blockIdx.z < neles / ${bn}){

            % for l in range(0, bk, blkx//bn):
                upt = (tidrowB  + ${l} + k) / ${ncola};
                vpt = (tidrowB + ${l} + k) - upt*${ncola};
                idx1 = upt*${ldim2} + regemap + vpt*SOA_SZ;
                regBtmp[${l*bn//blkx}] = r0[idx1];
                ## regBtmp[${l*bn//blkx}] = r0[rixB[(k/${bk})*${bn*bk//blkx} + ${l*bn//blkx}]];
            % endfor
        }
        else{
            % for l in range(0, bk, blkx//bn):
                upt = (tidrowB  + ${l} +k) / ${ncola};
                vpt = (tidrowB + ${l} + k) - upt*${ncola};
                idx1 = upt*${ldim2} + regemap + vpt*SOA_SZ;
                if (tidcolB + blockIdx.z*${bn} < neles){
                    regBtmp[${l*bn//blkx}] = r0[idx1];
                    ## regBtmp[${l*bn//blkx}] = r0[rixB[(k/${bk})*${bn*bk//blkx} + ${l*bn//blkx}]];
                }
            % endfor
        }

        ${pyfr.expand('compute', '')}
        __syncthreads();

        ${pyfr.expand('sharedwrite','')}

        __syncthreads();

    ## % endfor
    }
    for (int k=${-(ncol % -bk) + ncol - bk}; k < ${ncol}; k+=${bk}){


        % for l in range(0, bn, blkx//bk):
            regBtmp[${l*bk//blkx}] = 0.0;
        % endfor

        idx0 = blockIdx.x*${ldim} + (blockIdx.y*${bm} + tidrow)*${ncol} + tidcol + k;
        % for l in range(0, bm, blkx//bk):
            idx1 = idx0 + ${l*ncol};
            if (blockIdx.y*${bm} + tidrow + ${l} < ${ncol} && tidcol + k < ${ncol}){
                regAtmp[${l*bk//blkx}] = jac[idx1];
            }
        % endfor

        idx0 = stidx + blockIdx.z*${bn} + tidcolB;
        % for l in range(0, bk, blkx//bn):
            upt = (tidrowB  + ${l} + k) / ${ncola};
            vpt = (tidrowB  + ${l} + k) % ${ncola};
            idx1 =  upt*${ldim2} + SOA_IX(elemap[idx0], vpt, ${ncola});
            if (tidrowB + k + ${l}< ${ncol} && tidcolB + blockIdx.z*${bn}< neles){
                regBtmp[${l*bn//blkx}] = r0[idx1];
                ## regBtmp[${l*bn//blkx}] = r0[rixB[(k/${bk})*${bn*bk//blkx} + ${l*bn//blkx}]];
            }
        % endfor

        ${pyfr.expand('compute','')}

        __syncthreads();

        ${pyfr.expand('sharedwrite','')}

        __syncthreads();

    }
    ${pyfr.expand('compute','')}

    idx0 = stidx + blockIdx.z*${bn} + tidcol;
    upt2 = (tidrow + blockIdx.y*${bm});
    vpt2 = (tidrow + blockIdx.y*${bm});

    % for l in range(0, bm, blkx//bk):
        upt = (upt2 + ${l}) / ${ncola};
        vpt = upt2 + ${l} - upt*${ncola};
        % for m in range(0, bn, bk):
            idx1 = upt*${ldim2} + SOA_IX(elemap[idx0 + ${m}], vpt, ${ncola});
            
            if (tidcol + ${m} + blockIdx.z*${bn} < neles && tidrow + ${l} + blockIdx.y*${bm} < ${ncol}){
                r1[idx1] = res[${l*bn//blkx + m//bk}];
            }
        % endfor
    % endfor
}