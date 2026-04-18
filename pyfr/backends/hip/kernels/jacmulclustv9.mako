<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.backends.hip.kernels.compute'/>
<%include file='pyfr.backends.hip.kernels.sharedwrite'/>

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
jacmulclustv9(fpdtype_t* __restrict__ r0, fpdtype_t *__restrict__ r1,
        jac_fpdtype_t *__restrict__ jac, ixdtype_t *__restrict__ elemap, 
        ixdtype_t *__restrict__ cluster_neles, ixdtype_t *__restrict__ stidxb)

{   const ixdtype_t neles = cluster_neles[blockIdx.x];
    if (blockIdx.z <= neles / ${bn}){
        constexpr ixdtype_t nby = ${blkx}/${bk};

        ixdtype_t idx0, idx1;
        ixdtype_t sidx;
        ixdtype_t upt, vpt;
        ixdtype_t upt2, vpt2;
        ixdtype_t row, col;

        __shared__ jac_fpdtype_t sA[${(bm+2)*bk}];
        __shared__ fpdtype_t sB[${(bn+2)*bk}];

        jac_fpdtype_t regM[${bm*K//blkx}] = {0.0};
        fpdtype_t regN[${bn//K}] = {0.0};
        fpdtype_t regMtmp[${bm*K//blkx}] = {0.0};

        jac_fpdtype_t regAtmp[${bm*bk//blkx}];
        fpdtype_t regBtmp[${bn*bk//blkx}];

        // Accumulation register
        fpdtype_t res[${bm*K//blkx * bn//K}] = {0.0};

        ixdtype_t stidx = 0;

        stidx = stidxb[blockIdx.x];

        ixdtype_t tidrowA = threadIdx.x / ${bk};
        ixdtype_t tidcolA = threadIdx.x % ${bk};

        ixdtype_t tidrow = threadIdx.x / ${K};
        ixdtype_t tidcol = threadIdx.x % ${K};

        ixdtype_t tidcolB = threadIdx.x % ${bn};
        ixdtype_t tidrowB = threadIdx.x / ${bn};

        ixdtype_t tidrowsubB = tidcolB / ${K};
        ixdtype_t tidcolsubB = tidcolB % ${K};

        ## ixdtype_t regemap = elemap[stidx + blockIdx.z*${bn} + tidcolB];
        ## ixdtype_t regemap2[${bn//K}];

        ## % for i in range(0, bn, K):
            ## assert(stidx + blockIdx.z*${bn} + tidcol + ${i} < 20594 && "ERROR l66");
            ## assert(${i//K} < ${bn//K} && "ERROR l67");
            ## regemap2[${i//K}] = elemap[stidx + blockIdx.z*${bn} + tidcol + ${i}];
        ## % endfor

        for (ixdtype_t k=0; k < ${bm*bn//blkx}; k++){
            res[k] = 0.0;
        }

        % for l in range(0, bk, blkx//bn):
            ## assert((tidrowB + ${l})*${bn + 2} + tidcolB < ${bn+2}*${bk} && "ERROR L74");
            sB[(tidrowB + ${l})*${bn + 2} + tidcolB] = 0.0;
        % endfor
        __syncthreads();

        idx0 = blockIdx.x*${ldim} + (blockIdx.y*${bm} + tidrowA)*${ncol} + tidcolA;
        sidx = tidcolA*${bm+2} + tidrowA *${bm*K//blkx};
        % for l in range(0,  bm, blkx//bk):
            idx1 =idx0 + ${l*ncol};
            if (blockIdx.y*${bm} + tidrowA + ${l} < ${ncol} && tidcolA < ${ncol}){
                ## assert(idx1 < ${ldim}*gridDim.x && "ERROR L83");
                ## assert(sidx+ ${l%(blkx//K)*(bm*K//blkx)} + ${l*K//blkx} < ${(bm+2)*bk} && "ERROR L84");
                sA[sidx + ${l%(blkx//K)*(bm*K//blkx)} + ${l*K//blkx}] = jac[idx1];
            }
        % endfor

        idx0 = stidx + blockIdx.z*${bn} + tidcolB;
        sidx = tidcolsubB*${bn//K} + tidrowsubB + tidrowB*${bn+2};
        % for l in range(0, bk, blkx//bn):
            upt = (tidrowB + ${l}) / ${ncola};
            vpt = (tidrowB + ${l}) % ${ncola};
            idx1 = upt*${ldim2} + SOA_IX(elemap[stidx + blockIdx.z*${bn} + tidcolB], vpt , ${ncola});
            if (tidcolB + blockIdx.z*${bn} < neles && tidrowB + ${l} < ${ncol}){
                sB[${l*(bn+2)} + sidx] = r0[idx1];
            }
        % endfor
        __syncthreads();

        for (int k=${bk}; k < ${-(ncol % -bk) + ncol - bk}; k+=${bk}){
            % for l in range(bn*bk//blkx):
                regBtmp[${l}] = 0.0;
            % endfor

            if (blockIdx.y < ${ncol // bm}){
                idx0 = blockIdx.x*${ldim} + (blockIdx.y*${bm} + tidrowA)*${ncol} + tidcolA + k;
                % for l in range(0, bm, blkx//bk):
                    idx1 = idx0 + ${l*ncol};
                    ## assert(idx1 < ${ldim}*gridDim.x && "ERRIR L 108");
                    ## assert(${l*bk//blkx} < 40 && "ERROR L109");
                    regAtmp[${l*bk//blkx}] = jac[idx1];
                % endfor
            }
            else{
                idx0 = blockIdx.x*${ldim} + (blockIdx.y*${bm} + tidrowA)*${ncol} + tidcolA + k;
                % for l in range(0, bm, blkx//bk):
                    idx1 = idx0 + ${l*ncol};
                    if (blockIdx.y*${bm} + tidrowA + ${l} < ${ncol}){
                        ## assert(idx1 < ${ldim}*gridDim.x && "ERRIR L 116");
                        ## assert(${l*bk//blkx} < 40 && "ERROR L117");
                        regAtmp[${l*bk//blkx}] = jac[idx1];
                    }
                % endfor
            }

            if (blockIdx.z < neles / ${bn}){

                % for l in range(0, bk, blkx//bn):
                    upt = (tidrowB  + ${l} + k) / ${ncola};
                    vpt = (tidrowB + ${l} + k) - upt*${ncola};
                    idx1 = upt*${ldim2} + SOA_IX(elemap[stidx + blockIdx.z*${bn} + tidcolB], vpt, ${ncola});
                    ## assert(idx1 < ${ldim2}*320 && "Error L127");
                    ## assert(${l*bn//blkx} < 1 && "ERROR L128");
                    regBtmp[${l*bn//blkx}] = r0[idx1];
                % endfor
            }
            else {
                % for l in range(0, bk, blkx//bn):
                    upt = (tidrowB  + ${l} + k) / ${ncola};
                    vpt = (tidrowB + ${l} + k) - upt*${ncola};
                    idx1 = upt*${ldim2} + SOA_IX(elemap[stidx + blockIdx.z*${bn} + tidcolB], vpt, ${ncola});
                    if (tidcolB + blockIdx.z*${bn} < neles){
                        ## assert(idx1 < ${ldim2}*320 && "Error L136");
                        ## assert(${l*bn//blkx} < 1 && "ERROR L137");
                        regBtmp[${l*bn//blkx}] = r0[idx1];
                    }
                % endfor
            }

            ${pyfr.expand('compute', '')}
            __syncthreads();

            ${pyfr.expand('sharedwrite','')}
            __syncthreads();
        }

        for (int k=${-(ncol % -bk) + ncol - bk}; k < ${ncol}; k+=${bk}){
            % for l in range(0, bn*bk//blkx):
                ## assert(${l} < 1 && "ERROR L 167");
                regBtmp[${l}] = 0.0;
            % endfor

            idx0 = blockIdx.x*${ldim} + (blockIdx.y*${bm} + tidrowA)*${ncol} + tidcolA + k;
            % for l in range(0, bm, blkx//bk):
                idx1 = idx0 + ${l*ncol};
                if (blockIdx.y*${bm} + tidrowA + ${l} < ${ncol} && tidcolA + k < ${ncol}){
                    ## assert(${l*bk//blkx} < 40 && "ERROR L 157");
                    ## assert(idx1 < ${ldim}*gridDim.x && "ERRIR L 158");
                    regAtmp[${l*bk//blkx}] = jac[idx1];
                }
            % endfor

            idx0 = stidx + blockIdx.z*${bn} + tidcolB;
            % for l in range(0, bk, blkx//bn):
                upt = (tidrowB  + ${l} + k) / ${ncola};
                vpt = (tidrowB  + ${l} + k) % ${ncola};
                idx1 =  upt*${ldim2} + SOA_IX(elemap[stidx + blockIdx.z*${bn} + tidcolB], vpt, ${ncola});
                if (tidrowB + k + ${l}< ${ncol} && tidcolB + blockIdx.z*${bn}< neles){
                    ## assert(${l*bn//blkx} < 1 && "ERROR L 167");
                    ## assert(idx1 < ${ldim2}*320 && "Error L168");
                    regBtmp[${l*bn//blkx}] = r0[idx1];
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

        % for l in range(0, bm, blkx//K):
            upt = (upt2 + ${l}) / ${ncola};
            vpt = upt2 + ${l} - upt*${ncola};
            % for m in range(0, bn, K):
                idx1 = upt*${ldim2} + SOA_IX(elemap[idx0 + ${m//K}], vpt, ${ncola});

                if (tidcol + ${m} + blockIdx.z*${bn} < neles && tidrow + ${l} + blockIdx.y*${bm} < ${ncol}){
                    ## assert(idx1 < ${ldim2}*320 && "Error L193");
                    ## assert(${l*bn//blkx + m//K} < 10 && "ERROR L 194");
                    r1[idx1] = res[${l*bn//blkx + m//K}];
                }
            % endfor
        % endfor
    }
}