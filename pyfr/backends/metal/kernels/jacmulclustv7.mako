<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.backends.metal.kernels.compute'/>
<%include file='pyfr.backends.metal.kernels.sharedwrite'/>
kernel  void
jacmulclustv7(constant ixdtype_t& nclust, device const fpdtype_t* r1, device fpdtype_t* buff2,
        device const fpdtype_t* jac, device const ixdtype_t*  elemap, 
        device const ixdtype_t* cluster_neles, device const fpdtype_t*  r0,
        uint3 threadIdx [[thread_position_in_threadgroup]],
        uint3 blockIdx [[threadgroup_position_in_grid]])

{

    ixdtype_t idx0, idx1;
    ixdtype_t sidx;
    ixdtype_t upt, vpt;
    ixdtype_t row, col;

    threadgroup fpdtype_t sA[${(bm+2)*bk}];
    threadgroup fpdtype_t sB[${(bn+6)*bk}];

    fpdtype_t regM[${bm*bk//blkx}] = {0.0};
    fpdtype_t regN[${bn//bk}] = {0.0};

    fpdtype_t regAtmp[${bm*bk//blkx}];
    fpdtype_t regBtmp[${bn*bk//blkx}];

    // Accumulation register
    fpdtype_t res[${bm*bk//blkx * bn // bk}] = {0.0};

    ixdtype_t neles = cluster_neles[blockIdx.x];
    ixdtype_t stidx = 0;

    for (ixdtype_t i=0; i < blockIdx.x; i++){
        stidx += cluster_neles[i];
    }

    int tid = threadIdx.x
    ixdtype_t tidrow = tid / ${bk};
    ixdtype_t tidcol = tid % ${bk};
    ixdtype_t tidcolB;

    tidcolB = tidcol*${bn//bk} + 2*(tidcol /${bn//bk});

    for (ixdtype_t k=0; k < ${bm*bn//blkx}; k++){
        res[k] = 0.0;
    }
    % for l in range(0, bk, blkx//bk):
        % for m in range(0, bn, bk):
            sB[(tidrow + ${l})*${bn + 6} + tidcol + ${m}] = 0.0;
        % endfor
    % endfor
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    idx0 = blockIdx.x*${ldim} + (blockIdx.y*${bm} + tidrow)*${ncol} + tidcol;
    % for l in range(0,  bm, blkx//bk):
        idx1 =idx0 + ${l*ncol};
        if (blockIdx.y*${bm} + tidrow + ${l} < ${ncol} && tidcol < ${ncol}){
            sA[tidcol*${bm+2} + tidrow*${bm*bk//blkx} + ${l*bk//blkx}] = jac[idx1];
        }
    % endfor

    idx0 = (tidcol + stidx + blockIdx.z*${bn}) + (tidrow)*${tot_neles};
    % for l in range(0, bk, blkx//bk):
        % for m in range(0, bn, bk):
            idx1 = idx0 + ${m} + ${l*tot_neles};
            sB[(tidrow + ${l})*${bn + 6} + tidcolB + ${m//bk}] = r0[idx1];
        % endfor
    % endfor
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int k=${bk}; k < ${-(ncol % -bk) + ncol - bk}; k+=${bk}){
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
            idx0 = (tidcol + stidx + blockIdx.z*${bn}) + (tidrow + k)*${tot_neles};
            % for l in range(0, bk, blkx//bk):
                % for m in range(0, bn, bk):
                    idx1 = idx0 + ${l*tot_neles} + ${m};
                    regBtmp[${(l*bk//blkx)*bn//bk} + ${m//bk}] = r0[idx1];
                % endfor
            % endfor

        }
        else{
            idx0 = (tidcol + stidx + blockIdx.z*${bn}) + (tidrow + k)*${tot_neles};
            % for l in range(0, bk, blkx//bk):
                % for m in range(0, bn, bk):
                    idx1 = idx0 + ${l*tot_neles} + ${m};
                    if (tidcol + blockIdx.z*${bn} + ${m} < neles){
                        regBtmp[${(l*bk//blkx)*bn//bk} + ${m//bk}] = r0[idx1];
                    } 
                % endfor
            % endfor
        }

        ${pyfr.expand('compute', '')}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        ${pyfr.expand('sharedwrite','')}

        threadgroup_barrier(mem_flags::mem_threadgroup);
    
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


        idx0 = (tidcol + stidx + blockIdx.z*${bn}) + (tidrow + k)*${tot_neles};
        % for l in range(0, bk, blkx//bk):
            % for m in range(0, bn, bk):
                idx1 = idx0 + ${l*tot_neles} + ${m};
                if (tidrow  + k + ${l} < ${ncol} && tidcol + blockIdx.z*${bn} + ${m} < neles){
                    regBtmp[${(l*bk//blkx)*bn//bk} + ${m//bk}] = r0[idx1];
                }
            % endfor
        % endfor

        ${pyfr.expand('compute','')}

        threadgroup_barrier(mem_flags::mem_threadgroup);
        ${pyfr.expand('sharedwrite','')}

        threadgroup_barrier(mem_flags::mem_threadgroup);

    }
    ${pyfr.expand('compute','')}

    % for l in range(0, bm, blkx//bk):
        % for m in range(0, bn, bk):
            idx1 = stidx + blockIdx.z*${bn} + tidcol + ${m} + (tidrow + ${l} + blockIdx.y*${bm})*${tot_neles};

            if (tidcol + ${m} + blockIdx.z*${bn} < neles && tidrow + ${l} + blockIdx.y*${bm} < ${ncol}){
                buff2[idx1] = res[${l*bn//blkx + m//bk}];
            }
        % endfor
    % endfor
}