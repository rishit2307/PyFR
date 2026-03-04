<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.backends.opencl.kernels.compute'/>
<%include file='pyfr.backends.opencl.kernels.sharedwrite'/>
__kernel  void
jacmulclustv7(ixdtype_t nclust, __global fpdtype_t* restrict r1, __global fpdtype_t* restrict buff2,
        __global fpdtype_t* restrict jac, __global ixdtype_t* restrict  elemap, 
        __global ixdtype_t* restrict cluster_neles, __global fpdtype_t* restrict  r0)

{
    const ixdtype_t nby = ${blkx}/${bk};

    ixdtype_t idx0, idx1;
    ixdtype_t sidx;
    ixdtype_t upt, vpt;
    ixdtype_t row, col;

    __local fpdtype_t sA[${(bm+2)*bk}];
    __local fpdtype_t sB[${(bn+6)*bk}];

    fpdtype_t regM[${bm*bk//blkx}] = {0.0};
    fpdtype_t regN[${bn//bk}] = {0.0};

    fpdtype_t regAtmp[${bm*bk//blkx}];
    fpdtype_t regBtmp[${bn*bk//blkx}];

    // Accumulation register
    fpdtype_t res[${bm*bk//blkx * bn // bk}] = {0.0};

    ixdtype_t neles = cluster_neles[get_group_id(0)];
    ixdtype_t stidx = 0;

    for (ixdtype_t i=0; i < get_group_id(0); i++){
        stidx += cluster_neles[i];
    }

    int tid = (int)get_local_id(0);
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
    barrier(CLK_LOCAL_MEM_FENCE);

    int blockidx_x = get_group_id(0);
    int blockidx_y = get_group_id(1);
    int blockidx_z = get_group_id(2);
    
    idx0 = blockidx_x*${ldim} + (blockidx_y*${bm} + tidrow)*${ncol} + tidcol;
    % for l in range(0,  bm, blkx//bk):
        idx1 =idx0 + ${l*ncol};
        if (blockidx_y*${bm} + tidrow + ${l} < ${ncol} && tidcol < ${ncol}){
            sA[tidcol*${bm+2} + tidrow*${bm*bk//blkx} + ${l*bk//blkx}] = jac[idx1];
        }
    % endfor

    idx0 = (tidcol + stidx + blockidx_z*${bn}) + (tidrow)*${tot_neles};
    % for l in range(0, bk, blkx//bk):
        % for m in range(0, bn, bk):
            idx1 = idx0 + ${m} + ${l*tot_neles};
            sB[(tidrow + ${l})*${bn + 6} + tidcolB + ${m//bk}] = r0[idx1];
        % endfor
    % endfor
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k=${bk}; k < ${-(ncol % -bk) + ncol - bk}; k+=${bk}){
        % for l in range(bn*bk//blkx):
            regBtmp[${l}] = 0.0;
        % endfor

        if (blockidx_y < ${ncol // bm}){
            idx0 = blockidx_x*${ldim} + (blockidx_y*${bm} + tidrow)*${ncol} + tidcol + k;
            % for l in range(0, bm, blkx//bk):
                idx1 = idx0 + ${l*ncol};
                regAtmp[${l*bk//blkx}] = jac[idx1];
            % endfor
        }
        else{
            idx0 = blockidx_x*${ldim} + (blockidx_y*${bm} + tidrow)*${ncol} + tidcol + k;
            % for l in range(0, bm, blkx//bk):
                idx1 = idx0 + ${l*ncol};
                if (blockidx_y*${bm} + tidrow + ${l} < ${ncol}){
                    regAtmp[${l*bk//blkx}] = jac[idx1];
                }
            % endfor
        }
        if (blockidx_z < neles / ${bn}){
            idx0 = (tidcol + stidx + blockidx_z*${bn}) + (tidrow + k)*${tot_neles};
            % for l in range(0, bk, blkx//bk):
                % for m in range(0, bn, bk):
                    idx1 = idx0 + ${l*tot_neles} + ${m};
                    regBtmp[${(l*bk//blkx)*bn//bk} + ${m//bk}] = r0[idx1];
                % endfor
            % endfor

        }
        else{
            idx0 = (tidcol + stidx + blockidx_z*${bn}) + (tidrow + k)*${tot_neles};
            % for l in range(0, bk, blkx//bk):
                % for m in range(0, bn, bk):
                    idx1 = idx0 + ${l*tot_neles} + ${m};
                    if (tidcol + blockidx_z*${bn} + ${m} < neles){
                        regBtmp[${(l*bk//blkx)*bn//bk} + ${m//bk}] = r0[idx1];
                    } 
                % endfor
            % endfor
        }

        ${pyfr.expand('compute', '')}
       barrier(CLK_LOCAL_MEM_FENCE);

        ${pyfr.expand('sharedwrite','')}

        barrier(CLK_LOCAL_MEM_FENCE);
    
    }
    for (int k=${-(ncol % -bk) + ncol - bk}; k < ${ncol}; k+=${bk}){


        % for l in range(0, bn, blkx//bk):
            regBtmp[${l*bk//blkx}] = 0.0;
        % endfor

        idx0 = blockidx_x*${ldim} + (blockidx_y*${bm} + tidrow)*${ncol} + tidcol + k;
        % for l in range(0, bm, blkx//bk):
            idx1 = idx0 + ${l*ncol};
            if (blockidx_y*${bm} + tidrow + ${l} < ${ncol} && tidcol + k < ${ncol}){
                regAtmp[${l*bk//blkx}] = jac[idx1];
            }
        % endfor


        idx0 = (tidcol + stidx + blockidx_z*${bn}) + (tidrow + k)*${tot_neles};
        % for l in range(0, bk, blkx//bk):
            % for m in range(0, bn, bk):
                idx1 = idx0 + ${l*tot_neles} + ${m};
                if (tidrow  + k + ${l} < ${ncol} && tidcol + blockidx_z*${bn} + ${m} < neles){
                    regBtmp[${(l*bk//blkx)*bn//bk} + ${m//bk}] = r0[idx1];
                }
            % endfor
        % endfor

        ${pyfr.expand('compute','')}

        barrier(CLK_LOCAL_MEM_FENCE);

        ${pyfr.expand('sharedwrite','')}

        barrier(CLK_LOCAL_MEM_FENCE);

    }
    ${pyfr.expand('compute','')}

    % for l in range(0, bm, blkx//bk):
        % for m in range(0, bn, bk):
            idx1 = stidx +blockidx_z*${bn} + tidcol + ${m} + (tidrow + ${l} +blockidx_y*${bm})*${tot_neles};

            if (tidcol + ${m} + blockidx_z*${bn} < neles && tidrow + ${l} + blockidx_y*${bm} < ${ncol}){
                buff2[idx1] = res[${l*bn//blkx + m//bk}];
            }
        % endfor
    % endfor
}