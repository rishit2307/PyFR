<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
typedef ${pyfr.npdtype_to_ctype(jac_fpdtype)} jac_fpdtype_t;
__global__ void
jacmul(ixdtype_t nrow, ixdtype_t ncolb, ixdtype_t ldim,
        fpdtype_t *__restrict__ r0, fpdtype_t *__restrict__ r1,
        jac_fpdtype_t *__restrict__ jac)

{
    ixdtype_t tid = threadIdx.x + blockIdx.x*blockDim.x;
    ixdtype_t tidy = threadIdx.y;
    ixdtype_t jidx;
    ixdtype_t uid, vid, nv;
    ixdtype_t idx, sidx;
    fpdtype_t r1temp = 0.0;

    ixdtype_t nl;

    __shared__ fpdtype_t sA[${blkx}*${blksz}];

    if (tid < ncolb){
        #pragma unroll
        for (ixdtype_t i=0; i < ${ndof} - ${blky}; i+=${blky}){
            uid = (tidy + i) / ${ncola};
            vid = (tidy + i) % ${ncola};

            idx = uid*ldim + SOA_IX(tid, vid, ${ncola});
            r1[idx] = 0.0;
        }

        ixdtype_t i = ${ndof} - ${blky};
        uid = (tidy + i) / ${ncola};
        vid = (tidy + i) % ${ncola};

        idx = uid*ldim + SOA_IX(tid, vid, ${ncola});

        if (tidy + i < ${ndof})
            r1[idx] = 0.0;

    }

    % for i in range(0, ndof, blksz):
        nl = min(${i}+${blksz}, ${ndof});

        if (tid < ncolb){
            % if i < ndof - blksz:
                % for j in range(i, i + blksz, blky):
                    uid = (tidy + ${j}) / ${ncola};
                    vid = (tidy + ${j}) % ${ncola};

                    idx = uid*ldim + SOA_IX(tid, vid, ${ncola});
                    sidx = tidy*blockDim.x + threadIdx.x + (${j}-${i})*blockDim.x;
                    sA[sidx] = r0[idx];
                % endfor

            % else:
                % for j in range(i, ndof, blky):
                    % if j < ndof - blky:
                        uid = (tidy + ${j}) / ${ncola};
                        vid = (tidy + ${j}) % ${ncola};

                        idx = uid*ldim + SOA_IX(tid, vid, ${ncola});
                        sidx = tidy*blockDim.x + threadIdx.x + (${j}-${i})*blockDim.x;
                        sA[sidx] = r0[idx];
                    
                    % else:
                        uid = (tidy + ${j}) / ${ncola};
                        vid = (tidy + ${j}) % ${ncola};

                        idx = uid*ldim + SOA_IX(tid, vid, ${ncola});
                        sidx = tidy*blockDim.x + threadIdx.x + (${j}-${i})*blockDim.x;

                        if (tidy + ${j} < ${ndof})
                            sA[sidx] = r0[idx];
                    
                    % endif
                % endfor
            % endif
        }
        __syncthreads();

        for (ixdtype_t j=0; j < ${ndof}; j+=blockDim.y){
            r1temp = 0.0;
            if (tidy + j < ${ndof} && tid < ncolb){
                
                #pragma unroll 40
                for (ixdtype_t k=${i}; k < nl; ++k){

                    uid = k / ${ncola};
                    vid = k % ${ncola};
                    jidx = (tidy + j)*ldim*nrow + uid*ldim + SOA_IX(tid, vid, ${ncola});
                    sidx = (k-${i})*blockDim.x + threadIdx.x;
                    r1temp += sA[sidx] * jac[jidx];

                }
                uid = (tidy + j) / ${ncola};
                vid = (tidy + j) % ${ncola};

                idx = uid*ldim + SOA_IX(tid, vid, ${ncola});
                r1[idx] += r1temp;
            }
        }
        __syncthreads();
    % endfor
}