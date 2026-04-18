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

    // Fixed-size array
    fpdtype_t acc[${ndof // blky + ndof % blky}] = {0.0};

    ixdtype_t nl;

    __shared__ fpdtype_t sA[${blkx}*${blksz}];

    % for i in range(0, ndof, blky):
        uid = (tidy + ${i}) / ${ncola};
        vid = (tidy + ${i}) % ${ncola};

        idx = uid*ldim + SOA_IX(tid, vid, ${ncola});
        if (tid < ncolb && tidy + ${i} < ${ndof})
            r1[idx] = 0.0;
    
    % endfor

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
                
                #pragma unroll 20
                for (ixdtype_t k=${i}; k < nl; ++k){

                    uid = k / ${ncola};
                    vid = k % ${ncola};
                    jidx = (tidy + j)*ldim*nrow + uid*ldim + SOA_IX(tid, vid, ${ncola});
                    sidx = (k-${i})*blockDim.x + threadIdx.x;
                    r1temp += sA[sidx] * cast(jac[jidx]);

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