<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.backends.cuda.kernels.gemm'/>
<%include file='pyfr.backends.cuda.kernels.ltrmm'/>
<%include file='pyfr.backends.cuda.kernels.utrtri'/>

__global__ void
getf3(ixdtype_t nrow, ixdtype_t ncol,  ixdtype_t ldim, 
      fpdtype_t *__restrict__ jac, fpdtype_t *__restrict__ jacinv,  
      ixdtype_t *__restrict__ P)
{
    ixdtype_t idx, idx1, idx0, idx2;
    const ixdtype_t nb = 32;
    const ixdtype_t nby = ${blksz}/nb;
    fpdtype_t acc;
    ixdtype_t zero = 0;
    fpdtype_t fzero = 0.0;
    ixdtype_t one  = 1;
    ixdtype_t two = 2;
    fpdtype_t fone = 1.0;

    ixdtype_t tid = threadIdx.x;
    __shared__ fpdtype_t mv;
    __shared__ ixdtype_t mix;
    __shared__ fpdtype_t sC[nb*nb];
    __shared__ fpdtype_t sA[nb*nb];
    __shared__ fpdtype_t sB[nb*nb];
    __shared__ fpdtype_t sD[nb*nb];
    __shared__ fpdtype_t sE[nb*nb];
    __shared__ ixdtype_t ix;

    ixdtype_t tidrow = threadIdx.x / nb;
    ixdtype_t tidcol = threadIdx.x % nb;
    fpdtype_t temp;
    ixdtype_t srow;
    ixdtype_t drow;
    ixdtype_t rloc;


    for (ixdtype_t i=zero; i < nrow; i+=nby){
        for (ixdtype_t j=i+1; j < nrow; j+=nb){
            idx = blockIdx.x*ldim + (tidrow + i)*nrow + (tidcol + j);
            if (tidrow + i < nrow && tidcol + j < nrow)
                jacinv[idx] = 0.0;
        }
    }

    for (ixdtype_t j=zero; j < nrow; j+=blockDim.x)
    {
        if (tid < nrow - j)
            P[blockIdx.x*nrow + tid + j] = tid + j;
    }
    for (ixdtype_t i=zero; i < nrow; i+=nb){
        for (ixdtype_t j=i; j < min(i+nb, nrow); j++){
            if (threadIdx.x == zero)
                mv = fzero;

            acc = fzero;

            // Reduce to find the maximum in column j
            for (ixdtype_t k=j; k < nrow; k += blockDim.x){
                idx = blockIdx.x*ldim + (threadIdx.x + k)*nrow + j;
                if (threadIdx.x < nrow - k)
                    acc = fabs(jac[idx]);


                for (int off=warpSize/2; off > zero; off >>= 1)
                    acc = max(__shfl_down_sync(0xFFFFFFFFU, acc, off), acc);

                if (threadIdx.x % warpSize == zero)
                    sA[threadIdx.x / warpSize] = acc;

                __syncthreads();

                if (threadIdx.x / warpSize == zero){
                    acc = (threadIdx.x  < blockDim.x / warpSize) ? sA[threadIdx.x] : fzero;

                    for (int off=warpSize / 2; off > zero; off >>=1){
                        acc = max(__shfl_down_sync(0xFFFFFFFFU, acc, off), acc);
                    }
                }

                if (threadIdx.x == zero)
                    mv = max(mv, acc);
                    
                __syncthreads();
            }
            for (ixdtype_t k=j; k < nrow; k+=blockDim.x){
                idx = blockIdx.x*ldim + (threadIdx.x + k)*nrow + j;
                if (threadIdx.x < nrow - k){
                    if (jac[idx] == mv || jac[idx] == -mv){
                        P[blockIdx.x*nrow + j] = tid + k;
                        mix = tid + k;
                        ## ix = P[blockIdx.x*nrow + j];
                        ## P[blockIdx.x*nrow + j] = P[mix + blockIdx.x*nrow];
                        ## P[threadIdx.x + k + blockIdx.x*nrow] = ix;
                    }
                }
            }
            __syncthreads();
            

            // Swap the corresponding rows
            for (ixdtype_t k=zero; k < nrow; k+= blockDim.x){
                idx = blockIdx.x*ldim + j*nrow + threadIdx.x + k;

                if (threadIdx.x < nrow - k){
                    temp = jac[idx + (mix-j)*nrow];
                    jac[idx +  (mix-j)*nrow] = jac[idx];
                    jac[idx] = temp;
                }
            }

            __syncthreads();

            // Divide the column by maximum
            for (ixdtype_t k=j + one; k < nrow; k += blockDim.x){
                idx = blockIdx.x*ldim + (threadIdx.x + k)*nrow + j;
                idx0 = blockIdx.x*ldim + j*nrow + j;

                if (threadIdx.x < nrow - k)
                    jac[idx] /= jac[idx0];
            }
            __syncthreads();

            // Factorize the Panel
            idx1 = blockIdx.x*ldim + j*nrow + tidcol + j + one;
            for (ixdtype_t k=j+one; k < nrow; k+=nby){
                idx = blockIdx.x*ldim + (tidrow+k)*nrow + tidcol + j + one;
                idx0 = blockIdx.x*ldim + (tidrow+k)*nrow + j;

                if (tidcol < min(i + nb, nrow) - j - one && tidrow + k < nrow)
                    jac[idx] -= jac[idx1]*jac[idx0];
            }
            __syncthreads();
        }

        // Initialize shared memory for L21
        for (ixdtype_t l =zero; l < nb; l+=nby){
            sC[(tidrow+l)*nb + tidcol] = fzero;
            sB[(tidrow+l)*nb + tidcol] = fzero;
        }    
        __syncthreads();

        // Load L21 into shared memory
        idx = tidrow*nb + tidcol;
        for (ixdtype_t j=i; j < min(i+nb, nrow); j+=nby){
            idx1 = blockIdx.x*ldim + (tidrow+j)*nrow + tidcol + i;
            if (tidcol + i < nrow)
                sC[idx + (j-i)*nb] = jac[idx1];
        }

        __syncthreads();

        // Invert L21 and store in shared memory
        idx = (tid+one)*nb + tid;
        if (tid < nb-one)
            sB[idx] = -sC[idx];
        
        if (tid < nb)
            sB[tid*nb + tid] = fone;

        for (ixdtype_t j=tid+two; j < nb; j++){
            idx = j*nb + tid;

            sB[idx] = -sC[idx];

            for (ixdtype_t k=j-one; k > tid ; k--){
                idx1 = k*nb + tid;
                sB[idx] -= sC[j*nb + k]*sB[idx1];
            }
        }

        __syncthreads();
        ${pyfr.expand('write', 'i', 'i', 'sB', 'jacinv')};
        for (ixdtype_t j=i+nb; j < nrow; j+=nb){
            ${pyfr.expand('read', 'i', 'j', 'sA', 'jac')};
            ${pyfr.expand('l1trmm', 'sB', 'sA', 'sC')};
            ${pyfr.expand('write', 'i', 'j', 'sC', 'jac')}
        }

        for (ixdtype_t j=i-nb; j > -nb; j-=nb){
           ${pyfr.expand('read', 'i', 'j', 'sA', 'jac')};
           ${pyfr.expand('l1trmm', 'sB', 'sA', 'sC')}

           for (ixdtype_t k=i-nb; k > j; k-=nb){
                ${pyfr.expand('read', 'i', 'k', 'sA', 'jacinv')};
                ${pyfr.expand('read', 'k', 'j', 'sD', 'jac')};
                ${pyfr.expand('gemm', 'sA', 'sD', 'sE')};
            
                for (ixdtype_t l=zero; l < nb; l+=nby)
                    sC[(tidrow + l)*nb + tidcol] += sE[(tidrow + l)*nb + tidcol];
           }
           ${pyfr.expand('read', 'j', 'j', 'sA', 'jacinv')};
           ${pyfr.expand('l2trmm','sC','sA','sD')};
           ${pyfr.expand('write', 'i', 'j', '-sD', 'jacinv')}
        }

        for (ixdtype_t j=i+nb; j < nrow; j+=nb){
            ${pyfr.expand('read','j', 'i', 'sB', 'jac')};
            for(ixdtype_t k=i+nb; k < nrow; k+=nb){
                ${pyfr.expand('read', 'i', 'k', 'sA', 'jac')}
                ${pyfr.expand('gemm', 'sB', 'sA', 'sC')};

                for(ixdtype_t l=zero; l < nb; l+=nby){
                    idx = blockIdx.x*ldim + (tidrow + l + j)*nrow + (tidcol + k);
                    if (tidrow + l + j < nrow && tidcol +k < nrow)
                        jac[idx] -= sC[(tidrow+l)*nb + tidcol];
                }
            }
        }
        __syncthreads();
    }

    // Swap the columns
    for (ixdtype_t i=nrow-one; i >=zero; i--){
        if (tid == zero)
            ix = P[blockIdx.x*nrow + i];

    __syncthreads();
        for (ixdtype_t j=min(ix, i); j < nrow; j+=blockDim.x){
            idx = blockIdx.x*ldim + (tid + j)*nrow + i;
            idx1 = blockIdx.x*ldim + (tid + j)*nrow + ix;

            if (tid + j < nrow){
                temp = jacinv[idx];
                jacinv[idx] = jacinv[idx1];
                jacinv[idx1] = temp;
            } 
        }
    }

    // Solve TRSM Ux = L^(-1)*P

    for (ixdtype_t i=nrow; i > zero; i-=nb){
        ${pyfr.expand('read', 'max(i-nb, zero)', 'max(i-nb, zero)', 'sA', 'jac')};
        ${pyfr.expand('utrtri', 'sA', 'sB', 'min(nb, i)')};

        for (ixdtype_t j=i; j < nrow; j+=nb){
            ${pyfr.expand('read', 'max(i-nb, zero)', 'j', 'sC', 'jac')};
            for (ixdtype_t k=zero; k < nrow; k+=nb){
                ${pyfr.expand('read', 'j', 'k', 'sD', 'jacinv')};
                ${pyfr.expand('gemm', 'sC', 'sD', 'sE')};
                for (ixdtype_t l=zero; l < min(nb, i); l+=nby){
                    idx = blockIdx.x*ldim + (tidrow + l + max(i-nb, zero))*nrow + tidcol + k;
                    if (tidrow + l + max(i-nb, zero) < i && tidcol + k < nrow)
                        jacinv[idx] -= sE[(tidrow + l)*nb + tidcol];
                }
            }
        }

        for (ixdtype_t j=zero; j < nrow; j+=nb){
            ${pyfr.expand('read', 'max(i-nb, zero)', 'j', 'sD', 'jacinv')};
            ${pyfr.expand('gemm', 'sB', 'sD', 'sE')};

            for (ixdtype_t k=zero; k < min(nb, i); k+=nby){
                idx = blockIdx.x*ldim + (tidrow + k + max(i-nb, zero))*nrow + (tidcol + j);
                if (tidcol + j < nrow && tidrow + k + max(i-nb, zero) < i)
                    jacinv[idx] = sE[(tidrow + k)*nb + tidcol];
            }
            __syncthreads();
        }
    }
}