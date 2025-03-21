<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>


__global__ void
getf3(ixdtype_t nrow, ixdtype_t ldim, fpdtype_t *__restrict__ jac, 
      fpdtype_t *__restrict__ jacinv,  ixdtype_t *__restrict__ P)
{
    ixdtype_t idx, idx1, idx0, idx2;
    const ixdtype_t nb = 32;
    const ixdtype_t nby = ${blksz}/nb;
    fpdtype_t acc;

    ixdtype_t tid = threadIdx.x;
    __shared__ fpdtype_t mv;
    __shared__ ixdtype_t mix;
    __shared__ fpdtype_t sC[nb*nb];
    __shared__ fpdtype_t sA[${blksz}];
    __shared__ fpdtype_t sB[nb*nb];
    __shared__ ixdtype_t ix;

    ixdtype_t tidrow = threadIdx.x / nb;
    ixdtype_t tidcol = threadIdx.x % nb;
    fpdtype_t temp;
    ixdtype_t srow;
    ixdtype_t drow;
    ixdtype_t rloc;


    for (ixdtype_t j=0; j < nrow; j+=blockDim.x)
    {
        if (threadIdx.x < nrow - j)
            P[blockIdx.x*nrow + tid + j] = tid + j;
    }


    for (int i=0; i < nrow; i+=nb){
        for (int j=i; j < min(i+nb, nrow); j++){
            if (threadIdx.x == 0)
                mv = 0;

            acc = 0;

            // Reduce to find the maximum in column j
            for (int k=j; k < nrow; k += blockDim.x){
                idx = blockIdx.x*ldim + (threadIdx.x + k)*nrow + j;
                if (threadIdx.x < nrow - k)
                    acc = fabs(jac[idx]);


                for (int off=warpSize/2; off > 0; off >>= 1)
                    acc = max(__shfl_down_sync(0xFFFFFFFFU, acc, off), acc);

                if (threadIdx.x % warpSize == 0)
                    sA[threadIdx.x / warpSize] = acc;

                __syncthreads();

                if (threadIdx.x / warpSize == 0){
                    acc = (threadIdx.x  < blockDim.x / warpSize) ? sA[threadIdx.x] : 0;

                    for (int off=warpSize / 2; off > 0; off >>=1){
                        acc = max(__shfl_down_sync(0xFFFFFFFFU, acc, off), acc);
                    }
                }

                if (threadIdx.x == 0)
                    mv = max(mv, acc);
                    
                __syncthreads();
            }
            for (int k=j; k < nrow; k+=blockDim.x){
                idx = blockIdx.x*ldim + (threadIdx.x + k)*nrow + j;
                if (threadIdx.x < nrow - k){
                    if (jac[idx] == mv || jac[idx] == -mv){
                        mix = threadIdx.x + k;
                        ix = P[blockIdx.x*nrow + j];
                        P[blockIdx.x*nrow + j] = P[mix + blockIdx.x*nrow];
                        P[threadIdx.x + k + blockIdx.x*nrow] = ix;
                    }
                }
            }
            __syncthreads();
            

            // Swap the corresponding rows
            for (int k=0; k < nrow; k+= blockDim.x){
                idx = blockIdx.x*ldim + j*nrow + threadIdx.x + k;

                if (threadIdx.x < nrow - k){
                    temp = jac[idx + (mix-j)*nrow];
                    jac[idx +  (mix-j)*nrow] = jac[idx];
                    jac[idx] = temp;
                }
            }

            __syncthreads();

            // Divide the column by maximum
            for (int k=j + 1; k < nrow; k += blockDim.x){
                idx = blockIdx.x*ldim + (threadIdx.x + k)*nrow + j;
                idx0 = blockIdx.x*ldim + j*nrow + j;

                if (threadIdx.x < nrow - k)
                    jac[idx] /= jac[idx0];
            }
            __syncthreads();

            // Factorize the Panel
            idx1 = blockIdx.x*ldim + j*nrow + tidcol + j + 1;
            for (int k=j+1; k < nrow; k+=nby){
                idx = blockIdx.x*ldim + (tidrow+k)*nrow + tidcol + j + 1;
                idx0 = blockIdx.x*ldim + (tidrow+k)*nrow + j;

                if (tidcol < min(i + nb, nrow) - j - 1 && tidrow + k < nrow)
                    jac[idx] -= jac[idx1]*jac[idx0];
            }
            __syncthreads();
        }

        ## // Initialize shared memory for L21
        ## for (ixdtype_t l =0; l < nb; l+=nby){
        ##     sC[(tidrow+l)*nb + tidcol] = 0.0;
        ##     sB[(tidrow+l)*nb + tidcol] = 0.0;
        ## }    
        ## __syncthreads();

        ## // Load L21 into shared memory
        ## idx = tidrow*nb + tidcol;
        ## for (ixdtype_t j=i; j < min(i+nb, nrow); j+=nby){
        ##     idx1 = blockIdx.x*ldim + (tidrow+j)*nrow + tidcol + i;
        ##     if (tidcol + i < nrow)
        ##         sC[idx + (j-i)*nb] = jac[idx1];
        ## }

        ## __syncthreads();

        ## // Invert L21 and store in shared memory
        ## idx = (tid+1)*nb + tid;
        ## if (tid < nb-1)
        ##     sB[idx] = -sC[idx];
        
        ## for (ixdtype_t j=tid+2; j < nb; j++){
        ##     idx = j*nb + tid;

        ##     sB[idx] = -sC[idx];

        ##     for (ixdtype_t k=j-1; k > tid ; k--){
        ##         idx1 = k*nb + tid;
        ##         sB[idx] -= sC[j*nb + k]*sB[idx1];
        ##     }
        ## }

        ## __syncthreads();

        ## for (ixdtype_t j=i; j < min(nrow, i+nb); j+=nby){
        ##     idx1 = blockIdx.x*ldim + (tidrow + j)*nrow + tidcol + i;
        ##     idx0 = (tidrow + j-i)*nb + tidcol;

        ##     if (tidcol + i < nrow && tidrow + j < nrow)
        ##         jacinv[idx1] = sB[idx0];
        ## }

        ## for (ixdtype_t l =0; l < nb; l+=nby)
        ##     sC[(tidrow+l)*nb + tidcol] = 0.0;

        ## for (ixdtype_t j=i+nb; j< nrow; j+=nb){
        ##     for (ixdtype_t k=i; k < i+nb; k+=nby){
        ##         idx = blockIdx.x*ldim + (tidrow + k)*nrow + tidcol + j;
        ##         idx0 = (tidrow + k-i)*nb + tidcol;

        ##         if (tidcol + j < nrow)
        ##             sA[tidrow*nb + tidcol] = jac[idx];

        ##         __syncthreads();

        ##         for (ixdtype_t l=0; l < tidrow; l++){
        ##             ## if (tidcol + j < nrow){
        ##                 idx1 = (tidrow+k-i)*nb + l+k-i;
        ##                 sC[idx0] += sB[idx1]*sA[tidcol + l*nb];
        ##             ## }
        ##         }
        ##         sC[idx0] += sA[tidrow*nb + tidcol];

        ##         if (tidcol + j < nrow)
        ##             jac[idx] = sC[idx0];

        ##         for (ixdtype_t l=nby + k-i; l < nb; l+=nby){
        ##             for (ixdtype_t m=0; m < nby; m++){
        ##                 idx1 = (tidrow + l)*nb + tidcol;
        ##                 ## if(tidcol + j < nrow)
        ##                 sC[idx1] += sA[m*nb + tidcol]*sB[(tidrow+l)*nb + m+k-i];
        ##             }
        ##         }
        ##         __syncthreads();
        ##     }

        ##     for (ixdtype_t l =0; l < nb; l+=nby)
        ##         sC[(tidrow+l)*nb + tidcol] = 0.0;
        ## }

        ## for (ixdtype_t j=i+nb; j < nrow; j+=nb){
        ##     for (ixdtype_t k=j; k < min(nrow, j+nb); k+=nby){
        ##         idx = blockIdx.x*ldim + (tidrow + k)*nrow + tidcol + i;
        ##         idx0 = (tidrow + k-j)*nb + tidcol;

        ##         if (tidrow + k < nrow)
        ##             sB[idx0] = jac[idx];    
        ##     }
        ##     __syncthreads();

        ##     for (ixdtype_t k=i+nb; k < nrow; k+=nb){
        ##         for (ixdtype_t l=i; l < i+nb; l+=nby){
        ##             idx = blockIdx.x*ldim + (tidrow + l)*nrow + (tidcol + k);
        ##             idx0 = tidrow*nrow + tidcol;

        ##             if(tidcol + k < nrow)
        ##                 sA[idx0] = jac[idx];

        ##             __syncthreads();

        ##             for (ixdtype_t m=0; m < nb; m+=nby){
        ##                 for (ixdtype_t n=0; n < nby; n++){
        ##                     idx1 = (tidrow+m)*nb + n + l - i;
        ##                     sC[(tidrow+m)*nb + tidcol] += sB[idx1]*sA[n*nb + tidcol];

        ##                 }
        ##             }
        ##             __syncthreads();
        ##         }
        ##         for (ixdtype_t l=0; l < nb; l+=nby){
        ##             idx = blockIdx.x*ldim + (tidrow + l + j)*nrow + (tidcol + k);
        ##             if (tidrow + l + j < nrow && tidcol +k < nrow)
        ##                 jac[idx] -= sC[(tidrow + l)*nrow + tidcol];
        ##         }
        ##     }
        ## }

##         // Schur Complement
##         for (int j = i+nb; j < nrow; j+=nb){
##             idx = blockIdx.x*ldim + (i+tidrow)*nrow + j + tidcol;

##             if (tidcol < nrow - j)
##                 smv[tidrow*nb + tidcol] = jac[idx];

##             __syncthreads();
##             for (int k=i+nb; k < nrow; k+=nb){
##                 idx1 = blockIdx.x*ldim + (k+tidrow)*nrow + j + tidcol;
                
##                 if (tidrow < nrow - k && tidcol < nrow - j){
##                     for (int l=0; l < nb; l++){
##                         idx0 = blockIdx.x*ldim + (k+tidrow)*nrow + l + i;
##                         jac[idx1] -= jac[idx0]*smv[l*nb + tidcol];
##                     }
##                 }
##             }
##             __syncthreads();
           
##         }   
    }

  
}