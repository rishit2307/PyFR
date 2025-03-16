<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>


__global__ void
getf3(ixdtype_t nrow, ixdtype_t ldim, fpdtype_t *__restrict__ jac, 
      fpdtype_t *__restrict__ jacinv,  ixdtype_t *__restrict__ P)
{
    ixdtype_t idx, idx1, idx0, idx2;
    ixdtype_t nb = 16;
    fpdtype_t acc;

    ixdtype_t tid = threadIdx.x;
    __shared__ fpdtype_t mv;
    __shared__ ixdtype_t mix;
    __shared__ fpdtype_t smv[256];
    __shared__ fpdtype_t sB[256];
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
                    smv[threadIdx.x / warpSize] = acc;

                __syncthreads();

                if (threadIdx.x / warpSize == 0){
                    acc = (threadIdx.x  < blockDim.x / warpSize) ? smv[threadIdx.x] : 0;

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
            for (int k=j+1; k < nrow; k+=nb){
                idx = blockIdx.x*ldim + (tidrow+k)*nrow + tidcol + j + 1;
                idx0 = blockIdx.x*ldim + (tidrow+k)*nrow + j;

                if (tidcol < min(i + nb, nrow) - j - 1 && tidrow + k < nrow)
                    jac[idx] -= jac[idx1]*jac[idx0];
            }
            __syncthreads();
        }

        // Initialize shared memory for L21
        for (int j=0; j < 256; j+=blockDim.x)
            smv[threadIdx.x + j] = 0;

        __syncthreads();

        // Invert L21 and store in shared memory
        for (int j=0; j < nb; j++){
           
           idx = threadIdx.x + j*(j-1)/2;
           idx1 = idx;
           if (threadIdx.x < j){

            for (int k=j-1; k > threadIdx.x; k--){
                idx0 = blockIdx.x*ldim + (i+j)*nrow + k + i;
                idx1 -= k;
                smv[idx] -= jac[idx0]*smv[idx1];
            }
           }
           __syncthreads();

            if (threadIdx.x < j){
                idx0 = blockIdx.x*ldim + (i+j)*nrow + threadIdx.x + i;
                smv[idx] -= jac[idx0];
            }
            __syncthreads();
        }

        // Multiply L21^(-1) with A12
        idx0 = tidrow*(tidrow-1)/2;

        for (int j=i+nb; j < nrow; j += nb){
            idx = blockIdx.x*ldim + (i+tidrow)*nrow + tidcol + j;
            if (tidcol < nrow - j)
                sB[tidrow*nb + tidcol] = jac[idx];
            
            __syncthreads();

            if (tidcol < nrow - j){
                temp = sB[tidrow*nb + tidcol];
                for (int l=0; l< tidrow ; l++)
                    temp += sB[l*nb + tidcol]*smv[idx0 + l];
            }
            __syncthreads();

            if (tidcol < nrow - j)
                jac[idx] = temp;

        }
        __syncthreads();

        // Schur Complement
        for (int j = i+nb; j < nrow; j+=nb){
            idx = blockIdx.x*ldim + (i+tidrow)*nrow + j + tidcol;

            if (tidcol < nrow - j)
                smv[tidrow*nb + tidcol] = jac[idx];

            __syncthreads();
            for (int k=i+nb; k < nrow; k+=nb){
                idx1 = blockIdx.x*ldim + (k+tidrow)*nrow + j + tidcol;
                
                if (tidrow < nrow - k && tidcol < nrow - j){
                    for (int l=0; l < nb; l++){
                        idx0 = blockIdx.x*ldim + (k+tidrow)*nrow + l + i;
                        jac[idx1] -= jac[idx0]*smv[l*nb + tidcol];
                    }
                }
            }
            __syncthreads();
           
        }   
    }

  
}