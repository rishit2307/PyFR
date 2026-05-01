<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

__global__ void
axnpby(ixdtype_t nrow, ixdtype_t ncolb, ixdtype_t ldim,
       fpdtype_t* __restrict__ x0,
       ${', '.join(f'const fpdtype_t* __restrict__ x{i}'
                   for i in range(1, nv))},
       ${', '.join(f'fpdtype_t a{i}' for i in range(nv))})
{
% if inscales:
    const fpdtype_t _in[] = ${pyfr.carray(inscales)};
% endif
% if outscales:
    const fpdtype_t _out[] = ${pyfr.carray(outscales)};
% endif
    int i = blockIdx.y*blockDim.y + threadIdx.y;
    ixdtype_t j = ixdtype_t(blockIdx.x)*blockDim.x + threadIdx.x;
    ixdtype_t idx;

    if (j < ncolb && a0 == 0.0)
    {
    % for k in subdims:
        idx = i*ldim + SOA_IX(j, ${k}, ${ncola});
        x0[idx] = ${pyfr.axnpby_expr(k, 'idx', 1, nv=nv, in_idx=in_idx, inscale=inscales, outscale=outscales)};
    % endfor
    }
    else if (j < ncolb && a0 == 1.0)
    {
    % for k in subdims:
        idx = i*ldim + SOA_IX(j, ${k}, ${ncola});
        x0[idx] += ${pyfr.axnpby_expr(k, 'idx', 1, nv=nv, in_idx=in_idx, inscale=inscales, outscale=outscales)};
    % endfor
    }
    else if (j < ncolb)
    {
    % for k in subdims:
        idx = i*ldim + SOA_IX(j, ${k}, ${ncola});
        x0[idx] = ${pyfr.axnpby_expr(k, 'idx', 0, nv=nv, in_idx=in_idx, outscale=outscales)};
    % endfor
    }
}
