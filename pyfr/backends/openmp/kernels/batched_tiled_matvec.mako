<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%
x_leaddim = nvars*csubsz
x_blocksz = nupts*x_leaddim
mixed = pcdtype != pyfr.npdtype_to_ctype(fpdtype)
%>

struct batched_tiled_matvec_kargs
{
    ixdtype_t neles;
    ${pcdtype} *minv_v;
    fpdtype_t *x_v, *y_v;
};

void batched_tiled_matvec(const struct batched_tiled_matvec_kargs *restrict args)
{
    ixdtype_t neles = args->neles;
    ${pcdtype} *minv_v = args->minv_v;
    fpdtype_t *x_v = args->x_v, *y_v = args->y_v;

% if in_scale:
    static const fpdtype_t in_scale[] = ${pyfr.carray(in_scale)};
% endif
% if out_scale:
    static const fpdtype_t out_scale[] = ${pyfr.carray(out_scale)};
% endif

    #pragma omp parallel for ${schedule}
    for (ixdtype_t ib = 0; ib < (neles + ${csubsz} - 1) / ${csubsz}; ib++)
    {
        ixdtype_t rem = neles - ib*${csubsz};
        ${pcdtype} *minv = minv_v + ib*${ntiles_c**2*tcols**2*csubsz};
        fpdtype_t *x = x_v + ib*${x_blocksz}, *y = y_v + ib*${x_blocksz};

        for (ixdtype_t s = 0; s < ${csubsz // soasz}; s++)
        {
            ixdtype_t raw = rem - s*SOA_SZ;
            if (raw <= 0) continue;
            ixdtype_t active = min(raw, (ixdtype_t)SOA_SZ);

% if tcols > 8:
            for (ixdtype_t tr = 0; tr < ${ntiles_c}; tr++)
            {
                ixdtype_t row0 = tr*${tcols};
                ixdtype_t nrowt = min(${tcols}, ${block_size} - row0);
                alignas(64) fpdtype_t acc[${tcols}][SOA_SZ] = {{0}};

                for (ixdtype_t tc = 0; tc < ${ntiles_c}; tc++)
                {
                    ixdtype_t col0 = tc*${tcols};
                    ixdtype_t ncolt = min(${tcols}, ${block_size} - col0);
                    alignas(64) fpdtype_t xcache[${tcols}][SOA_SZ];

                    for (ixdtype_t lc = 0; lc < ncolt; lc++)
                    {
                        ixdtype_t col = col0 + lc;
                        ixdtype_t x_upt = col / ${nvars}, x_var = col % ${nvars};
                        fpdtype_t *x_ptr = x + x_upt*${x_leaddim} + (s*${nvars} + x_var)*SOA_SZ;
% if in_scale:
                        fpdtype_t xs = in_scale[x_var];
% endif

                        #pragma omp simd
                        for (ixdtype_t lane = 0; lane < active; lane++)
                            xcache[lc][lane] = x_ptr[lane]${'*xs' if in_scale else ''};
                    }

                    for (ixdtype_t lr = 0; lr < nrowt; lr++)
                    {
                        ${pcdtype} *trow = minv + tr*${ntiles_c*tcols**2*csubsz} + tc*${tcols**2*csubsz} + lr*${tcols*csubsz} + s*${tcols*soasz};

% if mixed:
                        alignas(64) fpdtype_t mcache[${tcols}][SOA_SZ];
                        for (ixdtype_t lc = 0; lc < ncolt; lc++)
                        {
                            ${pcdtype} *m_ptr = trow + lc*SOA_SZ;
                            for (ixdtype_t lane = 0; lane < SOA_SZ; lane++)
                                mcache[lc][lane] = m_ptr[lane];
                        }
% endif

                        for (ixdtype_t lc = 0; lc < ncolt; lc++)
                        {
                            fpdtype_t *mp = ${'mcache[lc]' if mixed else 'trow + lc*SOA_SZ'};

                            #pragma omp simd
                            for (ixdtype_t lane = 0; lane < active; lane++)
                                acc[lr][lane] += mp[lane]*xcache[lc][lane];
                        }
                    }
                }

                for (ixdtype_t row = row0; row < row0 + nrowt; row++)
                {
                    ixdtype_t y_upt = row / ${nvars}, y_var = row % ${nvars};
                    fpdtype_t *y_ptr = y + y_upt*${x_leaddim} + (s*${nvars} + y_var)*SOA_SZ;
% if out_scale:
                    fpdtype_t ys = out_scale[y_var];
% endif

                    #pragma omp simd
                    for (ixdtype_t lane = 0; lane < active; lane++)
                        y_ptr[lane] = ${'ys*' if out_scale else ''}acc[row - row0][lane];
                }
            }
% else:
            for (ixdtype_t row = 0; row < ${block_size}; row++)
            {
                alignas(64) fpdtype_t acc[SOA_SZ] = {0};
                ixdtype_t tr = row / ${tcols}, lr = row % ${tcols};

                for (ixdtype_t tc = 0; tc < ${ntiles_c}; tc++)
                {
                    ${pcdtype} *trow = minv + tr*${ntiles_c*tcols**2*csubsz} + tc*${tcols**2*csubsz} + lr*${tcols*csubsz} + s*${tcols*soasz};
                    ixdtype_t col0 = tc*${tcols};
                    ixdtype_t ncolt = min(${tcols}, ${block_size} - col0);

                    for (ixdtype_t lc = 0; lc < ncolt; lc++)
                    {
                        ixdtype_t col = col0 + lc;
                        ixdtype_t x_upt = col / ${nvars}, x_var = col % ${nvars};
                        ${pcdtype} *m_ptr = trow + lc*SOA_SZ;
                        fpdtype_t *x_ptr = x + x_upt*${x_leaddim} + (s*${nvars} + x_var)*SOA_SZ;

% if mixed:
                        alignas(64) fpdtype_t m_wide[SOA_SZ];
                        for (ixdtype_t lane = 0; lane < SOA_SZ; lane++)
                            m_wide[lane] = m_ptr[lane];
% endif

% if in_scale:
                        fpdtype_t xs = in_scale[x_var];
% endif

                        #pragma omp simd
                        for (ixdtype_t lane = 0; lane < active; lane++)
                            acc[lane] += ${'m_wide' if mixed else 'm_ptr'}[lane]*${'(x_ptr[lane]*xs)' if in_scale else 'x_ptr[lane]'};
                    }
                }

                ixdtype_t y_upt = row / ${nvars}, y_var = row % ${nvars};
                fpdtype_t *y_ptr = y + y_upt*${x_leaddim} + (s*${nvars} + y_var)*SOA_SZ;
% if out_scale:
                fpdtype_t ys = out_scale[y_var];
% endif

                #pragma omp simd
                for (ixdtype_t lane = 0; lane < active; lane++)
                    y_ptr[lane] = ${'ys*' if out_scale else ''}acc[lane];
            }
% endif
        }
    }
}
