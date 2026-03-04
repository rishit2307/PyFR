
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%pyfr:macro name='compute' params=''>
for (int l=0; l < ${bk}; l++){
    % for m in range(0, bm*bk//blkx):
        regM[${m}] = sA[l*${(bm+2)} + tidrow*${bm*bk//blkx} + ${m}];
    % endfor

    % for m in range(0, bn // bk):
        regN[${m}] = sB[l*${(bn+6)} + tidcolB + ${m}];
    % endfor

    % for m in range(bm*bk//blkx):
        % for n in range(bn // bk):
            res[${m*bn//bk + n}] += regM[${m}]*regN[${n}];
        % endfor
    % endfor
}
</%pyfr:macro>