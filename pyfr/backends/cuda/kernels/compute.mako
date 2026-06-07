
## <%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
## <%pyfr:macro name='compute' params=''>
## for (int l=0; l < ${bk}; l++){
##     % for m in range(0, bm*bk//blkx):
##         regM[${m}] = sA[l*${(bm+2)} + tidrow*${bm*bk//blkx} + ${m}];
##     % endfor

##     % for m in range(0, bn // bk):
##         regN[${m}] = sB[l*${(bn+2)} + tidcol*${bn//bk} + ${m}];
##     % endfor

##     % for m in range(bm*bk//blkx):
##         % for n in range(bn // bk):
##             res[${m*bn//bk + n}] += cast(regM[${m}])*regN[${n}];
##         % endfor
##     % endfor
## }
## </%pyfr:macro>


<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%pyfr:macro name='compute' params=''>
for (int l=0; l < ${bk}; l++){
    % for m in range(0, bm*K//blkx):
        regM[${m}] = sA[l*${(bm+2)} + tidrow*${bm*K//blkx} + ${m}];
    % endfor

    % for m in range(0, (bn // K)):
        regN[${m}] = sB[l*${(bn+2)} + tidcol*${bn//K} + ${m}];
    % endfor

    

    % for m in range(bm*K//blkx):
        % for n in range(bn // K):
            res[${m*bn//K + n}] += fpdtype_t(cast(regM[${m}]))*regN[${n}];
        % endfor
    % endfor
}
</%pyfr:macro>