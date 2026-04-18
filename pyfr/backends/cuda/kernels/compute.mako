
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
        ## assert(l*${(bm+2)} + tidrow*${bm*K//blkx} + ${m} < ${bm+2}*${bk} && "ERROR compute L26");
        ## assert(${m} < 10 && "ERRORcompute L27");

        regM[${m}] = sA[l*${(bm+2)} + tidrow*${bm*K//blkx} + ${m}];
    % endfor

    % for m in range(0, (bn // K)):
        ## assert(l*${(bn+2)} + tidcol*${bn//K} + ${m} < ${bn+2}*${bk} && "ERROR L33");
        ## assert(${m} < 10 && "ERROR L34");
        regN[${m}] = sB[l*${(bn+2)} + tidcol*${bn//K} + ${m}];
    % endfor

    % for m in range(0, bm*K//blkx):
        regMtmp[${m}] = fpdtype_t(cast(regM[${m}]));
    % endfor

    

    % for m in range(bm*K//blkx):
        % for n in range(bn // K):
            ## assert(${m*bn//K +n } < 10 && "ERROR L46 ocmpute");
            ## assert(${m} < ${bm*K//blkx} && "ERROR L 47 compute");
            ## assert(${n} < ${bn//K} && "ERROR L48 compute");
            res[${m*bn//K + n}] += regMtmp[${m}]*regN[${n}];
        % endfor
    % endfor
}
</%pyfr:macro>