<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%pyfr:macro name='sharedwrite' params=''>

% for l in range(0, bm, blkx//bk):
    sA[tidcol*${bm+2} + tidrow*${bm*bk//blkx} + ${l*bk//blkx}] = regAtmp[${l*bk//blkx}];
% endfor

% for l in range(0, bk, blkx//bk):
    % for m in range(0, bn, bk):
        sB[(tidrow + ${l})*${bn + 6} + tidcolB + ${m//bk}] = regBtmp[${(l*bk//blkx)*bn//bk} + ${m//bk}];
    % endfor
% endfor
## % for l in range(0, bk, blkx//bn):
##     sB[(tidcolBsub*${bn//bk} + tidrowBsub) + (tidrowB + ${l})*${bn+4}] = regBtmp[${l*bn//blkx}];
## % endfor

</%pyfr:macro>