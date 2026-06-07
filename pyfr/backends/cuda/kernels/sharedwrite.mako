## <%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
## <%pyfr:macro name='sharedwrite' params=''>

## ## sidx = tidcol*${bm+2} + tidrow*${bm*bk//blkx};
## ## % for l in range(0, bm, blkx//bk):
## ##     sA[sidx + ${l*bk//blkx}] = regAtmp[${l*bk//blkx}];
## ## % endfor

## ## sidx = tidcolsubB*${bn//bk} + tidrowsubB;
## ## % for l in range(0, bk, blkx//bn):
## ##     sB[(tidrowB + ${l})*${bn+2} + sidx] = regBtmp[${l*bn//blkx}];
## ## % endfor

## ## </%pyfr:macro>

<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%pyfr:macro name='sharedwrite' params=''>

sidx = tidcolA*${bm+2} + tidrowA*${bm*K//blkx};
% for l in range(0, bm, blkx//bk):
    sA[sidx +  ${l%(blkx//K) * (bm*K//blkx) + l*K//blkx}] = regAtmp[${l*bk//blkx}];
% endfor

sidx = tidcolsubB*${bn//K} + tidrowsubB + tidrowB*${bn+2};
% for l in range(0, bk, blkx//bn):
    sB[${l * (bn+2)} + sidx] = regBtmp[${l*bn//blkx}];
% endfor

</%pyfr:macro>