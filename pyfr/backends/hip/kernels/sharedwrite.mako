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

sidx = tidcolA*${bm+2};
% for l in range(0, bm, blkx//bk):
    ## assert(sidx +  (tidrowA + ${l%(blkx//K)})*${bm*K//blkx} + ${l*K//blkx} < ${(bm+2)*bk} && "ERROR sharedL21");
    ## assert( ${l*bk//blkx} < 40 && "ERROR sharedL22");
    sA[sidx +  (tidrowA + ${l%(blkx//K)})*${bm*K//blkx} + ${l*K//blkx}] = regAtmp[${l*bk//blkx}];
% endfor

sidx = tidcolsubB*${bn//K} + tidrowsubB;
% for l in range(0, bk, blkx//bn):
    ## assert((tidrowB + ${l})*${bn+2} + sidx < ${(bn+2)*bk} && "ERROR shared L27");
    ##     assert( ${l*bn//blkx} < 1 && "ERROR sharedL29");

    sB[(tidrowB + ${l})*${bn+2} + sidx] = regBtmp[${l*bn//blkx}];
% endfor

</%pyfr:macro>


