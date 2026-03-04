<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%pyfr:macro name='sharedwrite' params=''>

sidx = tidcol*${bm+2} + tidrow*${bm*bk//blkx};
% for l in range(0, bm, blkx//bk):
    sA[sidx + ${l*bk//blkx}] = regAtmp[${l*bk//blkx}];
% endfor

sidx = tidcolsubB*${bn//bk} + tidrowsubB;
% for l in range(0, bk, blkx//bn):
    sB[(tidrowB + ${l})*${bn+2} + sidx] = regBtmp[${l*bn//blkx}];
% endfor

</%pyfr:macro>