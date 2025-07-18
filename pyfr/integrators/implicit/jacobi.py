import numpy as np

from pyfr.integrators.base import BaseCommon
from pyfr.util import memoize


class BlockJacobi(BaseCommon):
    def __init__(self, system, backend, register, cfg, epsmc):
        self.system = system
        self.backend = backend
        self.register = register
        self.cfg = cfg
        self.epsmc = epsmc

        sect = 'solver-time-integrator'
        precision = cfg.get('backend', 'precision')
        if precision == 'double':
            self.fpdtype = np.float64
        else:
            self.fpdtype = np.float32

        self.jac_fpdtype = cfg.get(sect, 'jacobi-prec', precision)

        self.jac = {}
        self.jacinv = {}
        self.jacshuff = {}
        self.P = {}

        for etp in self.system.ele_types:
            i = self.system.ele_types.index(etp)
            nupts = self.system.ele_shapes[i][0]
            neles = self.system.ele_shapes[i][-1]
            nvars = self.system.ele_shapes[i][1]

            self.jac[etp] = np.empty((nupts*nvars, nupts, nvars, neles), dtype=self.fpdtype)
            self.P[etp] = np.zeros((neles, nupts*nvars), dtype=self.backend.ixdtype)
            self.jacinv[etp]  = np.zeros((neles, (nupts*nvars)**2), dtype=self.fpdtype)
            self.jacinv[etp] = backend.matrix(self.jacinv[etp].shape, self.jacinv[etp])
            self.jac[etp] = backend.matrix(self.jac[etp].shape, self.jac[etp])

            self.P[etp] = backend.matrix(self.P[etp].shape, self.P[etp], dtype=self.backend.ixdtype)

    def _bind_kerns(self, kerns, *args):
        for k in kerns:
            k.bind(*args)

    def _init_jac(self, etp, celes, rs):
        jac = self.jac[etp]
        kerns = [self.backend.kernel('jacinit', *[em[r] for r in rs] +[jac, celes])
                    for em in self.system.ele_banks]
        return kerns

    def _shuff_jac(self, etp):
        jac0, jac1 = self.jacinv[etp], self.jac[etp]
        kerns = [self.backend.kernel('jacshuffle', *[jac0, jac1])]

        return kerns
    
    @memoize
    def mul_jac(self, *rs):
        kern = []
        em = self.system.ele_banks
        for k, jac in self.jac.items():
            i = self.system.ele_types.index(k)
            kern.append(self.backend.kernel('jacmul', 
                            *[em[i][r] for r in rs] + [jac]))
        
        return kern

    def _eval_jac(self, tc, a, currstg):
        add, rhs = self._add, self.system.rhs
        backend = self.backend
        reg = self.register

        raux = reg._aux_regidx
        rcurr = reg._curr_regidx
        rcurr_rhs = reg._stage_regidx[currstg]

        # h = self.fpdtype(np.sqrt(1 + self.eval_norm2(rcurr))*self.epsmc)
        h = self.epsmc
        afac = 1.0/a

        for etp in sorted(self.system.ele_types):
            i = self.system.ele_types.index(etp)
            nupts = self.system.ele_shapes[i][0]
            celes = self.system.celes[etp]
            for col in range(self.system.ncolours[etp]):
                for v in range(self.system.nvars):
                    for npt in range(nupts):
                        self._addid(celes, [rcurr, raux], npt, v, col, h)
                        rhs(tc, raux, raux)
                        self._add(-1.0, raux, 1.0, rcurr_rhs)
                        kern = self._init_jac(etp, celes, [raux, rcurr])
                        self._bind_kerns(kern, npt, v, col, afac, h)
                        backend.run_kernels(kern)

        for etp in self.system.ele_types:

            kern = backend.kernel('getf3', *[self.jac[etp], self.jacinv[etp], 
                                  self.P[etp]])

            backend.run_kernels([kern])	

            kern = self._shuff_jac(etp)
            backend.run_kernels(kern)
        
        self.backend.wait()


    def _update_precision(self):
        jac_temp = {}
        backend = self.backend
        del self.jacinv

        for etp in self.system.ele_types:
            jac_temp[etp] = self.jac[etp].get()

            del self.jac[etp]
        
        self.jac = {}

        for etp in self.system.ele_types:
            self.jac[etp] = backend.matrix(jac_temp[etp].shape, 
                            jac_temp[etp], dtype=self.jac_fpdtype)
        
        del jac_temp
        del self._memoize_cache_