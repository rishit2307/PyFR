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

        self.jac = jac = {}
        self.jacinv = jacinv = {}
        self.jacshuff = {}
        self.P = {}

        for etp in self.system.ele_types:
            i = self.system.ele_types.index(etp)
            nupts = self.system.ele_shapes[i][0]
            neles = self.system.ele_shapes[i][-1]
            nvars = self.system.ele_shapes[i][1]

            self.jac[etp] = jac[etp] = np.empty((nupts*nvars, nupts, nvars, neles))
            self.P[etp] = np.zeros((neles, nupts*nvars))
            self.jacinv[etp] = jacinv[etp] = np.zeros((neles, (nupts*nvars)**2))
            self.jacinv[etp] = backend.matrix(jacinv[etp].shape, jacinv[etp])
            self.jac[etp] = backend.matrix(jac[etp].shape, jac[etp])

            self.P[etp] = backend.matrix(self.P[etp].shape, self.P[etp], dtype=self.backend.ixdtype)

    def _bind_kerns(self, kerns, *args):
        for k in kerns:
            k.bind(*args)

    def _init_jac(self, r0, etp, celes):
        jac = self.jac[etp]
        kerns = [self.backend.kernel('jacinit', *[em[r0]] + [jac, celes])
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
        jac, jacinv = self.jac, self.jacinv
        reg = self.register

        raux = reg._aux_regidx
        rcurr = reg._curr_regidx
        rcurr_rhs = reg._currstg_rhs_regidx(currstg)

        h = self.eval_norm2(rcurr)*self.epsmc
        
        for etp in sorted(self.system.ele_types):
            i = self.system.ele_types.index(etp)
            nupts = self.system.ele_shapes[i][0]
            celes = self.system.celes[etp]
            for col in range(self.system.ncolours[etp]):
                for v in range(self.system.nvars):
                    for npt in range(nupts):
                        self._addid(celes, [rcurr, raux], npt, v, col, h)
                        rhs(tc, raux, raux)
                        self._add(-a/h, raux, a/h, rcurr_rhs)
                        kerns = self._init_jac(raux, etp, celes)
                        self._bind_kerns(kerns, npt, v, col, 1.0)
                        backend.run_kernels(kerns)

        for etp in self.system.ele_types:

            kern = backend.kernel('getf3', *[jac[etp], jacinv[etp], 
                                  self.P[etp]])

            backend.run_kernels([kern])	

            shufkerns = self._shuff_jac(etp)
            backend.run_kernels(shufkerns)
        
        self.backend.wait()

