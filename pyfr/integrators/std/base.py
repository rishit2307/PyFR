from pyfr.integrators.base import BaseIntegrator
from pyfr.integrators.base import BaseCommon


class BaseStdIntegrator(BaseCommon, BaseIntegrator):
    formulation = 'std'

    def __init__(self, backend, systemcls, rallocs, mesh, initsoln, cfg):
        super().__init__(backend, rallocs, mesh, initsoln, cfg)

        # Sanity checks
        if self.controller_needs_errest and not self.stepper_has_errest:
            raise TypeError('Incompatible stepper/controller combination')
        
        if cfg.get('solver-time-integrator', 'multip', 'false') == 'true':
            self.gmresniter =  self.cfg.getint('solver-time-integrator', 'gmres-iter') + 1
        # Determine the amount of temp storage required by this method
            self.nregs = (self.gmresniter + self.stepper_nregs + self.eval_nreg
                      + self.eval_src + self.uru_nreg + self.duold_nreg)
        else:
            self.nregs = self.stepper_nregs

        # Construct the relevant system
        self.system = systemcls(backend, rallocs, mesh, initsoln,
                                nregs=self.nregs, cfg=cfg)

        # Event handlers for advance_to
        self.plugins = self._get_plugins(initsoln)

        # Commit the sytem
        self.system.commit()

        # Register index list and current index
        self._regidx = list(range(self.nregs))
        self._idxcurr = 0

        # Pre-process solution if necessary
        self.system.preproc(self.tcurr,
                            self.system.ele_scal_upts(self._idxcurr))

        # Global degree of freedom count
        self._gndofs = self._get_gndofs()

    @property
    def _du_regidx(self):
        return self.k
    
    @property
    def _mvec_regidx(self):
        return self.gmresniter + self.stepper_nregs + self.uru_nreg + self.duold_nreg + self.pseudo_nregs
    
    @property
    def _up_rup_regidx(self):
        st = self.gmresniter + self.uru_nreg
        ed = st + self.stepper_nregs
        return range(st, ed)
    
    @property
    def _u_ru_regidx(self):
        return range(self.gmresniter, self.gmresniter+self.uru_nreg)
    
    @property
    def _duold_regidx(self):
        ngmres = self.gmresniter
        return ngmres+self.uru_nreg+self.stepper_nregs
    
    @property
    def _pseudo_regidx(self):
        st = self.gmresniter+self.stepper_nregs+self.uprup_nreg+self.duold_nreg
        return range(st, st+self.pseudo_nregs)
    
    @property
    def _src_regidx(self):
        return self.nregs - 1

    def _gmres_j_regidx(self, j):
        return j

    @property
    def soln(self):
        if not self._curr_soln:
            self.system.postproc(self._idxcurr)
            self._curr_soln = self.system.ele_scal_upts(self._idxcurr)

        return self._curr_soln

    @property
    def grad_soln(self):
        system = self.system

        if not self._curr_grad_soln:
            system.postproc(self._idxcurr)
            system.compute_grads(self.tcurr, self._idxcurr)
            self._curr_grad_soln = [e.get() for e in system.eles_vect_upts]

        return self._curr_grad_soln

    @property
    def dt_soln(self):
        system = self.system

        if not self._curr_dt_soln:
            copy = self.soln
            system.rhs(self.tcurr, self._idxcurr, self._idxcurr)
            self._curr_dt_soln = system.ele_scal_upts(self._idxcurr)

            # Reset current register with original contents
            for c, e in zip(copy, system.ele_banks):
                e[self._idxcurr].set(c)

        return self._curr_dt_soln

    @property
    def controller_needs_errest(self):
        pass

    @property
    def entmin(self):
        return self.system.get_ele_entmin_int()

    @entmin.setter
    def entmin(self, value):
        self.system.set_ele_entmin_int(value)
