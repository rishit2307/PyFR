import numpy as np

from pyfr.integrators.base import kernel_getter
from pyfr.integrators.implicit.base import BaseImplicitIntegrator
from pyfr.integrators.implicit.precond import Preconditioner
from pyfr.integrators.registers import DynamicScalarRegister
from pyfr.util import subclass_where


class BaseKrylovSolver(BaseImplicitIntegrator):
    krylov_name = None
    _precond_temp = DynamicScalarRegister()

    _pcdtype_map = {
        'double': np.float64,
        'single': np.float32,
        'half': np.float16,
    }

    def __init__(self, backend, systemcls, mesh, initsoln, cfg):
        sect = 'solver-time-integrator'

        # Finite difference perturbation for JFNK and preconditioner
        if cfg.hasopt(sect, 'krylov-eps'):
            self._krylov_eps = cfg.getfloat(sect, 'krylov-eps')
            self._krylov_eps_adapt = False
        else:
            self._krylov_eps = backend.fpdtype_eps**0.5
            self._krylov_eps_adapt = True

        # Preconditioner type
        pname = cfg.get(sect, 'krylov-precond', 'none').lower()
        self._pccls = subclass_where(Preconditioner, name=pname)

        # Preconditioner working precision
        pstr = cfg.get(sect, 'krylov-precond-precision', 'double').lower()
        try:
            self._pcdtype = self._pcdtype_map[pstr]
        except KeyError:
            raise ValueError('Invalid krylov-precond-precision: must be '
                             'double, single, or half')

        # Clamp preconditioner dtype to backend dtype if necessary
        if np.finfo(self._pcdtype).bits > np.finfo(backend.fpdtype).bits:
            self._pcdtype = backend.fpdtype

        # Allocate preconditioner temp only when active
        self._size_register(self._precond_temp, self._pccls.active)

        super().__init__(backend, systemcls, mesh, initsoln, cfg)

        self._breakdown_tol = 1e3*self.backend.fpdtype_eps

    def _compute_krylov_eps(self, u_reg):
        # If we are adaptive then recompute eps
        if self._krylov_eps_adapt:
            unorm = self._norm2(u_reg)
            self._krylov_eps = ((1 + unorm)*self.backend.fpdtype_eps)**0.5

    def _pre_commit(self):
        ext = self.backend.get_extent('krylov')

        self._preconditioner = self._pccls(
            self.backend, self.system, krylov_eps=self._krylov_eps,
            pcdtype=self._pcdtype
        )

        self._preconditioner.init_matrices(ext)

    def _post_commit(self):
        self._preconditioner.init_kernels()

    @property
    def _precond_computed(self):
        return self._preconditioner.computed

    @property
    def _precond_gdt_built(self):
        return self._preconditioner.gdt_built

    @property
    def _precond_build_wtime(self):
        return self._preconditioner.build_wtime

    def _invalidate_precond(self):
        self._preconditioner.invalidate()

    def _compute_precond(self, t, u_reg, gamma_dt, rhs_fn, f0_reg, up_reg,
                         eps_scales=()):
        self._preconditioner.construct(
            t, u_reg, gamma_dt, rhs_fn, f0_reg, up_reg, self._add,
            self._precond_temp, eps_scales=eps_scales
        )

    def _apply_precond(self, in_reg, out_reg, in_scale=(), out_scale=()):
        kerns = self._get_precond_kerns(in_reg, out_reg,
                                        in_scale=tuple(in_scale),
                                        out_scale=tuple(out_scale))
        self.backend.run_kernels(kerns)

    @kernel_getter
    def _get_precond_kerns(self, emats, in_reg, out_reg, *, in_scale,
                           out_scale=()):
        idx = self.system.ele_banks.index(emats)
        return self._preconditioner.apply_kernel(emats, idx, in_reg, out_reg,
                                                 in_scale, out_scale)
