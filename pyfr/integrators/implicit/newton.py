import numpy as np
from pyfr.integrators.base import BaseCommon, BaseIntegrator
from pyfr.integrators.implicit.gmres import GMRESSolver
from pyfr.integrators.implicit.jacobi import BlockJacobi

class BaseNonLinearSolver(BaseCommon):
	def __init__(self, backend, systemcls, rallocs, mesh, initsoln, cfg, 
				 stage_nregs, stepper_nregs, tstart):
		
		self.backend = backend
		
		sect = 'solver-time-integrator'
		niters = cfg.getint(sect, 'niters', 10)
		self.ntol = cfg.get(sect, 'ntol', 1e-2)

		self.register = Register(stage_nregs, stepper_nregs, niters, self.solver_nregs)

		# Construct the relevant system
		self.system = system = systemcls(backend, rallocs, 
							   mesh, initsoln, nregs=self.register.nregs, 
							   cfg=cfg)

		self.gmres_solver = gmres_solver = GMRESSolver(backend, system, cfg, 
												 	   self.register, tstart)
		
		self._idxcurr = self.register._idxcurr

		# Global degree of freedom count
		self._gndofs = self._get_gndofs()


class NewtonSolver(BaseNonLinearSolver):
	solver_name = 'newton'
	solver_nregs = 1

	def _update_rhs(self, currstg, stepper_coeffs, dt):
		stepper_coeffs = [sc*dt for sc in stepper_coeffs]
		consts = [0.0, *stepper_coeffs, 1.0, -1.0]

		iter = self.gmres_solver.iter
		rdu = self._gmres_regidx[iter]
		regidxs = [rdu] + self._stage_regidx[:currstg]
		regidxs += self._stepper_regidx
		
		self._addv(consts, regidxs)

	def solve(self, currstg, stepper_coeffs, tcurr, dt):
		nnorm = np.inf
		gmres_solver = self.gmres_solver
		self.register.currstg = currstg
		
		while nnorm > self.ntol:
			self._update_rhs(currstg, stepper_coeffs, dt)
			gmres_solver.solve(tcurr, dt)
			

	def init_stage(self, currstg, t):
		rhs = self.system.rhs
		rU = self._stage_regidx[currstg]

		rhs(t, self._idxcurr, rU)

class Register:
	def __init__(self, stage_nregs, stepper_nregs, niters, solver_nregs):
		self.stage_nregs = stage_nregs
		self.stepper_nregs = stepper_nregs
		self.solver_nregs = solver_nregs

		self.aux_nregs = 1
		self.nregs = (self.solver_nregs + self.stage_nregs + 
					  self.stepper_nregs + niters +  
					  self.aux_nregs)
		
		self._idxcurr = 0
		self.currstg = 0

		self._regidx = list(range(self.nregs))

	@property
	def _stage_regidx(self):
		return self._regidx[:self.stage_nregs]
	
	@property
	def _stepper_regidx(self):
		return self._regidx[self.stage_nregs:self.stepper_nregs]
	
	@property
	def _du_regidx(self):
		ix = self.stepper_nregs + self.stage_nregs
		return self._regidx[ix:ix+self.solver_nregs]

	@property
	def _gmres_regidx(self):
		ix = self.stage_nregs + self.stepper_nregs + self.solver_nregs
		return self._regidx[ix:]

	@property
	def _currstg_regidx(self):
		return self._stage_regidx[self.currstg]
	
	@property
	def _aux_regidx(self):
		return self._regidx[-1]