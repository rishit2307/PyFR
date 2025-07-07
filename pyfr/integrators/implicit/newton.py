import numpy as np
from pyfr.integrators.base import BaseCommon, BaseIntegrator
from pyfr.integrators.implicit.gmres import GMRESSolver
from pyfr.integrators.implicit.jacobi import BlockJacobi
from pyfr.mpiutil import get_comm_rank_root

class BaseNonLinearSolver(BaseCommon):
	def __init__(self, backend, systemcls, rallocs, mesh, initsoln, cfg, 
				 stage_nregs, stepper_nregs, tstart, dt):
		
		self.backend = backend
		
		sect = 'solver-time-integrator'
		niters = cfg.getint(sect, 'gmres-niters', 10)
		self.ntol = cfg.getfloat(sect, 'ntol', 1e-2)

		self.register = Register(stage_nregs, stepper_nregs,
						         niters, self.solver_nregs, cfg)
		nregs = self.register.nregs

		# Construct the relevant system
		self.system = system = systemcls(backend, rallocs, 
							   mesh, initsoln, nregs=nregs, 
							   cfg=cfg)

		self.gmres_solver = GMRESSolver(backend, system, cfg, 
										self.register, tstart, dt)

		self._idxcurr = 0

	def obtain_solution(self, bcoeffs):
		regs = self.register
		regidxs = [regs._curr_regidx] + [regs._prev_regidx]
		regidxs += regs._stage_regidx
		coeffs = [0.0, 1.0] + bcoeffs

		self._addv(coeffs, regidxs)

	def store_current_solution(self):
		regs = self.register
		self._add(0.0, regs._prev_regidx, 1.0, regs._curr_regidx)

		return regs._prev_regidx

class NewtonSolver(BaseNonLinearSolver):
	solver_name = 'newton'
	solver_nregs = 1

	def init_step(self, t):
		rhs = self.system.rhs
		rprev = self.register._prev_regidx
		rprev_rhs = self.register._stage_regidx[0]

		rhs(t, rprev, rprev_rhs)

	def _update_rhs(self, tc, acoeffs, currstg):
		rhs = self.system.rhs
		gndofs = self._get_gndofs()

		rcurr_rhs = self.register._stage_regidx[currstg]
		rcurr = self.register._curr_regidx
		rprev = self.register._prev_regidx
		rdu0 = self.register._gmres_regidx[0]

		rhs(tc, rcurr, rcurr_rhs)
		ac0 = acoeffs[0]

		consts = [0.0, *acoeffs[1:], 1.0/ac0, -1.0/ac0]
		regidxs = [rdu0] + self.register._stage_regidx[:currstg+1]
		regidxs += [rprev, rcurr]

		self._addv(consts, regidxs)

		return self.eval_norm2(rdu0)/np.sqrt(gndofs)

	def solve(self, tc, acoeffs, currstg):
		rcurr = self.register._curr_regidx
		rduold = self.register._duold_regidx

		nnorm = self._update_rhs(tc, acoeffs, currstg)
		nnorm = np.inf
		newtoniter = 0
		comm, rank, root = get_comm_rank_root()

		while nnorm > self.ntol:
			rdu = self.gmres_solver.solve(tc, acoeffs[0], currstg)

			self._add(1.0, rcurr, 1.0, rdu)
			self._add(0.0, rduold, 1.0, rdu)

			nnorm = self._update_rhs(tc, acoeffs, currstg)
			newtoniter += 1 

			if rank == root:
				print(f'stage is {currstg}')
				print(f'nnorm is {nnorm}, newton is {newtoniter}')

class Register:
	def __init__(self, stage_nregs, stepper_nregs, niters, solver_nregs, cfg):
		self.stage_nregs = stage_nregs
		self.stepper_nregs = stepper_nregs
		self.solver_nregs = solver_nregs
		
		sect = 'solver-time-integrator'
		prec =  cfg.get(sect, 'precondition', None)

		self.aux_nregs = 1
		self.jacobi_nregs = 2
		self.niters = niters
		self.gmres_nregs = self.niters + 1
		self.nregs = (self.solver_nregs + self.stage_nregs + 
					  self.stepper_nregs + self.gmres_nregs
					  + self.aux_nregs + self.jacobi_nregs)
		
		if prec:
			self.nregs += self.gmres_nregs

		self._regidx = list(range(self.nregs))
	
	@property
	def _gmres_regidx(self):
		return self._regidx[:self.gmres_nregs]

	@property
	def _stepper_regidx(self):
		ix = self.gmres_nregs
		return self._regidx[ix:ix + self.stepper_nregs]

	@property
	def _stage_regidx(self):
		ix = self.gmres_nregs + self.stepper_nregs
		return self._regidx[ix : ix + self.stage_nregs]

	@property
	def _duold_regidx(self):
		return (self.stepper_nregs + self.stage_nregs 
		  		+ self.gmres_nregs)

	@property
	def _curr_regidx(self):
		return self._stepper_regidx[0]

	@property
	def _prev_regidx(self):
		return self._stepper_regidx[1]

	@property
	def _aux_regidx(self):
		return (self.stepper_nregs + self.stage_nregs
		  		+ self.gmres_nregs + self.solver_nregs
				+ self.jacobi_nregs)

	@property
	def _jacobi_regidx(self):
		ix =  (self.stepper_nregs + self.stage_nregs
		  		+ self.gmres_nregs + self.solver_nregs)
		return self._regidx[ix : ix + self.jacobi_nregs]

	@property
	def _prec_regidx(self):
		ix = self.nregs - self.gmres_nregs
		return self._regidx[ix:ix+self.gmres_nregs]