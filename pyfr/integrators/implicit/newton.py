import numpy as np
import math
from pyfr.integrators.base import BaseCommon, BaseIntegrator
from pyfr.integrators.implicit.gmres import GMRESSolver
from pyfr.integrators.implicit.jacobi import BlockJacobi
from pyfr.mpiutil import get_comm_rank_root, mpi

class BaseNonLinearSolver(BaseCommon):
	def __init__(self, backend, systemcls, rallocs, mesh, initsoln, cfg, 
				 nstages, stepper_nregs, tstart, dt):

		self.backend = backend

		sect = 'solver-time-integrator'
		self.gmres_maxiters = cfg.getint(sect, 'gmres-niters', 10)
		self.ntol = cfg.getfloat(sect, 'ntol', 0.01)
		self._crdown = cfg.getfloat(sect, 'crdown', 0.3)

		# Truncation Error Constant
		self._errbias = cfg.getfloat(sect, 'err-bias', 1.5)

		# Error norm
		self._norm = cfg.get(sect, 'errest-norm', 'l2')

		# Max iters
		self._max_newtoniters = cfg.getint(sect, 'newton-niters')

		# Total budget
		self._budget = self._max_newtoniters*self.gmres_maxiters

		# Total stages
		self.nstages = nstages

		# Linesearch parameters
		self._ls = cfg.getbool(sect, 'linesearch', False)
		if self._ls:
			self._ls_maxiter = cfg.getint(sect, 'linesearch-max-iter',
											5)
			self._ls_fact = cfg.getfloat(sect, 'linesearch-fact', 0.5)
			self._ls_alpha = cfg.getfloat(sect, 'linesearch-c1', 1e-4)

		self.register = Register(nstages, stepper_nregs,
						         self.gmres_maxiters, self.solver_nregs, cfg)
		nregs = self.register.nregs

		# Construct the relevant system
		self.system = system = systemcls(backend, rallocs, 
							   mesh, initsoln, nregs=nregs, 
							   cfg=cfg)
		self.gmres_solver = GMRESSolver(backend, system, cfg, 
										self.register, dt)

		self._idxcurr = self.register._stepper_regidx[0]

		# Get global ndofs
		self._gndofs = self._get_gndofs()

	@property
	def _idxprev(self):
		rprev = set(self.register._stepper_regidx[:2]) - {self._idxcurr}
		return next(iter(rprev))

class NewtonSolver(BaseNonLinearSolver):
	solver_name = 'newton'
	solver_nregs = 1

	def _rms_norm(self, reg):
		norm = self._eval_norm(reg)
		return norm / np.sqrt(self._gndofs)

	def init_step(self, t, nsteps=0, nrjctchain=0):
		rhs = self.system.rhs
		rcurr, rprev = self._idxcurr, self._idxprev
		rprev_rhs = self.register._stage_regidx[0]
		rcurr_rhs = self.register._stage_regidx[-1]

		if nsteps > 0 and nrjctchain == 0:
			kerns = self._get_copy_kerns(rprev_rhs, rcurr_rhs)
			self.backend.run_kernels(kerns)
		else:
			rhs(t, rprev, rprev_rhs)

	def obtain_solution(self, bcoeffs):
		rcurr, rprev = self._idxcurr, self._idxprev

		regidxs = [rcurr] + [rprev]
		regidxs += self.register._stage_regidx
		coeffs = [0, 1] + bcoeffs

		self._addv(coeffs, regidxs)

	def store_current_solution(self):
		rcurr, rprev = self._idxcurr, self._idxprev
		self._add(0, rprev, 1, rcurr)

	def _prev_stage_accum(self, acoeffs, currstg):

		raux = self.register._stageaccum_regidx
		rprev = self._idxprev

		regidxs = [raux] + self.register._stage_regidx[:currstg]
		regidxs += [rprev]
		consts = [0, *acoeffs[:-1], 1]

		self._addv(consts, regidxs)

		return raux

	def _residual(self, tc, acoeffs, currstg, *regs):

		# Get registers
		rcurr, rout = regs
		rprev = self._idxprev
		rcurr_rhs = self.register._stage_regidx[currstg]
		rstg = self.register._stageaccum_regidx

		# Eval RHS
		self.system.rhs(tc, rcurr, rcurr_rhs)

		# Eval Residual
		consts = [0, acoeffs[-1], 1, -1]
		regidxs = [rout, rcurr_rhs]
		regidxs += [rstg, rcurr]

		self._addv(consts, regidxs)

	def _linesearch(self, tc, nnorm_init, acoeffs, currstg,
				    rcurr, rdu):
		alpha = self._ls_alpha
		sold, snew = 1, 1
		comm, rank, root = get_comm_rank_root()

		def f(s):
			# Get aux registers
			rduaux = self.register._aux_regidx

			# Eval rhs
			self._add(0, rduaux, 1, rcurr, s, rdu)
			self._residual(tc, acoeffs, currstg, rduaux, rduaux)

			return 0.5*self._rms_norm(rduaux)**2

		# Get function vals
		f0, df0 = 0.5*nnorm_init**2, -nnorm_init**2
		fs = f(snew)

		# Return if we have a good stepsize
		if not math.isnan(fs) and fs < f0 + alpha*snew*df0:
			return snew

		# Evaluate stepsize
		sold = snew
		snew = -df0 / (2*(fs - f0 - df0))
		fs = f(snew)
		if rank == root:
			print(f'snew is {snew} L 165', flush=True)

		while fs > f0 + alpha*snew*df0:
			fc, fp = f(snew), f(sold)

			# Create the cubic and its derivative
			roots = [fc, fp, f0, df0]
			coeffs = np.poly(roots)
			dcoeffs = np.polyder(coeffs)

			# Find minima
			stemp = np.amax(np.roots(dcoeffs))

			sold = snew
			snew = sold * max(min(stemp/sold, 0.5), 0.1)

			fs = f(snew)
		
			if rank == root:
				print(f'snew is {snew} L 184', flush=True)
		return snew

	def _init_stage(self, acoeffs, currstg):
		rcurr, rprev = self._idxcurr, self._idxprev

		consts = [0, 1, *acoeffs[:-1]]
		regidxs=  [rcurr, rprev] + self.register._stage_regidx[:currstg]
		self._addv(consts, regidxs)

	def solve(self, tc, acoeffs, currstg):
		comm, rank, root = get_comm_rank_root()
		rcurr, rprev = self._idxcurr, self._idxprev
		gmres_solver = self.gmres_solver

		# Begin nonlinear iteration count
		newton_iter = 0
		gmres_niters = 0

		# Get registers
		rcurr = self._idxcurr
		rdu0 = self.register._gmres_regidx[0]

		# Accumulate previous stages
		self._prev_stage_accum(acoeffs, currstg)

		# Evaluate residual
		self._residual(tc, acoeffs, currstg, 
				       rcurr, rdu0)

		nnorm_init = self._rms_norm(rdu0)
		nnorm = nnorm_init

		while nnorm / nnorm_init > self.ntol:

			# Step size
			alpha = 1

			# Linear Solve
			rdu, iters = gmres_solver.solve(tc, acoeffs[-1], currstg,
								 			       rcurr)

			if self._ls:
				alpha = self._linesearch(tc, nnorm, acoeffs, 
							             currstg, rcurr, rdu)

			self._add(1, rcurr, alpha, rdu)
			self._residual(tc, acoeffs, currstg, rcurr, rdu0)

			nnorm = self._rms_norm(rdu0)
			newton_iter	+= 1
			gmres_niters += iters

		res = (nnorm/nnorm_init) / self.ntol
		err = gmres_niters / self._budget if math.isfinite(res) else res

		if rank == root:
			print(f'stage is {currstg}', flush=True)
			print(f'nnorm is {nnorm / nnorm_init}, newton is {newton_iter}', flush=True)

		return rcurr, rprev, err

class Register:
	def __init__(self, stage_nregs, stepper_nregs, niters, solver_nregs, cfg):
		self.stage_nregs = stage_nregs
		self.stepper_nregs = stepper_nregs
		self.solver_nregs = solver_nregs

		sect = 'solver-time-integrator'
		self.prec =  cfg.getbool(sect, 'precondition', False)

		self.aux_nregs = 1
		self.niters = niters
		self.gmres_nregs = self.niters + 1
		self.nregs = (self.solver_nregs + self.stage_nregs + 
					  self.stepper_nregs + self.gmres_nregs
					  + self.aux_nregs)
		
		if self.prec:
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
	def _stageaccum_regidx(self):
		return (self.stepper_nregs + self.stage_nregs 
		  		+ self.gmres_nregs)

	@property
	def _err_regidx(self):
		return self._stepper_regidx[2]

	@property
	def _aux_regidx(self):
		return (self.stepper_nregs + self.stage_nregs
		  		+ self.gmres_nregs + self.solver_nregs)

	@property
	def _prec_regidx(self):
		ix = self.nregs - self.gmres_nregs
		if self.prec:
			return self._regidx[ix:ix+self.gmres_nregs]
		else:
			return self._regidx[:self.gmres_nregs]