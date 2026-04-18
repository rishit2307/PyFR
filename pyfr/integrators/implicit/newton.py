import numpy as np
import math
from pyfr.integrators.base import BaseCommon, BaseIntegrator
from pyfr.integrators.implicit.gmres import GMRESSolver
from pyfr.integrators.implicit.jacobi import BlockJacobi
from pyfr.mpiutil import get_comm_rank_root, mpi

class BaseNonLinearSolver(BaseCommon):
	def __init__(self, backend, systemcls, rallocs, mesh, initsoln, cfg, 
				 stage_nregs, stepper_nregs, tstart, dt):
		
		self.backend = backend

		sect = 'solver-time-integrator'
		niters = cfg.getint(sect, 'gmres-niters', 10)
		self.ntol = cfg.getfloat(sect, 'ntol', 0.01)
		self._crdown = cfg.getfloat(sect, 'crdown', 0.3)

		# Truncation Error Constant
		self._errbias = cfg.getfloat(sect, 'err-bias', 1.5)

		# Error norm
		self._norm = cfg.get(sect, 'errest-norm', 'l2')

		# Max iters
		self._max_newtoniters = cfg.getint(sect, 'newton-niters')

		self.register = Register(stage_nregs, stepper_nregs,
						         niters, self.solver_nregs, cfg)
		nregs = self.register.nregs

		# Construct the relevant system
		self.system = system = systemcls(backend, rallocs, 
							   mesh, initsoln, nregs=nregs, 
							   cfg=cfg)
		self.gmres_solver = GMRESSolver(backend, system, cfg, 
										self.register, dt)

		self._idxcurr = self.register._stepper_regidx[0]

	@property
	def _idxprev(self):
		rprev = set(self.register._stepper_regidx[:2]) - {self._idxcurr}
		return next(iter(rprev))

class NewtonSolver(BaseNonLinearSolver):
	solver_name = 'newton'
	solver_nregs = 1

	def init_step(self, t):
		rhs = self.system.rhs
		rcurr, rprev = self._idxcurr, self._idxprev
		rprev_rhs = self.register._stage_regidx[0]

		rhs(t, rprev, rprev_rhs)

	def obtain_solution(self, bcoeffs):
		rcurr, rprev = self._idxcurr, self._idxprev

		regidxs = [rcurr] + [rprev]
		regidxs += self.register._stage_regidx
		coeffs = [0.0, 1.0] + bcoeffs

		self._addv(coeffs, regidxs)

	def store_current_solution(self):
		rcurr, rprev = self._idxcurr, self._idxprev
		self._add(0.0, rprev, 1.0, rcurr)

	def _errest(self, rcurr, rerr):
		comm, rank, root = get_comm_rank_root()

		# Get a set of kernels to estimate the integration error
		ekerns = self._get_reduction_kerns(rcurr, rerr, method='errest_imp',
										   norm=self._norm)

		# Bind the dynamic arguments
		for kern in ekerns:
			kern.bind(self._atol, self._rtol)

		# Run the kernels
		self.backend.run_kernels(ekerns, wait=True)

		# Pseudo L2 norm
		if self._norm == 'l2':
			# Reduce locally (element types + field variables)
			err = np.array([sum(v for k in ekerns for v in k.retval)])

			# Reduce globally (MPI ranks)
			comm.Allreduce(mpi.IN_PLACE, err, op=mpi.SUM)

			# Normalise
			err = np.sqrt(float(err) / self._get_gndofs())

		return err if not np.isnan(err) else 100

	def _update_rhs(self, tc, acoeffs, currstg):
		rhs = self.system.rhs
		gndofs = self._get_gndofs()
		rcurr, rprev = self._idxcurr, self._idxprev

		rcurr_rhs = self.register._stage_regidx[currstg]
		rdu0 = self.register._gmres_regidx[0]

		rhs(tc, rcurr, rcurr_rhs)

		consts = [0.0, *acoeffs, 1.0, -1.0]
		regidxs = [rdu0] + self.register._stage_regidx[:currstg+1]
		regidxs += [rprev, rcurr]

		self._addv(consts, regidxs)

		return self._eval_norm(rdu0)/np.sqrt(gndofs)

	def _init_stage(self, acoeffs, currstg):
		rcurr, rprev = self._idxcurr, self._idxprev

		consts = [0.0, 1.0, *acoeffs[:-1]]
		regidxs=  [rcurr, rprev] + self.register._stage_regidx[:currstg]
		self._addv(consts, regidxs)

	def solve(self, tc, acoeffs, currstg, nsteps):
		comm, rank, root = get_comm_rank_root()
		rcurr, rprev = self._idxcurr, self._idxprev

		# Begin nonlinear iteration count
		newton_iter = 0
		# Set the initial guess
		# self._init_stage(acoeffs, currstg)

		# Set the RHS
		nnorm = self._update_rhs(tc, acoeffs, currstg)
		nnorm_init = nnorm

		while nnorm / nnorm_init > self.ntol:

			# Linear Solve
			rdu = self.gmres_solver.solve(tc, acoeffs[-1], currstg,
								 			rcurr)

			# Add correction to current solution
			self._add(1.0, rcurr, 1.0, rdu)

			# Store RHS
			nnorm = self._update_rhs(tc, acoeffs, currstg)
			newton_iter	+= 1

			# Return if maxniters exceeded
			if newton_iter > self._max_newtoniters-1:

				# Reevaluate the jacobian if niters > maxniters
				self.gmres_solver._eval_jac(tc, acoeffs[-1], currstg, 
											              rcurr)

				if self.gmres_solver.clustering:
					self._cluster_jac = False

		if rank == root:
				print(f'stage is {currstg}', flush=True)
				print(f'nnorm is {nnorm /nnorm_init}, newton is {newton_iter}', flush=True)

		return rcurr, rprev, nnorm
class Register:
	def __init__(self, stage_nregs, stepper_nregs, niters, solver_nregs, cfg):
		self.stage_nregs = stage_nregs
		self.stepper_nregs = stepper_nregs
		self.solver_nregs = solver_nregs

		sect = 'solver-time-integrator'
		self.prec =  cfg.getbool(sect, 'precondition', False)

		self.aux_nregs = 1
		self.jacobi_nregs = 3
		self.niters = niters
		self.gmres_nregs = self.niters + 1
		self.nregs = (self.solver_nregs + self.stage_nregs + 
					  self.stepper_nregs + self.gmres_nregs
					  + self.aux_nregs + self.jacobi_nregs)
		
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
	def _duold_regidx(self):
		return (self.stepper_nregs + self.stage_nregs 
		  		+ self.gmres_nregs)

	@property
	def _err_regidx(self):
		return self._stepper_regidx[2]

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
		if self.prec:
			return self._regidx[ix:ix+self.gmres_nregs]
		else:
			return self._regidx[:self.gmres_nregs]