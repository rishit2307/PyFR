import numpy as np
from pyfr.integrators.base import BaseCommon
from pyfr.integrators.implicit.jacobi import BlockJacobi
from pyfr.mpiutil import get_comm_rank_root, mpi
import nvtx
class GMRESSolver(BaseCommon):
	def __init__(self, backend, system, cfg, register, tstart, dt):

		self.system = system
		self.backend = backend

		sect = 'solver-time-integrator'
		self.niters = cfg.getint(sect, 'gmres-niters', 10)
		self.ltol = cfg.getfloat(sect, 'gmres-tol', 1e-3)

		precision = cfg.get('backend', 'precision')
		if precision == 'double':
			self.epsmc = np.sqrt(np.finfo(np.float64).eps)
			self.fpdtype = np.float64
		else:
			self.epsmc = np.sqrt(np.finfo(np.float32).eps)
			self.fpdtype = np.float32
		
		# self.epsmc = np.sqrt(np.finfo(np.float64).eps)

		self.register = register
		self._dt = dt

		self.prec = cfg.get(sect, 'precondition', None)
		if self.prec in ['left', 'right']:
			self.jacobi_solver = BlockJacobi(system, backend, register, cfg ,self.epsmc)
			self.dtjac_start = tstart
			self.dtjac_out_init = -1
			self.dtjac_out = cfg.getfloat(sect, 'dtjac-out', np.inf)

			self.nsmooth = cfg.getint(sect, 'nsmooth', 1)
			self.jac_fpdtype = cfg.get(sect, 'jacobi-prec')

			if self.jac_fpdtype != self.fpdtype:
				self._prec_updated = False
			else:
				self._prec_updated = True
		else:
			self.prec = None
	
	def _init_solver(self, tc, a, currstg):
		self.iter = 0
		rcurr = self.register._curr_regidx
		self.e1 = np.zeros((self.niters+1), dtype=self.fpdtype)
		self.e1[0] = 1.0

		self.sn = np.zeros((self.niters), dtype=self.fpdtype)
		self.cs = np.zeros((self.niters), dtype=self.fpdtype)

		self.y = [[] for _ in range(len(self.system.ele_types))]

		self.rcurr_norm = self.eval_norm2(rcurr)
		self.tc, self.a = tc, a
		self.currstg = currstg
	@nvtx.annotate(color="red")
	def _eval_mat_vec(self, rdu, rmv):
		add, rhs = self._add, self.system.rhs
		epsmc = self.epsmc
		reg = self.register
		tc, a = self.tc, self.a
		currstg = self.currstg
		dtype = self.fpdtype

		rcurr = reg._curr_regidx
		rcurr_rhs = reg._stage_regidx[currstg]

		xabs = self.eval_norm2(rdu)
		# rcurr_norm = self.eval_norm1(rcurr)/self._get_gndofs()

		# if xabs > 1e-10:
		if xabs > 1e-4:
			eps = dtype(np.sqrt(1 + self.rcurr_norm)*epsmc/xabs)
		else:
			eps = dtype(np.sqrt(1 + self.rcurr_norm)*epsmc)
		# eps = self._eval_step_size(rcurr, rdu)


		# if xabs > 1e-5:
		# 	eps = epsmc*rcurr_norm/xabs + epsmc
		# else:
		# 	eps = epsmc*rcurr_norm

		fac = dtype(1.0/eps)

		add(0.0, rmv, 1.0, rcurr, eps, rdu)
		rhs(tc, rmv, rmv)

		add(-fac, rmv, fac, rcurr_rhs, 1.0/a, rdu)

	def _jacobi_prec(self, rin):
		if self.prec == None:
			return rin

		comm, rank, root=  get_comm_rank_root()

		r0, r1, r2 = self.register._jacobi_regidx
		self._add(0.0, r0, 0.0, rin)

		for i in range(self.nsmooth):
			self._eval_mat_vec(r0, r1)
			self._add(-1.0, r1, 1.0, rin)

			kerns = self.jacobi_solver.mul_jac(r1, r2)
			self.backend.run_kernels(kerns)

			self._add(1.0, r0, 1.0, r2)

		return r0

	def _arnoldi(self):
		comm, rank, root = get_comm_rank_root()

		rdu = self.register._gmres_regidx[self.iter]
		rmv = self.register._aux_regidx
		rkp1 = self.register._gmres_regidx[self.iter+1]
		rprec = self.register._prec_regidx[self.iter]

		h = np.zeros((self.iter+1), dtype=self.fpdtype)

		if self.prec == 'left':	
			self._eval_mat_vec(rdu, rmv)
			rmv = self._jacobi_prec(rmv)
		
		else:
			r0 = self._jacobi_prec(rdu)
			self._add(0.0, rprec, 1.0, r0)
			self._eval_mat_vec(r0, rmv)

		rji = self.register._gmres_regidx

		kerns = [self._get_dot_kerns(rji[j], rmv) for j in range(self.iter+1)]
		self.backend.run_kernels([krn for kern in kerns for krn in kern])
		self.backend.wait()

		for i, kern in enumerate(kerns):
			h[i] = sum([v.retval for v in kern])

		comm.Allreduce(mpi.IN_PLACE, h, op=mpi.SUM)

		self._addv([1.0] + list(-h), [rmv] + [rji[j] for j in range(self.iter+1)])
		qnorm = self.eval_norm2(rmv)
		self._add(0.0, rkp1, 1.0/qnorm, rmv)

		h = np.append(h, qnorm)

		return h
	@nvtx.annotate(color="green")
	def solve(self, tc, acoeff, currstg):
		self._init_solver(tc, acoeff, currstg)
		reg = self.register
		cs, sn = self.cs, self.sn
		jac_dtype = self.jac_fpdtype

		comm, rank, root = get_comm_rank_root()

		if self.prec and tc <= self._dt + self.dtjac_start:
			self.jacobi_solver._eval_jac(tc, acoeff, currstg)

			if rank == root:
				print(f'jacobian evaluated')

		elif self.prec and not self._prec_updated:
			self.jacobi_solver._update_precision()
			self._prec_updated = True
			if rank == root:
				print(f'jacobian precision updated')

		rdu, rduold = reg._gmres_regidx[self.iter], reg._duold_regidx
		rmv = reg._aux_regidx
		self._eval_mat_vec(rduold, rmv)
		self._add(1.0, rdu, -1.0, rmv)

		r0 = rdu if self.prec in ('right', None) else self._jacobi_prec(rdu)
		rnorm = self.eval_norm2(r0)
		self._add(0.0, rdu, 1/rnorm, r0)

		H = np.zeros((self.niters+1, self.niters), dtype=self.fpdtype)

		beta = rnorm*self.e1

		for k in range(self.niters):
			self.iter = k
			H[:k+2, k] = self._arnoldi()

			H[:k+2, k], cs[k], sn[k] = self._giv_rot(H[:k+2, k], 
													cs, sn ,k)

			beta[k+1] = -sn[k] * beta[k]
			beta[k] = cs[k] * beta[k]

			err = abs(beta[k+1])/abs(rnorm)

			if err < self.ltol:
				if rank == root:
					print(f'GMRES converged in {k} iterations, error is {err}')		
				break

		if k == self.niters-1 and err > self.ltol:
			if rank == root:
				print(f'GMRES did not converge in {self.niters} iterations, error is {err}')

		y =  np.linalg.solve(H[:k+1, :k+1], beta[:k+1])
		rdu = reg._gmres_regidx[self.iter]
		rduold = reg._duold_regidx

		consts = [0.0]+list(y)
		rp = self.register._prec_regidx[:self.iter+1]
		regidxs = [rdu] + [r for r in rp]

		self._addv(consts, regidxs)
		self._add(1.0, rdu, 1.0, rduold)

		return rdu

	def _giv_rot(self, h, cs, sn, k):
		for i in range(k):
			temp = cs[i] * h[i] + sn[i] * h[i+1]

			h[i+1] = -sn[i] * h[i] + cs[i] * h[i+1]
			h[i] = temp
		
		cs_k, sn_k = self.giv(h[k], h[k+1])

		h[k] = cs_k * h[k] + sn_k * h[k+1]
		h[k+1] = 0.0

		return h, cs_k, sn_k
	
	def giv(self, v1, v2):
		tt = np.sqrt(v1**2 + v2**2)
		cs = v1/tt
		sn = v2/tt

		return cs, sn






