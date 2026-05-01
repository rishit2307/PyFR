import numpy as np

from pyfr.integrators.base import BaseCommon
from pyfr.integrators.implicit.jacobi import BlockJacobi
from pyfr.mpiutil import get_comm_rank_root, mpi

class GMRESSolver(BaseCommon):
	def __init__(self, backend, system, cfg, register, dt):

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

		self.register = register
		self._dt = dt

		self.prec = cfg.getbool(sect, 'precondition', False)
		if self.prec:
			self.jacobi_solver = BlockJacobi(system, backend, register, cfg ,self.epsmc)

			self.nsmooth = cfg.getint(sect, 'nsmooth', 1)
			self.jac_fpdtype = cfg.get(sect, 'jacobi-prec')
			self.jac_evaluated  = False
			self.clustering = cfg.getbool(sect, 'clustering', False)
			self._cluster_jac = False if self.clustering else True

			if self.jac_fpdtype != precision:
				self._prec_updated = False
			else:
				self._prec_updated = True
			self.mul_jac = self.jacobi_solver.mul_jac
			

		else:
			self.prec = None

		# Get system variables
		gndofs = self._get_gndofs()
		ndims = self.system.ndims
		privarmap = self.system.elementscls.privarmap
		pri_to_con = self.system.elementscls.pri_to_con

		# Get scales
		hasv = lambda v:cfg.hasopt(sect, f'typ-{v}')
		getv = lambda v:cfg.getfloat(sect, f'typ-{v}')

		check = all([hasv(p) for p in privarmap[ndims]])
		self._scales, self._invscales = (), ()

		if check:

			pvars = [getv(p) for p in privarmap[ndims]]
			convars = np.array(pri_to_con(pvars, cfg))
	
			# Commit scales to backend
			self._scales = np.sqrt(gndofs)*convars
			self._invscales = tuple(1/self._scales)
			self._scales=  tuple(self._scales)
			print(self._invscales, self._scales)

		# Allocate Storage for GMRES 
		self._e1 = np.empty((self.niters+1))
		self._sn = np.empty((self.niters))
		self._cs = np.empty((self.niters))

		self._H = np.empty((self.niters+1, self.niters))

	def _init_solver(self, tc, a, currstg, rcurr):
		self.iter = 0
		self._rcurr = rcurr

		self._e1.fill(0)
		self._e1[0] = 1
		self._sn.fill(0)
		self._cs.fill(0)
		self._H.fill(0)

		self.rcurr_norm = self._eval_norm(rcurr)
		self.tc, self.a = tc, a
		self.currstg = currstg

	def _eval_mat_vec(self, rdu, rmv):
		add, rhs = self._add, self.system.rhs
		epsmc = self.epsmc
		reg = self.register
		tc, a = self.tc, self.a
		currstg = self.currstg
		dtype = self.fpdtype
		rcurr = self._rcurr
		comm, rank, root = get_comm_rank_root()

		rcurr_rhs = reg._stage_regidx[currstg]

		xabs = self._eval_norm(rdu)

		if xabs > 1e-4:
			eps = np.sqrt(1 + self.rcurr_norm)*epsmc/xabs
		else:
			eps = np.sqrt(1 + self.rcurr_norm)*epsmc
		# eps = epsmc
		# utyp = self.utyp
		# dottyp = self._eval_dot([utyp, rdu], scaling=True)
		# dotcurr = abs(self._eval_dot([rcurr, rdu]))

		# rdunrm = self._eval_norm(rdu)**2
		# eps = max(dottyp, dotcurr)*np.sign(dotcurr)*epsmc/rdunrm

		fac = dtype(a/eps)

		_scales, _invscales = self._scales, self._invscales

		add(0.0, rmv, 1.0, rcurr, eps, rdu, 
	        inscales=_scales, in_idx=(2,))

		rhs(tc, rmv, rmv)

		add(-fac, rmv, fac, rcurr_rhs, 1, rdu,
	        inscales=_scales, in_idx=(2,), 
			outscales=_invscales)

	def _jacobi_prec(self, rin):
		if self.prec == None:
			return rin

		r0, r1, r2 = self.register._jacobi_regidx
		kerns = self.mul_jac(rin, r0, inscales=self._scales, 
					         outscales=self._invscales)
		self.backend.run_kernels(kerns)

		return r0

	def _arnoldi(self):
		comm, rank, root = get_comm_rank_root()

		rdu = self.register._gmres_regidx[self.iter]
		rmv = self.register._aux_regidx
		rkp1 = self.register._gmres_regidx[self.iter+1]
		rprec = self.register._prec_regidx[self.iter]
		H = self._H

		r0 = self._jacobi_prec(rdu)
		if self.prec:
			self._add(0, rprec, 1, r0)
		self._eval_mat_vec(r0, rmv)

		rji = self.register._gmres_regidx

		for j in range(self.iter+1):

			H[j:j+1, self.iter] = h = self._eval_dot([rmv, rji[j]])
			self._addv([1, -h], [rmv, rji[j]])

		# dotregs = [rmv] + [rji[j] for j in range(self.iter+1)]
		# H[:self.iter+1, self.iter] = h = self._eval_dot(dotregs)

		# self._addv([1] + list(-h), [rmv] + 
		# 	                  [rji[j] for j in range(self.iter+1)])

		qnorm = self._eval_norm(rmv)
		self._add(0, rkp1, 1/qnorm, rmv)

		H[self.iter+1, self.iter] = qnorm

	def solve(self, tc, acoeff, currstg, rcurr):
		self._init_solver(tc, acoeff, currstg, rcurr)
		H = self._H
		reg = self.register
		cs, sn = self._cs, self._sn

		comm, rank, root = get_comm_rank_root()

		# Evaluate the Jacobians
		if self.prec and not self.jac_evaluated:
			self.jacobi_solver._eval_jac(tc, acoeff, currstg, rcurr)

			self.jac_evaluated = True
			if rank == root:
				print(f'jacobian evaluated', flush=True)

			if not self._cluster_jac:

				self._cluster_jac = True
				self.mul_jac = self.jacobi_solver.mul_jac_kmeans
				print(f'rank is {rank}, kmeans done')
				self._prec_updated = True

			elif not self._prec_updated:
				self.jacobi_solver._update_precision()
				self._prec_updated = True
				if rank == root:
					print(f'jacobian precision updated', flush=True)


		rdu, rduold = reg._gmres_regidx[self.iter], reg._duold_regidx
		rmv = reg._aux_regidx
		# self._eval_mat_vec(rduold, rmv)
		# self._add(1.0, rdu, -1.0, rmv)
		self._add(0, rmv, 1, rdu, 
			      inscales=self._invscales, in_idx=(1,))

		# Right or Left Preconditioning
		rnorm = self._eval_norm(rmv)

		self._add(0, rdu, 1/rnorm, rmv)

		beta = rnorm*self._e1

		for k in range(self.niters):
			self.iter = k
			
			# Arnoldi
			self._arnoldi()
			
			# Givens Rotation
			self._giv_rot(k)

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

		# Solve the least squares system
		y =  np.linalg.solve(H[:k+1, :k+1], beta[:k+1])
		rdu = reg._gmres_regidx[self.iter]

		consts = [0]+list(y)
		rp = self.register._prec_regidx[:self.iter+1]
		regidxs = [rdu] + [r for r in rp]

		self._addv(consts, regidxs, outscales=self._scales)
		# self._add(1.0, rdu, 1.0, rduold)

		return rdu, self.iter

	def _giv_rot(self, k):
		H, cs, sn = self._H, self._cs, self._sn

		for i in range(k):
			temp = cs[i]*H[i, k] + sn[i]*H[i+1, k]
			H[i+1, k] = -sn[i]*H[i, k] + cs[i]*H[i+1, k]
			H[i, k] = temp

		cs[k], sn[k] = self.giv(H[k, k], H[k+1, k])
		H[k, k] = cs[k]*H[k, k] + sn[k]*H[k+1, k]
		H[k+1, k] = 0

	def giv(self, v1, v2):
		tt = np.sqrt(v1**2 + v2**2)
		cs = v1/tt
		sn = v2/tt

		return cs, sn