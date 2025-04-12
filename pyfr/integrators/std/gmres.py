import numpy as np
from collections import defaultdict
import copy
import itertools as it
import time
import math
from pyfr.inifile import Inifile
from pyfr.integrators.std.base import BaseStdIntegrator
from pyfr.integrators.std.controllers import BaseStdController
from pyfr.integrators.std.steppers import BaseStdStepper
from pyfr.util import memoize, subclass_where
from pyfr.mpiutil import get_comm_rank_root, mpi
class GMRESmultip(BaseStdIntegrator):
	def __init__(self, backend, systemcls, rallocs, mesh, initsoln, cfg):
		sect = 'solver-time-integrator'
		
		# Get the multigrid cycle
		self.cycle, self.csteps = zip(*cfg.getliteral(sect, 'cycle'))

		self.levels = sorted(set(self.cycle), reverse=True)

		self.backend = backend

		cn = cfg.get(sect, 'controller')
		pn = cfg.get(sect, 'scheme')
		cc = subclass_where(BaseStdController,
							controller_name=cn)
		pc = subclass_where(BaseStdStepper, 
							stepper_name=pn)
		
		bases = [cc, pc]

		self._order = order = self.level = cfg.getint('solver', 'order')
		
		self.levels = sorted(set(self.cycle), reverse=True)
		self.mpniters = cfg.getint(sect, 'mpniters', 1)
		self.pintgs = {}

		for l in self.levels:

			if l == order:
				mcfg = cfg
			else:
				mcfg = Inifile(cfg.tostr())
				mcfg.set('solver', 'order', l)
				mcfg.set('solver-time-integrator', 'gmres-iter', 0)
				tau = mcfg.getfloat('solver-time-integrator', 'tau')
			
			class lpsint(*bases):
				name = 'GMRES-multip'
				pseudo_nregs = 3
				eval_nreg = pseudo_nregs + 1
				eval_src = 0 if l == self._order else 1
				uru_nreg = 2 if l == self._order else 0
				duold_nreg = 1 if l == self._order else 0
				aux_gmres = 1 if l == self._order else 0

				def _eval_jac(self):
					add, rhs = self._add, self.system.rhs
					rup, rrup = self._up_rup_regidx
					r0, r1, *r = self._pseudo_regidx

					t, dt, dtfac = self.t, self.dt, self.dtfac
					self.jac = jac = {}
					self.jacinv = jacinv = {}
					self.jacshuff = {}
					self.P = {}
					comm, rank, root = get_comm_rank_root()

					for etp in self.system.ele_types:

						i = self.system.ele_types.index(etp)
						nupts = self.system.ele_shapes[i][0]
						neles = self.system.ele_shapes[i][-1]
						nvars = self.system.ele_shapes[i][1]

						# self.jac[etp] = jac[etp] = np.random.rand(nupts*nvars, nupts, nvars, neles)


						self.jac[etp] = jac[etp] = np.empty((nupts*nvars, nupts, nvars, neles))
						self.P[etp] = np.zeros((neles, nupts*nvars))
						self.jacinv[etp] = jacinv[etp] = np.zeros((neles, (nupts*nvars)**2))
						self.jacinv[etp] = backend.matrix(jacinv[etp].shape, jacinv[etp])
						self.jac[etp] = backend.matrix(jac[etp].shape, jac[etp])

						self.P[etp] = backend.matrix(self.P[etp].shape, self.P[etp], dtype=self.backend.ixdtype)

					for etp in sorted(self.system.ele_types):
						celes = self.system.celes[etp]
						for col in range(self.system.ncolours[etp]):
							for v in range(self.system.nvars):
								for npt in range(nupts):

									self._addid(celes, [rup, r0], npt, v, col)
									self.backend.wait()
									rhs(t+dt, r0, r1)
									self.backend.wait()

									self._add(-1.0/1e-8, r1, 1.0/1e-8, rrup)
									kerns = self._init_jac(r1, etp, celes)
									self.bind_kerns(kerns, npt, v, col, dtfac/dt)
									backend.run_kernels(kerns)
									self.backend.wait()

					print(f'jacinit is done mpi , rank is {rank}')
					for etp in self.system.ele_types:

						kern = backend.kernel('getf3', *[jac[etp], jacinv[etp], self.P[etp]])

						backend.run_kernels([kern])	

						shufkerns = self._shuff_jac(etp)
						backend.run_kernels(shufkerns)

				def bind_kerns(self, kerns, *args):
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
				def _mul_jac(self, *rs):
					kern = []
					em = self.system.ele_banks
					for k, jac in self.jac.items():
						i = self.system.ele_types.index(k)
						kern.append(self.backend.kernel('jacmul', 
									  *[em[i][r] for r in rs] + [jac]))
					
					return kern

				def _verify_jac(self):
					pass
				def richardson(self, nsmooth):
					add = self._add
					tau = mcfg.getfloat('solver-time-integrator', 'tau')

					r0, r1, r2 = self._pseudo_regidx
					rmv, rsrc = self._mvec_regidx, self._src_regidx
					
					for j in range(nsmooth):
						# rmv = A*r0
						self._eval_mat_vec(r0, rmv)

						# rmv = b - A*r0
						add(-1.0, rmv, 1.0, rsrc)

						# r0 = x + tau(b - A*r1)
						add(1.0, r0, tau, rmv)

				def jacobi(self, nsmooth, hclass=None,r=None,p=None):
					r0, r1, *r = self._pseudo_regidx
					rmv = self._mvec_regidx
					rsrc = self._src_regidx
					add = self._add
					for _ in range(nsmooth):

						self._eval_mat_vec(r0, rmv)

						# rmv = b - Axi
						add(-1.0, rmv, 1.0, rsrc)

						# r1 = J^-1*(b - Axi)

						kerns = self._mul_jac(rmv, r1)
						backend.run_kernels(kerns)

						# r0 = r0 + w*r1
						add(1.0, r0, 2/3, r1)

				def jac_mult(self, nsmooth, f=None, hclass=None, r=None, p=None):
					if f:
						self.jacobi(nsmooth, hclass=hclass, p=p, r=r)
					else:
						self.richardson(nsmooth)

			self.pintgs[l] = lpsint(backend, systemcls, rallocs, 
									mesh, initsoln, mcfg)
		
		self.system = self.pintgs[order].system	
		self._init_projmats()
	
	def plugins(self):
		return self.pintgs[self._order].plugins

	@property
	def tlist(self):
		return self.pintg.tlist
	
	@property
	def tstart(self):
		return self.pintg.tstart
	
	@property
	def tend(self):
		return self.pintg.tend
	
	@property
	def tcurr(self):
		return self.pintg.tcurr
	
	@property
	def pintg(self):
		return self.pintgs[self.level]
	
	def _init_projmats(self):
		self.projmats = defaultdict(list)
		cmat = lambda m: self.backend.const_matrix(m, tags={'align'}) 

		for l in self.levels[1:]:
			for i in range(len(self.pintg.system.ele_types)):
				b1 = self.pintgs[l].system.ubasis[i]
				b2 = self.pintgs[l + 1].system.ubasis[i]

				self.projmats[l, l + 1].append(cmat(b1.proj_to(b2)))
				self.projmats[l + 1, l].append(cmat(b2.proj_to(b1)))
		
		for l in self.levels:
			for i in range(len(self.system.ele_types)):
				b1 = self.pintgs[self._order].system.ubasis[i]
				b2 = self.pintgs[l].system.ubasis[i]
				if not self.projmats[self._order, l]:
					self.projmats[self._order, l].append(cmat(b1.proj_to(b2)))
					self.projmats[l, self._order].append(cmat(b2.proj_to(b1)))

	def _init_loworder(self, l1, l2):

		r1up, rr1up = self.pintgs[l1]._up_rup_regidx
		r2up, rr2up = self.pintgs[l2]._up_rup_regidx

		self.backend.run_kernels(
		self.mgproject(l1, r1up ,l2, r2up)
		)
		t, dt = self.pintg.t, self.pintg.dt

		# rr2up = R(Un+1, k)
		rhs = self.pintgs[l2].system.rhs
		rhs(t+dt, r2up, rr2up)

		comm, rank, root = get_comm_rank_root()
		kerns = self.pintgs[l2]._get_norm2_kerns(r2up)
		self.backend.run_kernels(kerns, wait=True)
		Un = np.array([sum(kern.retval**2 for kern in kerns)])

		comm.Allreduce(mpi.IN_PLACE, Un, op=mpi.SUM)
		self.pintgs[l2].Un = np.sqrt(float(Un))

		for i in range(len(self.system.ele_types)):
			nuptl2, nvarl2, nelel2 = self.pintgs[l2].system.ele_shapes[i]
			etp = self.system.ele_types[i]
			self.pintgs[l2].jac = jac = defaultdict(list)
			for e in range(nelel2):
				jac[etp, e] = self.projmats[l1, l2] @ self.pintgs[l1].jac[etp, e]

		self.pintgs[l2].nfeval += 1
		
	@memoize
	def mgproject(self, l1, l1reg, l2, l2reg):
		projk = []
		for i, a in enumerate(self.projmats[l1, l2]):
			b = self.pintgs[l1].system.ele_banks[i][l1reg]
			c = self.pintgs[l2].system.ele_banks[i][l2reg]
			projk.append(self.backend.kernel('mul', a, b, out=c))

		return projk

	def restrict(self, l1, l2):
		# r0, r1, r2, r3, r4, *r5 = self.pintgs[l1]._regidx
		# r5 = r5[0]
		self.level = l1
		rmv = self.pintg._mvec_regidx
		r0, *r = self.pintg._pseudo_regidx
		rsl1 = self.pintg._src_regidx
		
		# rmv = A*r0
		self.pintg._eval_mat_vec(r0, rmv)
		# rmv = b - A*r0
		self.pintg._add(-1.0, rmv, 1.0, rsl1)

		self.level = l2
		rsl2 = self.pintg._src_regidx

		self.backend.run_kernels(self.mgproject(l1, rmv, l2, rsl2))

	def prolongate(self, l1, l2):

		r0, r1, r2 = self.pintgs[l1]._pseudo_regidx
		rl0, rl1, rl2 = self.pintgs[l2]._pseudo_regidx
		self.backend.run_kernels(self.mgproject(l1, r0, l2, rl1))

		self.pintgs[l2]._add(1.0, rl0, 1.0, rl1)


	def solve_gmres(self):
		comm, rank, root = get_comm_rank_root()
		self.level = self._order
		y = self.pintg.y
		cs, sn = self.pintg.cs, self.pintg.sn
		ltol = self.pintg.ltol
		rnorm, m = self.pintg.rnorm, self.pintg.m
		add = self.pintg._add

		r0 = self.pintg._du_regidx

		rnorm = self.pintg.eval_norm2(r0)
		
		add(0.0, r0, 1/rnorm, r0)


		H = np.zeros((m+1, m))
	
		beta = rnorm*self.pintg.e1	
		for k in range(m):
			self.pintg.k = k
			H[:k+2, k] = self.arnoldi(k)


			H[:k+2, k], cs[k], sn[k] = self.giv_rot(H[:k+2, k], 
													cs, sn ,k)

			beta[k+1] = -sn[k] * beta[k]
			beta[k] = cs[k] * beta[k]

			err = abs(beta[k+1])/abs(rnorm)

			if err < ltol:
				if rank == root:
					print(f'GMRES converged in {k} iterations, error is {err}')
					
				break

		if k == m-1 and err > ltol:
			if rank == root:
				print(f'GMRES did not converge in {m} iterations, error is {err}')

		y =  np.linalg.solve(H[:k+1, :k+1], beta[:k+1])


		rdu, rj = self.pintg._du_regidx, self._gmres_j_regidx
		
		consts = [0.0]+list(y)
		# regidxs = [rdu] + [rj(j) for j in range(k+1)]
		rp = self.pintg._prec_regs
		regidxs = [rdu] + [r for r in rp]
		self._addv(consts, regidxs)


		# self.mg_vcycle()
		
		# err = self.pintg._res()
		self._add(1.0, rdu, 1.0, self.pintg._duold_regidx)

		# if rank == root:
		# 	print(f'GMRES error beta is {beta[k+1]}')
		# 	# print(f'GMRES error rduolnorm is {rduoldnorm}')
		# 	print(f'actual error is {err}')

	def mg_vcycle(self):
		if not self.mpniters:
			return
		
		cycle, csteps = self.cycle, self.csteps

		self.level = self._order
		r0, *r = self.pintg._pseudo_regidx
		self.pintg._add(0.0, r0, 0.0, self.pintg._du_regidx)

		for i in range(self.mpniters):
			for l in self.levels[1:]:
				self.level = l
				r0 = self.pintg._pseudo_regidx[0]
				self.pintg._add(0.0, r0, 0.0, self.pintg._mvec_regidx)
			
			for l, m, n in it.zip_longest(cycle, cycle[1:], csteps):
				self.level = l
				self.pintg.jac_mult(n, f='jacobi')

				if m is not None and l > m:
					self.restrict(l, m)
				elif m is not None and l < m:
					self.prolongate(l, m)
			
			self.level = self._order

	def arnoldi(self, k):
		self.level = self._order
		add = self.pintg._add

		h = np.zeros(k+1)
		comm, rank, root = get_comm_rank_root()
		rkp1 = self.pintg._gmres_j_regidx(k+1)
		rdu = self.pintg._du_regidx
		rmv = self.pintg._mvec_regidx
		r0 = self.pintg._pseudo_regidx[0] if self.mpniters else rdu

		self.mg_vcycle()
		rprec = self.pintg._prec_regidx
		add(0.0, rprec, 1.0, r0)
		self.pintg._eval_mat_vec(r0, rmv)
		rji = self.pintg._gmres_j_regidx

		kerns = [self._get_dot_kerns(rji(j), rmv) for j in range(k+1)]
		self.backend.run_kernels([krn for kern in kerns for krn in kern])

		self.backend.wait()
		try:
			del self.pintgs[self._order].jacinv
		except AttributeError:
			pass

		for i, kern in enumerate(kerns):
			h[i] = sum([v.retval for v in kern])

		comm.Allreduce(mpi.IN_PLACE, h, op=mpi.SUM)
		self._addv([1.0] + list(-h), [rmv] + [rji(j) for j in range(k+1)])
		qnorm = self.pintg.eval_norm2(rmv)

		h = np.append(h, qnorm)
		add(0.0, rkp1, 1.0/qnorm, rmv)

		return h

	def giv_rot(self, h, cs, sn, k):
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
	
	def advance_to(self, t):
		
		while self.tcurr < t:
			self.level = self._order

			# Decide on the time step
			dt = max(min(t - self.pintg.tcurr, self.pintg._dt), self.pintg.dtmin)

			add = self._add
			nnorm = np.inf
			s = 1.0
			ntol = self.ntol =1e-2
			comm, rank, root = get_comm_rank_root()

			nonlin_iter = 0
			nsteps = int(self.tcurr // dt)

			while nnorm > ntol:

				for l in self.levels:
					self.level = l
					self.pintg._init_step(self.tcurr, dt)
				
				self.level = self._order
				if rank == root:
					print(f't is {self.tcurr}, dt is {dt}')

				self.pintg._init_gmres()

				self.pintg._res(ev_rru=True)
				if (self.pintg.nacptsteps == 0) and nonlin_iter==0:
					self.pintg._eval_jac()
					print('Jacobian evaluated')

				# import pdb;pdb.set_trace()
				# for l, m in it.zip_longest(self.levels, self.levels[1:]):
				# 	if m is not None:
				# 		self._init_loworder(l, m)

				self.solve_gmres() 

				# rU = Un+1,k+1 = Un+1,k + s*dUk
				rUp, rrUp = self.pintgs[self._order]._up_rup_regidx
				rdU = self.pintgs[self._order]._du_regidx
				add(1.0, rUp, s, rdU)
				add(0.0, self.pintg._duold_regidx, 1.0, self.pintg._du_regidx)

				nnorm = self.pintg.newton_res()
				nonlin_iter += 1
				if rank == root:
					print(nnorm)
					print(f'Newton iteration is {nonlin_iter}')

			rU, rrU = self.pintgs[self._order]._u_ru_regidx
			# r0 = Un+1 = r2

			add(0.0, rU, 1.0, rUp)

			
			for l in self.levels:
				if rank == root:
					print(f'nfeval at {l} is {self.pintgs[l].nfeval}')

			idxcurr = rU
			self.pintg._accept_step(dt, idxcurr)
