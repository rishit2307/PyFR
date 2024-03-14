import numpy as np
from collections import defaultdict
import copy
import itertools as it

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
			
			class lpsint(*bases):
				name = 'GMRES-multip'

				def _eval_jac(self):
					add, rhs = self._add, self.system.rhs
					comm, rank, root = get_comm_rank_root()
					r0, r1, r2, r3, *r4 = self._regidx
					r4, r5 = r4[0], r4[1]
					t, dt, dtfac = self.t, self.dt, self.dtfac


					self.jac = jac = defaultdict(list)
					rhs(t+dt, r2, r1)
						
					

					for col, etp in sorted(self.system.celes.keys()):
						i = self.system.ele_types.index(etp)
						ur0 = self.system.ele_banks[i][r2].get()
						dr1 = self.system.ele_banks[i][r1].get()

						for j in range(len(self.system.ele_types)):
							u = self.system.ele_scal_upts(r2)[j]
							self.system.ele_banks[j][r5].set(u)

						nupts = self.system.ele_shapes[i][0]
						for v in range(self.system.nvars):
							for npt in range(nupts):
								eidx = self.system.celes[col, etp]
								
								eps = np.zeros_like(ur0)

								eps[npt, v, eidx] = 1e-8

								ur = ur0+eps

								self.system.ele_banks[i][r5].set(ur)

								rhs(t+dt, r5, r4)
								self.backend.wait()
						
								dr2 = self.system.ele_banks[i][r4].get()

								dr = (dr1 - dr2)/1e-8

								for e in eidx:
									jac[etp, e].append(dr[..., e].T.reshape(-1))
					cond = []
					for i, eshape in enumerate(self.system.ele_shapes):
						nele = eshape[-1]
						etp = self.system.ele_types[i]
						for e in range(nele):
							shape = len(jac[etp, e])

							jac[etp, e] = np.array(jac[etp, e]).T + (dtfac/dt)*np.eye(shape)
							cond.append(np.linalg.cond(jac[etp, e]))

							jac[etp, e] = np.linalg.inv(jac[etp, e])

					cond = comm.allreduce(cond, op=mpi.MAX)

					if rank == root:
						print(f'rank is {rank} cond number is {np.amax(cond)}')
				
				def richardson(self, nsmooth):
					r0, r1, r2, r3, r4, *r5 =  self._regidx
					r5, r6, r7, r8 = r5[0], r5[1], r5[2], r5[3]
					add = self._add
					tau = self.tau
					for _ in range(nsmooth):
						# r5 = A*r1
						self._eval_mat_vec(r2, r1, r5, r4)

						# r5 = b - A*r1
						add(-1.0, r5, 1.0, r3)

						# r1 = x + tau(b - A*r1)
						add(1.0, r1, tau, r5)

					# for _ in range(nsmooth):
					# 	## First stage
					# 	# r5 = A*r1
					# 	self._eval_mat_vec(r2, r1, r5, r4)
					# 	# r5 = b - A*r1
					# 	add(-1.0, r5, 1.0, r3)
						
					# 	## Second stage
					# 	# r6 = r5*dtau/2 + r1
					# 	add(0.0, r6, 1.0, r1, tau/2.0, r5)
					# 	# r7 = A*r6
					# 	self._eval_mat_vec(r2, r6, r7, r4)
					# 	# r7 = b - A*r6
					# 	add(-1.0, r7, 1.0, r3)
						
					# 	## Accumulate
					# 	# r5 = r1 + dtau/6(r5  + 2*r7)
					# 	add(tau/6.0, r5, 1.0, r1, tau/3.0, r7)
						
					# 	## Third Stage
					# 	# r6 = r7*dtau/2 + r1
					# 	add(0.0, r6, tau/2, r7, 1.0, r1)
					# 	# r7 = A*r6
					# 	self._eval_mat_vec(r2, r6, r7, r4)
					# 	# r7 = b - A*r6
					# 	add(-1.0, r7, 1.0, r3)

					# 	## Accumulate
					# 	# r5 = r5 + dtau*r7/3
					# 	add(1.0, r5, tau/3, r7)

					# 	# r6 = dtau*r7 + r1
					# 	add(0.0, r6, tau, r7, 1.0, r1)
					# 	# r7 = A*r6
					# 	self._eval_mat_vec(r2, r6, r7, r4)
					# 	# r7 = b - A*r7
					# 	add(-1.0, r7, 1.0, r3)

					# 	# r5 = r5 + dtau*r7/6
					# 	add(1.0, r5, tau/6, r7)

					# 	# r1 = r5
					# 	add(0.0, r1, 1.0, r5)


				def jacobi(self, nsmooth, hclass=None,r=None,p=None):
					r0, r1, r2, r3, r4, *r5 =  self._regidx
					r5 = r5[0]
					
					xi = [self.system.ele_banks[i][r1].get()
							for i in range(len(self.system.ele_types))]
					jac = self.jac
					for _ in range(nsmooth):

						self._eval_mat_vec(r2, r1, r5, r4)
						
						Axi = [self.system.ele_scal_upts(r5)[i]
							   for i in range(len(self.system.ele_types))]

						for i in range(len(self.system.ele_types)):
							etp = self.system.ele_types[i]
							nupts = self.system.ele_shapes[i][0]
							# nhpt = hclass.system.ele_shapes[i][0] 
							neles = self.system.ele_shapes[i][-1]
							nvars = self.system.nvars
							b = self.system.ele_scal_upts(r3)[i]
							
							# pr = p[i].get()
							# re = r[i].get()

							# ax = pr @ Axi[i].reshape(nupts, -1)
							# ax = ax.reshape(nhpt, nvars, neles)
							# b = pr @ b[i].reshape(nupts, -1)
							# b = b.reshape(nhpt, nvars, neles)

							
							for e in range(neles):
								

								tmp = jac[etp, e] @ Axi[i][..., e].T.reshape(-1)
								xi[i][..., e] -= (2/3) * tmp.reshape(nvars, nupts).T

								tmp2 = jac[etp, e] @ b[..., e].T.reshape(-1)
								xi[i][... ,e] += (2/3) * tmp2.reshape(nvars, nupts).T 
							self.system.ele_banks[i][r1].set(xi[i])

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
		rl0, rl1, rl2, rl3, *rl4 = self.pintgs[l2]._regidx
		rl4 = rl4[0]
		
		r0, r1, r2, r3, *r4 = self.pintgs[l1]._regidx
		self.backend.run_kernels(self.mgproject(l1, r2 ,l2, rl2))


		t, dt = self.pintg.t, self.pintg.dt

		# r4 = R(Un+1, k)
		rhs = self.pintgs[l2].system.rhs
		rhs(t+dt, rl2, rl4)
		self.pintgs[l2].nfeval += 1
		
	
	def mgproject(self, l1, l1reg, l2, l2reg):
		projk = []
		for i, a in enumerate(self.projmats[l1, l2]):
			b = self.pintgs[l1].system.ele_banks[i][l1reg]
			c = self.pintgs[l2].system.ele_banks[i][l2reg]
			projk.append(self.backend.kernel('mul', a, b, out=c))

		return projk

	def restrict(self, l1, l2):
		r0, r1, r2, r3, r4, *r5 = self.pintgs[l1]._regidx
		r5 = r5[0]
		rl0, rl1, rl2, rl3, rl4, *rl5 = self.pintgs[l2]._regidx
		rl5, rl8, rl7= rl5[0], rl5[-1], rl5[-2]
		add = self.pintgs[l1]._add

		# r5 = A*r1
		self.pintgs[l1]._eval_mat_vec(r2, r1, r5, r4)
		# r5 = b - A*r1
		add(-1.0, r5, 1.0, r3)

		self.backend.run_kernels(self.mgproject(l1, r5, l2, rl3))
		# self.backend.run_kernels(self.mgproject(l1, r1, l2, rl8))

		# self.level = l2
		# add = self.pintg._add

		# # rl7 = A*rl8
		# self.pintg._eval_mat_vec(rl2, rl8, rl7, rl4)

		# # rl3 = rl3 + rl7
		# add(1.0, rl3, 1.0, rl7)

	def prolongate(self, l1, l2):
		r0, r1, *r2 = self.pintgs[l1]._regidx
		r8 = r2[-1]
		rf0, rf1, *rf5 = self.pintgs[l2]._regidx
		rf8 = rf5[-1]
		add = self.pintgs[l1]._add

		# # r8 = e = y^s - y^ns
		# add(-1.0, r8, 1.0, r1)
		self.backend.run_kernels(self.mgproject(l1, r1, l2, rf8))

		add = self.pintgs[l2]._add
		# r1 = ys + e
		add(1.0, rf1, 1.0, rf8)

	def solve_gmres(self, x):
		comm, rank, root = get_comm_rank_root()
		self.level = self._order
		y = self.pintg.y
		cs, sn = self.pintg.cs, self.pintg.sn
		ltol = self.pintg.ltol
		rnorm, m = self.pintg.rnorm, self.pintg.m
		add = self.pintg._add

		r0, r1, r2, r3, *r4 = self.pintg._regidx

		# for i, etype in enumerate(self.system.ele_types):
		#         rnorm[etype] = (np.linalg.norm(self.system.ele_banks[i][r1].get()))**2
			
		# for etype in eletype:
		#         rnorm[etype] = np.sqrt(comm.allreduce(rnorm[etype], op=mpi.SUM))

		rnorm = sum([np.linalg.norm(self.system.ele_banks[i][r1].get())**2
							  for i in range(len(self.system.ele_types))])
		rnorm = np.sqrt(comm.allreduce(rnorm, op=mpi.SUM))

		Q = [[] for i in range(len(self.system.ele_types))]
			
		for i, etype in enumerate(self.system.ele_types):
			nupts, nvars, neles = self.system.ele_shapes[i]
			Q[i] = np.zeros((nupts, nvars, neles, m+1))
			Q[i][..., 0] = self.system.ele_banks[i][r1].get() / rnorm

		H = np.zeros((m+1, m))
		beta = rnorm*self.pintg.e1

		for k in range(m):
			H[:k+2, k], q = self.arnoldi(Q, k)

			for i in range(len(self.system.ele_types)):
				Q[i][..., k+1] = q[i]

			H[:k+2, k], cs[k], sn[k] = self.giv_rot(H[:k+2, k], 
													cs, sn ,k)

			beta[k+1] = -sn[k] * beta[k]
			beta[k] = cs[k] * beta[k]

			err = abs(beta[k+1]) / rnorm

			if err < ltol:
				if rank == root:
					print(f'GMRES converged in {k} iterations, error is {err}')
					
				break

		if k == m-1 and err > ltol:
			if rank == root:
				print(f'GMRES did not converge in {m} iterations, error is {err}')
		
		y =  np.linalg.solve(H[:k+1, :k+1], beta[:k+1])
		# print(f'eigvals are {np.amax(np.abs(np.linalg.eigvals(H[:m, :m])))}')
		netype = len(self.system.ele_types)
		cycle, csteps = self.cycle, self.csteps

		for i in range(len(self.system.ele_types)):
			self.system.ele_banks[i][r3].set(Q[i][..., :k+1] @ y)

		niters = self.mpniters
		if niters:
			
			x0 = [np.zeros_like(self.pintgs[self._order].system.ele_scal_upts(r2)[i])
							for i in range(netype)]
			for i in range(netype):
						self.pintgs[self._order].system.ele_banks[i][r1].set(x0[i])
			

			for _ in range(niters):
				for l in self.levels[1:]:
					self.level = l
					x0 = [np.zeros_like(self.pintg.system.ele_scal_upts(r2)[i])
							for i in range(netype)]

					for i in range(netype):
						self.pintg.system.ele_banks[i][r1].set(x0[i])

				for l, m, n in it.zip_longest(cycle, cycle[1:], csteps):
					self.level = l

					pr1 = self.projmats[l, self._order]
					pr2 = self.projmats[self._order, l]
					# self.pintg.jac_mult(n, hclass=self.pintgs[self._order], p=pr1, r=pr2)
					# self.pintg.jac_mult(n, f='jacobi', hclass=self.pintgs[self._order], p=pr1, r=pr2)
					if l == self._order:
						self.pintg.jac_mult(n)
					else:
						self.pintg.jac_mult(n, f='jacobi')

					if m is not None and l > m:
						self.restrict(l, m)
					elif m is not None and l < m:
						self.prolongate(l, m)

			add(0.0, r3, 1.0, r1)

		# import pdb;pdb.set_trace()
		for i in range(len(self.system.ele_types)):
			x[i] += self.system.ele_banks[i][r3].get()
			self.system.ele_banks[i][r3].set(x[i])

	def arnoldi(self, Q, k):
		self.level = self._order
		add = self.pintg._add

		h = np.zeros(k+2)
		netype = len(self.system.ele_types)

		cycle, csteps = self.cycle, self.csteps
		


		r0, r1, r2, r3, *r4 = self.pintg._regidx
		r4 = r4[0]

		comm, rank, root = get_comm_rank_root()

		for i in range(netype):
			self.system.ele_banks[i][r3].set(Q[i][..., k])
		
		niters = self.mpniters

		if niters:

			
			x0 = [np.zeros_like(self.pintgs[self._order].system.ele_scal_upts(r2)[i])
							for i in range(netype)]
			for i in range(netype):
						self.pintgs[self._order].system.ele_banks[i][r1].set(x0[i])
			

			for _ in range(niters):
				for l in self.levels[1:]:
					self.level = l
					x0 = [np.zeros_like(self.pintg.system.ele_scal_upts(r2)[i])
							for i in range(netype)]

					for i in range(netype):
						self.pintg.system.ele_banks[i][r1].set(x0[i])

				for l, m, n in it.zip_longest(cycle, cycle[1:], csteps):
					self.level = l

					pr1 = self.projmats[l, self._order]
					pr2 = self.projmats[self._order, l]
					
					# self.pintg.jac_mult(n, hclass=self.pintgs[self._order], p=pr1, r=pr2)

					# self.pintg.jac_mult(n, f='jacobi', hclass=self.pintgs[self._order], p=pr1, r=pr2)
					if l == self._order:
						self.pintg.jac_mult(n)
					else:
						self.pintg.jac_mult(n, f='jacobi')
					# print(f'After Jac_mult for l is {l}')
					# for r in range(6):
					# 	print(f'isnan {r} is {np.isnan(self.pintgs[l].system.ele_scal_upts(r)[0]).any()}')

					if m is not None and l > m:
						self.restrict(l, m)
						# print(f'After restrict for l is {l}, m is {m}')
						# for r in range(6):
						# 	print(f'isnan {r} is , l is {l}, {np.isnan(self.pintgs[l].system.ele_scal_upts(r)[0]).any()}')
						# 	print(f'isnan {r} is , m is {m}, {np.isnan(self.pintgs[m].system.ele_scal_upts(r)[0]).any()}')
						
					elif m is not None and l < m:
						self.prolongate(l, m)
						# print(f'After prolongate for l is {l}, m is {m}')
						# for r in range(6):
						# 	print(f'isnan {r} is , l is {l}, {np.isnan(self.pintgs[l].system.ele_scal_upts(r)[0]).any()}')
						# 	print(f'isnan {r} is , m is {m}, {np.isnan(self.pintgs[m].system.ele_scal_upts(r)[0]).any()}')
						

			add(0.0, r3, 1.0, r1)

		self.pintg._eval_mat_vec(r2, r3, r1, r4)

		q = [self.system.ele_banks[i][r1].get() for i in range(netype)]

		for j in range(k+1):
			h[j] = sum([np.dot(q[i].reshape(-1), Q[i][..., j].reshape(-1))
						for i in range(len(self.system.ele_types))])
			h[j] = comm.allreduce(h[j], op=mpi.SUM)

			for i in range(netype):
				q[i] -= h[j] * Q[i][..., j]
			

		qnorm = sum([np.linalg.norm(q[i])**2 for i in range(netype)])
		qnorm = np.sqrt(comm.allreduce(qnorm, op=mpi.SUM))
		h[k+1] = qnorm
		# for i, etype in enumerate(self.system.ele_types):
		#     qnorm[etype] = (np.linalg.norm(q[i]))**2
		
		# for etype in eletype:
		#     qnorm[etype] = np.sqrt(comm.allreduce(qnorm[etype], mpi.SUM))

		for i in range(netype):
			q[i] /= h[k+1]

		return h, q
	
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
		cycle, csteps = self.cycle, self.csteps

		while self.tcurr < t:
			self.level = self._order
			# Decide on the time step
			dt = max(min(t - self.pintg.tcurr, self.pintg._dt), self.pintg.dtmin)
			

			r0, r1, r2, r3, *r4 = self.pintg._regidx
			r4 = r4[0]
			add = self._add
			nnorm = np.inf
			s = 1.0
			ntol = self.ntol = 1e-4
			comm, rank, root = get_comm_rank_root()

			nonlin_iter = 0

			while nnorm > ntol:


				for l in self.levels:
					self.level = l
					self.pintg._init_step(self.tcurr, dt)
				
				self.level = self._order
				self.pintg._init_gmres()
				x = copy.deepcopy(self.system.ele_scal_upts(r3))

				# Init the low order systems
				# nl = len(self.levels) - 1
				for l, m in it.zip_longest(self.levels, self.levels[1:]):
					if m is not None:
						self._init_loworder(l, m)
				# # Eval the coarsest grid jacobians
				for l in self.levels:
					self.pintgs[l]._eval_jac()
					
				# self.pintgs[self._order]._eval_jac()

				
				self.pintg._res()
	
				self.solve_gmres(x) 

				# r2 = Un+1,k+1 = Un+1,k + s*dUk
				add(1.0, r2, s, r3)

				nnorm = self.pintg.newton_res()

				nonlin_iter += 1
				if rank == root:
					print(nnorm)
					print(nonlin_iter)
			
			# r0 = Un+1 = r2
			add(0.0, r0, 1.0, r2)
			if rank == root:
				print("Step completed")
			
			for l in self.levels:
				print(f'nfeval at {l} is {self.pintgs[l].nfeval}')

			idxcurr = r0
			self.pintg._accept_step(dt, idxcurr)