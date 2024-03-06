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
		self.pintgs = {}
		
		for l in self.levels:
			if l == order:
				mcfg = cfg
			else:
				mcfg = Inifile(cfg.tostr())
				mcfg.set('solver', 'order', l)
			
			class lpsint(*bases):
				name = 'GMRES-multip'

				def _eval_jac(self, t, dt, dtfac=2.0):
					add, rhs = self._add, self.system.rhs
					comm, rank, root = get_comm_rank_root()
					r0, r1, r2, r3, *r4 = self._regidx
					r5 = r4[-1]

					self.jac = jac = defaultdict(list)
					for i in range(len(self.system.ele_types)):
						ur0 = self.system.ele_banks[i][r2].get()
						rhs(t+dt, r2, r1)
						self.backend.wait()
						dr1 = self.system.ele_banks[i][r1].get()
						
						nupts = self.system.ele_shapes[i][0]
						for col in sorted(self.system.celes.keys()):
							for v in range(self.system.nvars):
								for npt in range(nupts):
									eidx = self.system.celes[col]
									
									eps = np.zeros_like(ur0)

									eps[npt, v, eidx] = 1e-8

									ur = ur0+eps
									self.system.ele_banks[i][r5].set(ur)

									rhs(t+dt, r5, r5)
									self.backend.wait()
							
									dr2 = self.system.ele_banks[i][r5].get()

									dr = (dr1 - dr2)/1e-8

									for e in eidx:
										jac[e].append(dr[..., e].T.reshape(-1))
					cond = []
					
					for e in range(self.system.neles):
						shape = len(jac[e])

						jac[e] = np.array(jac[e]).T + (dtfac/dt)*np.eye(shape)
						cond.append(np.linalg.cond(jac[e]))

						jac[e] = np.linalg.inv(jac[e])
						
					
					
					cond = comm.allreduce(cond, op=mpi.MAX)
					if rank == root:
						print(f'rank is {rank} cond number is {np.amax(cond)}')
				
				def richardson(self, eps, tau, t, dt, dtfac):
					r0, r1, r2, r3, r4, r5 =  self._regidx
					rhs, add = self.system.rhs, self._add

					# r5 = Un+1,k+1 = Un+1,k+eps*x
					add(0.0, r5, 1.0, r2, eps, r1)

					# r5 = rhs(Un+1, k+1)
					rhs(t+dt, r5, r5)

					# r5 = rhs(Un+1,k+eps*x)/eps + x/dt
					add(-1.0/eps, r5, dtfac/dt, r1)

					# r5 = b - Ax
					add(-1.0, r5, 1.0, r3)

					# r1 = x + tau*(b - Ax)
					add(1.0, r1, tau, r5)

					# r5 = rhs(Un+1,k)
					rhs(t+dt, r2, r5)

					# r1 = x + tau*(b-Ax)
					add(1.0, r1, -tau/eps, r5)


				def jacobi(self, eps, tau, t, dt, dtfac, nsmooth):
					r0, r1, r2, r3, r4, r5 =  self._regidx
					rhs, add = self.system.rhs, self._add

					# r5 = Un+1,k+1 = Un+1,k+eps*x
					add(0.0, r5, 1.0, r2, eps, r1)

					# r5 = rhs(Un+1, k+1)
					rhs(t+dt, r5, r5)

					# r5 = rhs(Un+1,k+eps*x)/eps + x/dt
					add(-1.0/eps, r5, dtfac/dt, r1)

					# r4 = rhs(Un+1, k)
					rhs(t+dt, r2, r4)

					# r5 = -Axi = rhs(Un+1,k+eps*x)/eps + x/dt - rhs(Un+1, k)/eps
					add(-1.0, r5, -1.0/eps, r4)
					
					Axi = [self.system.ele_scal_upts(r5)[i]
							for i in range(len(self.system.ele_types))]
					
					xi = [np.zeros_like(Axi[i]) 
		   				for i in range(len(self.system.ele_types))]
					
					for _ in range(nsmooth):
						for i in range(len(self.system.ele_types)):
							nupts = self.system.ele_shapes[i][0]
							nvars = self.system.nvars
							b = self.system.ele_scal_upts(r3)[i]
							for e in range(len(self.system.neles)):
								tmp = self.jac[e] @ Axi[i][..., e].T.reshape(-1)
								xi[i][..., e] +=(2/3)*tmp.reshape(nvars, nupts).T
								tmp2 = self.jac[e] @ b[..., e].T.reshape(-1)
								xi[i][... ,e] += (2/3)*tmp2.reshape(nvars, nupts).T 

				def jac_mult(self, t, dt, dtfac, tau, nsmooth):
					r0, r1, r2, r3, r4, r5 =  self._regidx
					rhs, add = self.system.rhs, self._add
					comm, rank, root = get_comm_rank_root()

					netype = len(self.system.ele_types)

					Un = sum([np.linalg.norm(self.system.ele_scal_upts(r2)[i])**2
								for i in range(netype)])
					
					Un = np.sqrt(comm.allreduce(Un, op=mpi.SUM))
					epsmc =  np.sqrt(np.finfo(float).eps)

					for _ in range(nsmooth):
						xn = sum([np.linalg.norm(self.system.ele_scal_upts(r1)[i])**2
				 										 for i in range(netype)])
						xn = comm.allreduce(xn, op=mpi.SUM)
		
						eps =  epsmc*np.sqrt(Un + 1)/(np.sqrt(xn) + epsmc**2)

						self._richardson(eps, tau, t, dt, dtfac)
				# def jac_mult(self, lclass):
				# 	r0, r1, r2, r3, *r4 = lclass._regidx
				# 	r4, r5 = r4[0], r4[1]
					
				# 	for i in range(len(self.system.ele_types)):
				# 		nupts, nvars, neles = lclass.system.ele_shapes[i]

				# 		qin = self.system.ele_banks[i][r3].get()
				# 		qout = np.zeros_like(qin)

				# 		for e in range(lclass.system.neles):
				# 			tmp = lclass.jac[e] @ qin[..., e].T.reshape(-1)
				# 			qout[..., e] = tmp.reshape(nvars, nupts).T
						
				# 		self.system.ele_banks[i][r3].set(qout)

				
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

	def _init_loworder(self, l1, l2):
		rl0, rl1, *rl2 = self.pintgs[l2]._regidx
		rl2 = rl2[0]
		r0, r1, r2, *r3 = self.pintgs[l1]._regidx
		self.backend.run_kernels(self.mgproject(l1, r2 ,l2, rl2))
	
	def mgproject(self, l1, l1reg, l2, l2reg):
		projk = []
		for i, a in enumerate(self.projmats[l1, l2]):
			b = self.pintgs[l1].system.ele_banks[i][l1reg]
			c = self.pintgs[l2].system.ele_banks[i][l2reg]
			projk.append(self.backend.kernel('mul', a, b, out=c))

		return projk

	def restrict(self, l1, l2, t, dt, dtfac=2.0):
		r0, r1, r2, r3, r4, r5 = self.pintgs[l1]._regidx
		rl0, rl1, rl2, rl3, rl4, rl5 = self.pintgs[l2]._regidx
		add, rhs = self.pintgs[l1]._add, self.pintgs[l1].system.rhs
		comm, rank, root = get_comm_rank_root()
		netype = len(self.system.ele_types)
		epsmc = np.sqrt(np.finfo(float).eps)
		
		Un = sum([np.linalg.norm(self.system.ele_scal_upts(r2)[i])**2
								for i in range(netype)])
		Un = np.sqrt(comm.allreduce(Un, op=mpi.SUM))

		xn = sum([np.linalg.norm(self.system.ele_scal_upts(r1)[i])**2
				 			for i in range(netype)])
		xn = comm.allreduce(xn, op=mpi.SUM)
		
		eps =  epsmc*np.sqrt(Un + 1)/(np.sqrt(xn) + epsmc**2)
		

		# r5 = Un+1,k + eps*y
		add(0.0, r5, 1.0, r2, eps, r1)

		# r5 = rhs(Un+1, k + eps*y)
		rhs(t+dt, r5, r5)

		# r5 = rhs(Un+1, k + eps*y)/eps + x/dt
		add(-1.0/eps, r5, dtfac/dt, r1)

		y = [self.pintgs[l1].system.ele_scal_upts(r1)[i]
	   			for i in range(len(self.pintgs[l1].system.ele_types))]
		
		# r1 = rhs(Un+1, k)
		rhs(t+dt, r2, r1)

		# r5 =  rhs(Un+1, k + eps*y)/eps + x/dt - rhs(Un+1,k)/eps
		add(1.0, r5, 1.0/eps, r1)

		for i in range(len(self.system.ele_types)):
			self.pintgs[l1].system.ele_banks[i][r1].set(y[i])

		# r5 = b  - Ax
		add(-1.0, r5, 1.0, r3)

		self.backend.run_kernels(self.mgproject(l1, r5, l2, rl3))

	def prolongate(self, l1, l2):
		r0, r1, *r2 = self.pintgs[l1]._regidx
		rf0, rf1, *rf5 = self.pintgs[l2]._regidx
		rf5 = rf5[-1]

		self.backend.run_kernels(self.mgproject(l1, r1, l2, rf5))

		add = self.pintgs[l2]._add
		
		# r1 = ys + e
		add(1.0, rf1, 1.0, rf5)


	def solve_gmres(self, t, dt, x, dtfac=1.0):
		comm, rank, root = get_comm_rank_root()
		self.level = self._order
		y = self.pintg.y
		cs, sn = self.pintg.cs, self.pintg.sn
		ltol = self.pintg.ltol
		rnorm, m = self.pintg.rnorm, self.pintg.m
		add, rhs_with_postproc = self.pintg._add, self.pintg.system.rhs
		l2u, eletype = self.pintg.l2u, self.pintg.eletype
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
			H[:k+2, k], q = self.arnoldi(Q, k, t, dt, dtfac)

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

		netype = len(self.system.ele_types)
		cycle, csteps = self.cycle, self.csteps
		self.system.ele_banks[i][r3].set(Q[i][..., :k+1] @ y)

		for l in self.levels:
			self.level = l
			x0 = [np.zeros_like(self.pintg.system.ele_scal_upts(r2)[i])
		   					for i in range(netype)]
			for i in range(netype):
						self.pintg.system.ele_banks[i][r1].set(x0[i])
					

		for l, m, n in it.zip_longest(cycle, cycle[1:], csteps):
			self.level = l
			tau = 0.01

			self.pintg.jac_mult(t, dt, dtfac, tau, n)
			if m is not None and l > m:
				self.restrict(l, m, t, dt, dtfac=2.0)
			elif m is not None and l < m:
				self.prolongate(l, m)

		add(0.0, r3, 1.0, r1)

		# import pdb;pdb.set_trace()
		for i in range(len(self.system.ele_types)):

			x[i] += self.system.ele_banks[i][r3].get()
			self.system.ele_banks[i][r3].set(x[i])

	def arnoldi(self, Q, k, t, dt, dtfac):
		self.level = self._order
		add, rhs_with_postproc = self.pintg._add, self.system.rhs

		h = np.zeros(k+2)
		netype = len(self.system.ele_types)
		eletype = self.pintg.eletype
		cycle, csteps = self.cycle, self.csteps
		


		r0, r1, r2, r3, *r4 = self.pintg._regidx
		r4 = r4[0]

		comm, rank, root = get_comm_rank_root()

		for i in range(netype):
			self.system.ele_banks[i][r3].set(Q[i][..., k])
		
		Un = sum([np.linalg.norm(self.system.ele_scal_upts(r2)[i])**2
				for i in range(netype)])
		Un = np.sqrt(comm.allreduce(Un, op=mpi.SUM))

		Qn = sum([np.linalg.norm(self.system.ele_scal_upts(r3)[i])**2
				  for i in range(netype)])
		Qn = comm.allreduce(Qn, op=mpi.SUM)
		
		eps =  self.pintg.epsmc*np.sqrt(Un + 1)/np.sqrt(Qn)

		
		for l in self.levels:
			self.level = l
			x0 = [np.zeros_like(self.pintg.system.ele_scal_upts(r2)[i])
		   					for i in range(netype)]
			for i in range(netype):
						self.pintg.system.ele_banks[i][r1].set(x0[i])
					

		for l, m, n in it.zip_longest(cycle, cycle[1:], csteps):
			self.level = l
			tau = 0.01
			# import pdb;pdb.set_trace()
			self.pintg.jac_mult(t, dt, dtfac, tau, n)
			# print(f'After Jac_mult for l is {l}')
			# for r in range(6):
			# 	print(f'isnan {r} is {np.isnan(self.pintgs[l].system.ele_scal_upts(r)[0]).any()}')

			if m is not None and l > m:
				self.restrict(l, m, t, dt, dtfac=2.0)
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
		
		# r1 = Un+1,k + eps*Q
		add(0.0, r1, eps, r3, 1.0, r2)

		# r1 = R(Un+eps*Q)
		rhs_with_postproc(t+dt, r1, r1)      

		# r1 = R(Un+eps*Q)/eps + Q/dt
		add(-1.0/eps, r1, dtfac/dt, r3)

		# r1 = R(Un+eps*Q)/eps + Q/dt  - R(Un)/eps 
		add(1.0, r1, 1.0/eps, r4)
		
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
				self.pintg._init_gmres(self.tcurr, dt, dtfac=2.0)
				x = copy.deepcopy(self.system.ele_scal_upts(r3))

				# Init the low order systems
				nl = len(self.levels) - 1
				for l, m in it.zip_longest([self._order]*nl, self.levels[1:]):

					self._init_loworder(l, m)
					# # Eval the coarsest grid jacobians
					# if m == min(self.levels):
					# 	self.pintgs[m]._eval_jac(self.tcurr, dt, dtfac=2.0)

				self.pintg._res(self.tcurr, dt, dtfac=2.0)

				self.solve_gmres(self.tcurr, dt, x, dtfac=2.0)

				# r2 = Un+1,k+1 = Un+1,k + s*dUk
				add(1.0, r2, s, r3)

				nnorm = self.pintg.newton_res(self.tcurr, dt)

				nonlin_iter += 1
				if rank == root:
					print(nnorm)
					print(nonlin_iter)
			
			# r0 = Un+1 = r2
			add(0.0, r0, 1.0, r2)
			if rank == root:
				print("Step completed")

			idxcurr = r0
			self.pintg._accept_step(dt, idxcurr)