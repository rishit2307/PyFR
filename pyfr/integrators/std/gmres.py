
import numpy as np
from collections import defaultdict
import copy
import itertools as it
import time
import math
import nvtx
from pyfr.inifile import Inifile
from pyfr.integrators.std.base import BaseStdIntegrator
from pyfr.integrators.std.controllers import BaseStdController
from pyfr.integrators.std.steppers import BaseStdStepper
from pyfr.util import memoize, subclass_where
from pyfr.mpiutil import get_comm_rank_root, mpi
import sklearn
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

				def _init_jacobians(self):
					backend = self.backend

					self.jac = jac = {}
					self.P = {}

					for etp in self.system.ele_types:
						i = self.system.ele_types.index(etp)
						nupts = self.system.ele_shapes[i][0]
						neles = self.system.ele_shapes[i][-1]
						nvars = self.system.ele_shapes[i][1]

						self.jac[etp] = jac[etp] = np.empty((nupts*nvars, nupts, nvars, neles))
						self.P[etp] = np.zeros((neles, nupts*nvars))

						self.jacinv = jacinv = {}
						self.jacinv[etp] = jacinv[etp] = np.zeros((neles, (nupts*nvars)**2))
						self.jacinv[etp] = backend.matrix(jacinv[etp].shape, jacinv[etp])
						self.jac[etp] = backend.matrix(jac[etp].shape, jac[etp])

						self.P[etp] = backend.matrix(self.P[etp].shape, self.P[etp], dtype=self.backend.ixdtype)
						print(f'init jacobians')

				def _eval_jac(self):
					add, rhs = self._add, self.system.rhs
					rup, rrup = self._up_rup_regidx
					r0, r1, *r = self._pseudo_regidx

					jac, jacinv = self.jac, self.jacinv

					t, dt, dtfac = self.t, self.dt, self.dtfac
						
					# self.jac[etp] = jac[etp] = np.random.rand(nupts*nvars, nupts, nvars, neles)
					h = self.eval_norm2(rup)*self.epsmc
					for etp in sorted(self.system.ele_types):
						i = self.system.ele_types.index(etp)
						nupts = self.system.ele_shapes[i][0]

						celes = self.system.celes[etp]
						for col in range(self.system.ncolours[etp]):
							for v in range(self.system.nvars):
								for npt in range(nupts):

									self._addid(celes, [rup, r0], npt, v, col, h)
									rhs(t+dt, r0, r1)
									self._add(-1.0/h, r1, 1.0/h, rrup)
									kerns = self._init_jac(r1, etp, celes)
									self.bind_kerns(kerns, npt, v, col, dtfac/dt)
									backend.run_kernels(kerns)

					for etp in self.system.ele_types:
						
						kern = backend.kernel('getf3', *[jac[etp], jacinv[etp], self.P[etp]])

						backend.run_kernels([kern])	

						shufkerns = self._shuff_jac(etp)
						backend.run_kernels(shufkerns)

					self.backend.wait()

				def _eval_jac_cpu(self):
					add, rhs = self._add, self.system.rhs
					rup, rrup = self._up_rup_regidx
					r0, r1, *r = self._pseudo_regidx
					t, dt, dtfac = self.t, self.dt, self.dtfac

					h = self.eval_norm2(rup)*self.epsmc

					if hasattr(self, 'jacinv'):
						del self.jacinv

					for etp in sorted(self.system.ele_types):
						i = self.system.ele_types.index(etp)
						(npts, nv, neles) = self.system.ele_shapes[i]
						jac = np.zeros((npts*nv, npts*nv, neles))

						celes = self.system.celes[etp].get()[0]

						for col in range(self.system.ncolours[etp]):
							for v in range(nv):
								for u in range(npts):
									tpup = self.system.ele_banks[0][rup].get()
									tpup[u, v, celes == col] += h
									self.system.ele_banks[0][r0].set(tpup)

									rhs(t+dt, r0, r1)
									add(-1.0/h, r1, 1.0/h, rrup)
									tpr1 = self.system.ele_banks[0][r1].get()[..., celes==col]
									jac[:, u*nv + v, celes == col] = tpr1.reshape(npts*nv, -1)

					jac += np.eye(npts*nv)[..., None]*dtfac/dt
					self.jacinv_cpu = np.linalg.inv(jac.transpose(2, 0, 1))

					del jac
					

				def _cluster_jac(self):
					self.nclusters = nclusters = 1				
					self.ele_cluster_map = []

					del self.jacinv
					del self.P
					del self._memoize_cache_
					

					comm, rank, root = get_comm_rank_root()
					nupts, nvars, neles = self.system.ele_shapes[0]
					N = nupts*nvars
					jacinv_cpu = self.jac['hex'].get().reshape(N, N, neles)

					del self.jac


					jacinv_cpu = jacinv_cpu.transpose(2, 0, 1)

					self.jacinv_diff = np.zeros((neles, N, N))
					self.jacinv_avg = np.zeros((nclusters, N, N))
					print(f'rank is {rank}, begin kms, cluster terms are {nclusters}')

					kmeans = sklearn.cluster.MiniBatchKMeans
					self.kms = kmeans(n_clusters=nclusters).fit(jacinv_cpu.reshape(neles, -1))
					print(f'rank is {rank}, kms done')

					for lab in np.unique(self.kms.labels_):
						eid = lab == self.kms.labels_
						self.jacinv_avg[lab] = np.mean(jacinv_cpu[eid], axis=0)

					for ele in range(neles):
						lab = self.kms.labels_[ele]
						self.jacinv_diff[ele] = self.jacinv_avg[lab] - jacinv_cpu[ele]

					del jacinv_cpu
					print('avg done')

				def _compute_jac_svd(self):
					comm, rank, root = get_comm_rank_root()

					# nclust = self.nclusters
					nclust = 10

					nupts, nvars, neles = self.system.ele_shapes[0]
					N = nupts*nvars

					print(f'svd terms are {nclust}')
					print(f'rank is {rank} ,shape is {self.jacinv_diff.shape}')
					self.jacinv = np.zeros((neles, nupts*nvars, nupts*nvars))

					# comm.Barrier()

					# nb = 1000
					# U = np.zeros((neles, N**2))
					# W = np.zeros((neles, N**2))
					# V = np.zeros((neles, N))
					# jdiff = np.zeros((nb, N**2))

					# jdiff = self.jacinv_diff[:nb].reshape(-1, N, N).swapaxes(1, 2).reshape(-1, N**2)
					# diffb = backend.matrix(jdiff.shape, jdiff)
					# Ub = backend.matrix(U[:nb].shape, U[:nb])
					# Vb = backend.matrix(V[:nb].shape, V[:nb])
					# Wb = backend.matrix(W[:nb].shape, W[:nb])

					# for e in range(0, neles, nb):
					# 	print(f'rank is {rank}, e is {e}')
					# 	ix = min(nb, neles - e)

					# 	jdiff[:ix] = self.jacinv_diff[e:e+nb].reshape(-1, N, N).swapaxes(1, 2).reshape(-1, N**2)
					# 	diffb.set(jdiff)

					# 	kern =  backend.kernel('svd', *[diffb, Ub, Vb, Wb, ix])
					# 	backend.run_kernels([kern], wait=True)

					# 	U[e:e+nb] = Ub.get()[:ix].reshape(-1, N, N).swapaxes(1, 2).reshape(-1, N**2)
					# 	V[e:e+nb] = Vb.get()[:ix]
					# 	W[e:e+nb] = Wb.get()[:ix].reshape(-1, N, N).swapaxes(1, 2).reshape(-1, N**2)

					
					# comm.Barrier()
					# del self.jacinv_diff
					# del jdiff
					# del diffb
					# del Ub
					# del Wb
					# del Vb
					# U = U.reshape(neles, N, N)
					# W = W.reshape(neles, N, N)
					# U = (U[..., :nclust] * V[:, None, :nclust]) @ W[:, :nclust]
					
					# del V, W
					U, V, W = np.linalg.svd(self.jacinv_diff)

					U = U[..., :nclust]
					V = V[..., :nclust]
					W = W[:, :nclust]
					print(f'svd done, U shape is {U.shape}')
					print(f'svd done, V shape is {V.shape}')
					print(f'svd done, V shape is {W.shape}')
					# comm.Barrier()

					U= (U* V[:, None]) @ W

					# print('diff done')
					del V
					del W
					del self.jacinv_diff

					for ele in range(neles):
						lab = self.kms.labels_[ele]
						self.jacinv[ele] = self.jacinv_avg[lab] - U[ele]
						# self.jacinv[ele] = self.jacinv_avg[lab]

					# del U
					# del self.jacinv_avg
					del self.jacinv_avg, U

					self.jacinv = self.jacinv.reshape(neles, nupts*nvars, nupts, nvars)

					self.jacinv = self.jacinv.transpose(1, 2, 3, 0)
					self.jacinv = backend.matrix(self.jacinv.shape, self.jacinv)

					print('final done')

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
				@memoize
				def _mul_jac_kmeans(self, *rs):
					kern = []
					em = self.system.ele_banks

					if hasattr(self, 'jac'):
						jacinv = self.jac['hex']
						print(f'HI from jac kmeans')
					else:
						print(f'HI from jacinv kmeans')
						jacinv = self.jacinv

					kern.append(self.backend.kernel('jacmul', 
									 *[em[0][r] for r in rs] + [jacinv]))

					return kern

				def _verify_jac(self):
					pass

				def jacobi(self, nsmooth):
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
						add(1.0, r0, 1.0, r1)


				def jacobi_cpu(self, nsmooth):
					r0, r1, r2 = self._pseudo_regidx
					rmv = self._mvec_regidx
					rsrc = self._src_regidx

					nupts, nvars, neles = self.system.ele_shapes[0]
					add = self._add
					for _ in range(nsmooth):

						self._eval_mat_vec(r0, rmv)

						# rmv = b - Axi
						add(-1.0, rmv, 1.0, rsrc)

						# r1 = J^-1*(b - Axi)
						kerns = self._mul_jac_kmeans( rmv, r1)
						backend.run_kernels(kerns)

						# r0 = r0 + w*r1
						add(1.0, r0, 1.0, r1)

				def _jacobi_direct(self):
					r0, r1, *r = self._pseudo_regidx
					rmv  = self._mvec_regidx
					rsrc = self._src_regidx

					# rmv = D^(-1) * rsrc
					kerns = self._mul_jac(rsrc, rmv)
					backend.run_kernels(kerns)
					
					# r0 = A*rmv
					self._eval_mat_vec(rmv, r0)

				def jac_mult(self, nsmooth):
					self.jacobi_cpu(nsmooth)



			self.pintgs[l] = lpsint(backend, systemcls, rallocs, 
									mesh, initsoln, mcfg)
		
		self.system = self.pintgs[order].system
		self.pintg._init_jacobians()
	
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
		self.mul_time = 0
	
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

		with nvtx.annotate(message="LINALG_SOLVE", color='blue'):
			y =  np.linalg.solve(H[:k+1, :k+1], beta[:k+1])

		rdu, rj = self.pintg._du_regidx, self._gmres_j_regidx
		
		consts = [0.0]+list(y)

		rp = self.pintg._prec_regs
		regidxs = [rdu] + [r for r in rp]
		self._addv(consts, regidxs)

		self._add(1.0, rdu, 1.0, self.pintg._duold_regidx)

	# @nvtx.annotate(color='yellow')
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
				self.pintg.jac_mult(n)



	def arnoldi(self, k):
		self.mg_vcycle()

		self.level = self._order
		add = self.pintg._add

		h = np.zeros(k+1)
		comm, rank, root = get_comm_rank_root()
		rkp1 = self.pintg._gmres_j_regidx(k+1)
		rdu = self.pintg._du_regidx
		rmv = self.pintg._mvec_regidx
		r0 = self.pintg._pseudo_regidx[0] if self.mpniters else rdu

		rprec = self.pintg._prec_regidx
		add(0.0, rprec, 1.0, r0)
		self.pintg._eval_mat_vec(r0, rmv)
		rji = self.pintg._gmres_j_regidx

		kerns = [self._get_dot_kerns(rji(j), rmv) for j in range(k+1)]
		self.backend.run_kernels([krn for kern in kerns for krn in kern])
		self.backend.wait()

		for i, kern in enumerate(kerns):
			h[i] = sum([v.retval for v in kern])
		
		with nvtx.annotate("MPI_DOT_CALL", color='green'):
			comm.Allreduce(mpi.IN_PLACE, h, op=mpi.SUM)

		self._addv([1.0] + list(-h), [rmv] + [rji(j) for j in range(k+1)])
		qnorm = self.pintg.eval_norm2(rmv)
		add(0.0, rkp1, 1.0/qnorm, rmv)

		h = np.append(h, qnorm)
		return h

	# @nvtx.annotate(color='orange')
	def giv_rot(self, h, cs, sn, k):
		for i in range(k):
			temp = cs[i] * h[i] + sn[i] * h[i+1]

			h[i+1] = -sn[i] * h[i] + cs[i] * h[i+1]
			h[i] = temp
		
		cs_k, sn_k = self.giv(h[k], h[k+1])

		h[k] = cs_k * h[k] + sn_k * h[k+1]
		h[k+1] = 0.0


		return h, cs_k, sn_k

	# @nvtx.annotate(color='blue')
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
			while nnorm > ntol:

				for l in self.levels:
					self.level = l
					self.pintg._init_step(self.tcurr, dt)

				self.level = self._order
				if rank == root:
					print(f't is {self.tcurr}, dt is {dt}')

				self.pintg._init_gmres()

				self.pintg._res(ev_rru=True)
				if self.pintg.nacptsteps == 0 and self.mpniters > 0:
					# self.pintg._eval_jac_cpu()
					self.pintg._eval_jac()
					print(f'rank is {rank}, Jacobian evaluated')

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

				if self.pintg.nacptsteps == 0 and nnorm < ntol:
					print(f"HI from cluster")
					self.pintg._cluster_jac()
					print(f'rank is {rank}, cluster done')
					self.pintg._compute_jac_svd()
					print(f'rank is {rank}, svd done')


			rU, rrU = self.pintgs[self._order]._u_ru_regidx
			# r0 = Un+1 = r2

			add(0.0, rU, 1.0, rUp)

			for l in self.levels:
				if rank == root:
					print(f'nfeval at {l} is {self.pintgs[l].nfeval}')

			idxcurr = rU
			self.pintg._accept_step(dt, idxcurr)

