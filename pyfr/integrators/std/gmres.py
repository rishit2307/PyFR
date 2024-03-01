import numpy as np
from collections import defaultdict

from pyfr.inifile import Inifile
from pyfr.integrators.std.base import BaseStdIntegrator
from pyfr.integrators.std.controllers import BaseStdController
from pyfr.integrators.std.steppers import BaseStdStepper
from pyfr.util import memoize, subclass_where
from pyfr.mpiutil import get_comm_rank_root, mpi
class GMRESmultip(BaseStdIntegrator):
	def __init__(self, backend, systemcls, rallocs, mesh, initsoln, cfg):
		sect = 'solver-time-integrator'
		cn = cfg.get(sect, 'controller')
		pn = cfg.get(sect, 'scheme')

		self.backend = backend

		cc = subclass_where(BaseStdController,
							controller_name=cn)

		pc = subclass_where(BaseStdStepper, 
							stepper_name=pn)
		
		bases = [cc, pc]

		self._order = order = self.level = cfg.getint('solver', 'order')
		self._lorder = lorder = cfg.getint('solver', 'low-order', 2)
		print(f'self._lorder is {lorder}')
		self.levels = [order, lorder]
		self.pintgs = dict()
		
		for l in self.levels:
			if l == order:
				mcfg = cfg
			else:
				mcfg = Inifile(cfg.tostr())
				mcfg.set('solver', 'order', lorder)
			
			class lpsint(*bases):
				name = 'GMRES-multip'

				def _eval_jac(self, lclass, t, dt, dtfac):
					add, rhs = lclass._add, lclass.system.rhs
					comm, rank, root = get_comm_rank_root()
					r0, r1, r2, r3, *r4 = lclass._regidx

					lclass.jac = jac = defaultdict(list)
					for i in range(len(self.system.ele_types)):
						
						nupts = lclass.system.ele_shapes[i][0]
						for col in sorted(lclass.system.celes.keys()):
							for v in range(lclass.system.nvars):
								for npt in range(nupts):
									eidx = lclass.system.celes[col]
									ur0 = lclass.system.ele_banks[i][r0].get()
									eps = np.zeros_like(ur0)

									eps[npt, v, eidx] = 1e-8

									ur = ur0+eps
									lclass.system.ele_banks[i][r1].set(ur)

									rhs(t+dt, r0, r2)
									lclass.backend.wait()
									rhs(t+dt, r1, r3)
									lclass.backend.wait()
				
									dr1 = lclass.system.ele_banks[i][r2].get()
									dr2 = lclass.system.ele_banks[i][r3].get()

									dr = (dr1 - dr2)/1e-8

									for e in eidx:
										jac[e].append(dr[..., e].T.reshape(-1))
					cond = []
					
					for e in range(lclass.system.neles):
						shape = len(jac[e])

						jac[e] = np.array(jac[e]).T + (dtfac/dt)*np.eye(shape)
						cond.append(np.linalg.cond(jac[e]))
					# import pdb;pdb.set_trace()
					for e in range(lclass.system.neles):
						jac[e] = np.linalg.inv(jac[e])
						
					
					
					cond = comm.allreduce(cond, op=mpi.MAX)
					if rank == root:
						print(f'rank is {rank} cond number is {np.amax(cond)}')
				

				def jac_mult(self, lclass, t, dt, ri, dtfac):
					r0, r1, r2, r3, *r4 = lclass._regidx
					r4, r5 = r4[0], r4[1]
					epsmc = np.sqrt(np.finfo(float).eps)
					rhs, add = lclass.system.rhs, lclass._add
					comm, rank, root = get_comm_rank_root()
					netype = len(lclass.system.ele_types)
					nsmooth = 10

					l1 = int(self.system.cfg.get('solver', 'order'))
					l2 = int(lclass.system.cfg.get('solver', 'order'))
					kern = []
					for i in range(len(self.system.ele_types)):
						proj = self.projmat[i, l1,  l2]
						q = self.system.ele_banks[i][ri]
						qout = lclass.system.ele_banks[i][r4]
						kern.append(self.backend.kernel('mul', proj, q, out=qout))
					
					self.backend.run_kernels(kern, wait=True)

					

					Un = sum([np.linalg.norm(lclass.system.ele_scal_upts(r0)[i])**2
												for i in range(netype)])
	   				
					Un = np.sqrt(comm.allreduce(Un, op=mpi.SUM))
					
					x0 = [np.zeros_like(lclass.system.ele_scal_upts(r4)[i])
		   					for i in range(netype)]

					for i in range(netype):
						lclass.system.ele_banks[i][r3].set(x0[i])
					
					for i in range(nsmooth):
						Qn = sum([np.linalg.norm(lclass.system.ele_scal_upts(r3)[i])**2
				  								for i in range(netype)])

						Qn = comm.allreduce(Qn, op=mpi.SUM)
						eps = epsmc*np.sqrt(Un + 1)/(np.sqrt(Qn) + epsmc**2)

						# r2 = x/dt
						add(0.0, r2, dtfac/dt, r3)

						# r3 = u + eps*x
						add(eps, r3, 1.0, r0)

						# r3 = rhs(u+eps*x)
						rhs(t+dt, r3, r3)

						# r3 = rhs(u+eps*x)/eps + x/dt
						add(-1.0/eps, r3, 1.0 ,r2)

						# r2 = rhs(u)
						rhs(t+dt, r0, r2)

						# r3 = rhs(u+eps*x)/eps + x/dt - rhs(u)/eps
						add(1.0, r3, 1.0/eps, r2)

						Axi = lclass.system.ele_scal_upts(r3)

						for i in range(netype):
							nupts, nvars, neles = lclass.system.ele_shapes[i]
							b = lclass.system.ele_scal_upts(r4)[i]
							for e in range(lclass.system.neles):
								tmp = (2/3)*lclass.jac[e] @ Axi[i][..., e].T.reshape(-1)
								x0[i][..., e] -= tmp.reshape(nvars, nupts).T
								tmp2 = (2/3)*lclass.jac[e] @ b[..., e].T.reshape(-1)
								x0[i][..., e] += tmp2.reshape(nvars, nupts).T 

							lclass.system.ele_banks[i][r3].set(x0[i])

					for i in range(len(self.system.ele_types)):
						qin = lclass.system.ele_banks[i][r3]
						proj = self.projmat[i, l2, l1]
						qout = self.system.ele_banks[i][ri]
						kern.append(self.backend.kernel('mul', proj, qin, out=qout))
					
					self.backend.run_kernels(kern, wait=True)
		
				def _init_loworder(self, lclass):
					r0, r1, *r2 = lclass._regidx
					r2 = r2[0]
					l1 = int(self.system.cfg.get('solver', 'order'))
					l2 = int(lclass.system.cfg.get('solver', 'order'))

					for i in range(len(self.system.ele_types)):
						proj = self.projmat[i, l1, l2]
						b = self.system.ele_banks[i][r2]
						c = lclass.system.ele_banks[i][r0]
						kern = [self.backend.kernel('mul', proj, b, out=c)]
						self.backend.run_kernels(kern, wait=True)


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
		self.projmat = defaultdict(list)
		cmat = lambda m: self.backend.const_matrix(m, tags={'align'}) 
		l1, l2 = self._order, self._lorder

		for i, etp in enumerate(self.system.ele_types):
			b1 = self.pintgs[self._order].system.ubasis[i]
			b2 = self.pintgs[self._lorder].system.ubasis[i]
			self.projmat[i, l1, l2] = cmat(b1.proj_to(b2))
			self.projmat[i, l2, l1] = cmat(b2.proj_to(b1))
		
		self.pintgs[self._order].projmat = defaultdict(list)
		for i, etp in enumerate(self.system.ele_types):
			self.pintgs[self._order].projmat[i, l1, l2] = self.projmat[i, l1, l2]
			self.pintgs[self._order].projmat[i, l2, l1] = self.projmat[i, l2, l1]


	def advance_to(self, t):
		# print(f'rank is {rank}, self.pintgs[self._lordr] is {self.pintgs[self._lorder]}')
		self.pintgs[self._order].advance_to(t, lvl=self.pintgs[self._lorder])