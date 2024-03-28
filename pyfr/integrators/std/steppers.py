import numpy as np
import copy
from pyfr.integrators.std.base import BaseStdIntegrator
from pyfr.util import memoize, subclass_where
from pyfr.mpiutil import get_comm_rank_root, mpi
from collections import defaultdict

class BaseStdStepper(BaseStdIntegrator):
	def collect_stats(self, stats):
		super().collect_stats(stats)

		# Total number of RHS evaluations
		stats.set('solver-time-integrator', 'nfevals', self._stepper_nfevals)

	def _init_step(self, t, dt):
		self.t, self.dt = t, dt
		name = self.cfg.get('solver-time-integrator', 'scheme')
		stp_class = subclass_where(BaseStdStepper, stepper_name=name)
		self.dtfac = stp_class.dtfac
		self.tau = self.cfg.getfloat('solver-time-integrator', 'tau', 0.01)
		
		comm, rank, root = get_comm_rank_root()
		if rank == root:
			print(f'tau is {self.tau}')

		self.nfeval = 0

		prec = self.cfg.get('backend', 'precision')
		if prec == 'double':
			self.epsmc = np.sqrt(np.finfo(float).eps)
		else:
			self.epsmc = np.sqrt(np.finfo(np.float32).eps)

	def _eval_mat_vec(self, rdU, rrhs, ev_rru=False):

		add, rhs = self._add, self.system.rhs
		epsmc = self.epsmc
		t, dt, dtfac = self.t, self.dt, self.dtfac

		rU, rrU = self._up_rup_regidx

		# Calculate eps
		# xn = sum([np.linalg.norm(self.system.ele_scal_upts(rdU)[i])**2
		# 		  for i in range(netp)])
		# xn = comm.allreduce(xn, op=mpi.SUM)

		# Un = sum([np.linalg.norm(self.system.ele_scal_upts(rU)[i])**2
		# 						for i in range(netp)])				
		# Un = np.sqrt(comm.allreduce(Un, op=mpi.SUM))

		# eps= epsmc*np.sqrt(Un + 1)/(np.sqrt(xn) + epsmc**2)
		# eps = epsmc*np.sqrt(self.Un)
		eps = 1e-8
		
		# rrhs = rU + eps*rdU
		add(0.0, rrhs, 1.0, rU, eps, rdU)

		# rrhs = rhs(rU + eps*rdU)
		rhs(t+dt, rrhs, rrhs)

		self.nfeval += 1

		if ev_rru: 
			# rrU = rhs(t+dt, rU, rrU)
			rhs(t+dt, rU, rrU)
			self.nfeval += 1
		
		# rrhs = Ax = rhs(rU+eps*rdU)/eps + dtfac*dU/dt - rhs(rU)/eps
		add(-1.0/eps, rrhs, 1.0/eps, rrU, dtfac/dt, rdU)


	def _init_gmres(self):
		self.m = self.cfg.getint('solver-time-integrator', 'gmres-iter')
		self.rnorm= dict()

		self.ltol = 1e-16
		self.eletype = dict()

		comm, rank, root = get_comm_rank_root()

		self.eletype = set(comm.allreduce(self.system.ele_types, op=mpi.SUM))
		self.eletype = comm.bcast(self.eletype, root=root)

		self.e1 = np.zeros(self.m+1)
		self.e1[0] = 1.0

		self.sn = np.zeros(self.m)
		self.cs = np.zeros(self.m)
		self.k = 0
		self.y = [[] for _ in range(len(self.system.ele_types))]
		ru, rru = self._u_ru_regidx
		kern = self._get_reduction_kerns(ru, method='gmresnorm', norm='l2')
		self.backend.run_kernels(kern, wait=True)

		Un = np.array([sum(v for k in kern for v in k.retval)])
		comm.Allreduce(mpi.IN_PLACE, Un, op=mpi.SUM)
		self.Un = np.sqrt(float(Un))

	
class StdEulerStepper(BaseStdStepper):
	stepper_name = 'euler'
	stepper_has_errest = False
	stepper_nregs = 2
	stepper_order = 1

	@property
	def _stepper_nfevals(self):
		return self.nsteps

	def step(self, t, dt):
		add, rhs_with_postproc = self._add, self.system.rhs
		ut, f = self._regidx
		rhs_with_postproc(t, ut, f)
		add(1.0, ut, dt, f)

		return ut
	
class Euler(BaseStdStepper):
	stepper_name = 'backward-euler'
	stepper_has_errest = False
	stepper_nregs = 5
	stepper_order = 1

	@property
	def _stepper_nfevals(self):
		return self.nsteps

	def _res(self, t, dt, dtfac=1.0, eps=None):
		add, rhs_with_postproc = self._add, self.system.rhs
		eletype =  self.eletype
		comm, rank, root = get_comm_rank_root()

		r0, r1, r2, r3, *r4 = self._regidx
		r4 = r4[0]

		self.l2u = sum([np.linalg.norm(self.system.ele_scal_upts(r2)[i])**2 for 
						i in range(len(self.system.ele_types))])
		
		self.l2u = np.sqrt(comm.allreduce(self.l2u, mpi.SUM))
		
		dUn = sum([np.linalg.norm(self.system.ele_scal_upts(r3)[i])**2
				for i in range(len(self.system.ele_types))])
		dUn = comm.allreduce(dUn, op=mpi.SUM)
		dUn = np.sqrt(dUn)
		
		# max_etp = max(l2u, key=l2u.get)
		eps = self.epsmc*np.sqrt(self.l2u+1)/(dUn + self.epsmc**2)

		# r1 = Un+1,k + eps*DUn+1,k
		add(0.0, r1, eps, r3, 1.0, r2)

		# r1 = R(Un+1,k+1)
		rhs_with_postproc(t+dt, r1, r1)

		# r1 = R(Un+1,k+1)/eps + dUnk/dt
		add(-1.0/eps, r1, dtfac/dt, r3)

		# # r3 = R(Un)
		# rhs_with_postproc(t, r0, r3)

		# r4 = R(Un+1, k)
		rhs_with_postproc(t+dt, r2, r4)

		# r3 = r4 = R(Un+1, k)
		add(0.0, r3, 1.0, r4)

		# r1 = R(Un+1,k+1)/eps + dUnk/dt - R(Un+1,k)/eps
		add(1.0, r1, 1.0/eps, r4)

		# # r3 = R(Un)/2 + R(Un+1,k)/2 
		# add(dtfac/2., r3, dtfac/2., r4)

		# r3 = R(Un+1,k)  + Un+1,k/dt - Un/dt
		add(1.0, r3, -dtfac/dt, r2, dtfac/dt, r0)

		# r1 = b - Ax
		add(-1.0, r1, 1.0, r3)

	def newton_res(self, t, dt):
		add, rhs_with_postproc = self._add, self.system.rhs
		self.normele = dict()
		comm, rank, root = get_comm_rank_root()

		r0, r1, r2, r3, *r4 = self._regidx
		r4 = r4[0]

		# r1 = Du/Dt
		add(0.0, r1, 1/dt, r2, -1/dt, r0)

		# r4 = R(Un+1)
		rhs_with_postproc(t+dt, r2, r4)

		# r1 = Du/dt + R(Un+1)
		add(1.0, r1, -1, r4)
		# import pdb;pdb.set_trace()
		self.normele = sum([np.linalg.norm(self.system.ele_banks[i][r1]
							.get())**2 for i in range(len(self.system.ele_types))])
		
		self.normele = np.sqrt(comm.allreduce(self.normele, op=mpi.SUM))
		return self.normele/self._get_gndofs()
		# return self.normele
	
	def step(self, t, dt):
		r0, r1, r2, r3, *r4 = self._regidx
		r4 = r4[0]
		add = self._add
		nnorm = np.inf
		s = 1.0
		ntol = self.ntol = 1e-4

		comm, rank, root = get_comm_rank_root()
		print(f't is {t}, dt is {dt}')
		nonlin_iter = 0

		while nnorm > ntol:
			self._init_gmres()

			x = copy.deepcopy(self.system.ele_scal_upts(r3))

			self._res(t, dt, dtfac=1.0)

			self.solve_gmres(t, dt, x, dtfac=1.0)

			# r2 = Un+1,k+1 = Un+1,k + s*dUk
			add(1.0, r2, s, r3)

			nnorm = self.newton_res(t, dt)

			nonlin_iter+= 1
			if rank == root:
				print(f'Newton residual is {nnorm}')
				print(nonlin_iter)

		# r0 = Un+1 = r2
		add(0.0, r0, 1.0, r2)

		print("Step completed")

		return r0


class Trapezoidal(BaseStdStepper):
	stepper_name = 'trapezium'
	stepper_has_errest = False
	stepper_nregs = 2
	stepper_order = 1
	dtfac = 2.0

	@property
	def _stepper_nfevals(self):
		return self.nsteps
	
	def _res(self):
		t, dt, dtfac = self.t, self.dt, self.dtfac

		add, rhs_with_postproc = self._add, self.system.rhs
		rmv = self._mvec_regidx

		self._eval_mat_vec(self._duold_regidx, rmv, ev_rru=True)
		
		ru, rru = self._u_ru_regidx
		rup, rrup = self._up_rup_regidx
		rdu = self._gmres_j_regidx(0)
		

		# rru = R(Un)
		rhs_with_postproc(t, ru, rru)
		self.nfeval += 1

		# rdu0 = R(Un)/2 + R(Un+1,k)/2 
		add(0.0, rdu, dtfac/2., rru, dtfac/2., rrup)

		# rdu0 = R(Un)/2 + R(Un+1,k)/2  + Un+1,k/dt - Un/dt
		add(1.0, rdu, -dtfac/dt, rup, dtfac/dt, ru)

		# rdu0 = b - Ax
		add(1.0, rdu, -1.0, rmv)


	def newton_res(self, tp=0):
		add, rhs_with_postproc = self._add, self.system.rhs
		self.normele = dict()
		comm, rank, root = get_comm_rank_root()
		t, dt = self.t, self.dt

		rU, rrU = self._u_ru_regidx
		rUp, rrUp = self._up_rup_regidx

		# rhs_with_postproc(t, rU, rrU)

		rv = self._mvec_regidx
		add(0.0, rv, -1/2, rrU, 1/dt, rUp, -1/dt, rU)

		# add(-1/2, rv, 1/dt, rUp, -1/dt, rU)

		rhs_with_postproc(t+dt, rUp, rrUp)
		add(1.0, rv, -1/2, rrUp)

		self.nfeval+=1
		# r1 = R(Un)
		# rhs_with_postproc(t, r0, r1)

		# r1 = R(Un)/2 + Du/Dt
		# add(-1/2, r1, 1/dt, r2, -1/dt, r0)
		
		# if tp == 1.0:
		# # r4 = R(Un+1)
		#     rhs_with_postproc(t, r2, r4)
		
		# else:
		#     rhs_with_postproc(t, r2, r4)

		# r4 = R(Un+1)
		# rhs_with_postproc(t+dt, r2, r4)

		# r1 = R(Un)/2 + Du/dt + R(Un+1)/2
		# add(1.0, r1, -1/2, r4)

		self.normele = sum([np.linalg.norm(self.system.ele_banks[i][rv]
							.get())**2 for i in range(len(self.system.ele_types))])
		
		self.normele = np.sqrt(comm.allreduce(self.normele, op=mpi.SUM))
		return self.normele/self._get_gndofs()
		# return self.normele

class StdTVDRK3Stepper(BaseStdStepper):
	stepper_name = 'tvd-rk3'
	stepper_has_errest = False
	stepper_nregs = 3
	stepper_order = 3

	@property
	def _stepper_nfevals(self):
		return 3*self.nsteps

	def step(self, t, dt):
		add, rhs_with_postproc = self._add, self.system.rhs

		# Get the bank indices for each register (n, n+1, rhs)
		r0, r1, r2 = self._regidx

		# Ensure r0 references the bank containing u(t)
		if r0 != self._idxcurr:
			r0, r1 = r1, r0

		# First stage; r2 = -∇·f(r0); r1 = r0 + dt*r2
		rhs_with_postproc(t, r0, r2)
		add(0.0, r1, 1.0, r0, dt, r2)

		# Second stage; r2 = -∇·f(r1); r1 = 0.75*r0 + 0.25*r1 + 0.25*dt*r2
		rhs_with_postproc(t + dt, r1, r2)
		add(0.25, r1, 0.75, r0, 0.25*dt, r2)

		# Third stage; r2 = -∇·f(r1);
		#              r1 = 1.0/3.0*r0 + 2.0/3.0*r1 + 2.0/3.0*dt*r2
		rhs_with_postproc(t + 0.5*dt, r1, r2)
		add(2.0/3.0, r1, 1.0/3.0, r0, 2.0/3.0*dt, r2)

		# Return the index of the bank containing u(t + dt)
		return r1

class StdRK2Stepper(BaseStdStepper):
	stepper_name = 'rk2'
	stepper_has_errest = False
	stepper_nregs = 3
	stepper_order = 4

	@property
	def _stepper_nfevals(self):
		return 2*self.nsteps

	def step(self, t, dt):
		add, rhs_with_postproc = self._add, self.system.rhs

		r0, r1, r2 = self._regidx

		# Ensure r0 references the bank containing u(t)
		if r0 != self._idxcurr:
			r0, r1 = r1, r0

		rhs_with_postproc(t, r0, r1)

		add(0.0, r2, dt, r1, 1.0, r0)

		rhs_with_postproc(t+dt, r2, r2)

		add(dt/2, r1, dt/2, r2)

		add(1.0, r1, 1.0, r0)

		return r1

class StdRK4Stepper(BaseStdStepper):
	stepper_name = 'rk4'
	stepper_has_errest = False
	stepper_nregs = 3
	stepper_order = 4

	@property
	def _stepper_nfevals(self):
		return 4*self.nsteps

	def step(self, t, dt):
		add, rhs_with_postproc = self._add, self.system.rhs

		# Get the bank indices for each register
		r0, r1, r2 = self._regidx
		# Ensure r0 references the bank containing u(t)
		if r0 != self._idxcurr:
			r0, r1 = r1, r0

		# First stage; r1 = -∇·f(r0)
		rhs_with_postproc(t, r0, r1)

		# Second stage; r2 = r0 + dt/2*r1; r2 = -∇·f(r2)
		add(0.0, r2, 1.0, r0, dt/2.0, r1)
		rhs_with_postproc(t + dt/2.0, r2, r2)

		# As no subsequent stages depend on the first stage we can
		# reuse its register to start accumulating the solution with
		# r1 = r0 + dt/6*r1 + dt/3*r2
		add(dt/6.0, r1, 1.0, r0, dt/3.0, r2)

		# Third stage; here we reuse the r2 register
		# r2 = r0 + dt/2*r2
		# r2 = -∇·f(r2)
		add(dt/2.0, r2, 1.0, r0)
		rhs_with_postproc(t + dt/2.0, r2, r2)

		# Accumulate; r1 = r1 + dt/3*r2
		add(1.0, r1, dt/3.0, r2)

		# Fourth stage; again we reuse r2
		# r2 = r0 + dt*r2
		# r2 = -∇·f(r2)
		add(dt, r2, 1.0, r0)
		rhs_with_postproc(t + dt, r2, r2)

		# Final accumulation r1 = r1 + dt/6*r2 = u(t + dt)
		add(1.0, r1, dt/6.0, r2)

		# Return the index of the bank containing u(t + dt)
		return r1


class StdRKVdH2RStepper(BaseStdStepper):
	# Coefficients
	a = []
	b = []
	bhat = []

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		# Register our pointwise kernel
		self.backend.pointwise.register('pyfr.integrators.std.kernels.rkvdh2')

		# Compute the coefficients
		self.c = [0.0] + [sum(self.b[:i]) + ai for i, ai in enumerate(self.a)]
		self.e = [b - bh for b, bh in zip(self.b, self.bhat)]

		self._nstages = len(self.c)

	@memoize
	def _get_rkvdh2_kerns(self, stage, r1, r2, rold=None, rerr=None):
		kerns = []
		tplargs = {
			'a': self.a, 'b': self.b, 'e': self.e,
			'stage': stage, 'nstages': self._nstages,
			'nvars': self.system.nvars, 'errest': rold is not None
		}

		for dims, em in zip(self.system.ele_shapes, self.system.ele_banks):
			if rold is not None:
				kern = self.backend.kernel(
					'rkvdh2', tplargs=tplargs, dims=[dims[0], dims[2]],
					r1=em[r1], r2=em[r2], rold=em[rold], rerr=em[rerr],
				)
			else:
				kern = self.backend.kernel(
					'rkvdh2', tplargs=tplargs, dims=[dims[0], dims[2]],
					r1=em[r1], r2=em[r2],
				)

			kerns.append(kern)

		return kerns

	@property
	def stepper_has_errest(self):
		return self.controller_needs_errest and len(self.bhat)

	@property
	def _stepper_nfevals(self):
		return len(self.b)*self.nsteps

	@property
	def stepper_nregs(self):
		return 4 if self.stepper_has_errest else 2

	def step(self, t, dt):
		run_kernels = self.backend.run_kernels
		rhs_with_postproc = self.system.rhs

		r1 = self._idxcurr
		r2, *rs = set(self._regidx) - {r1}

		# Evaluate the stages in the scheme
		for i, ci in enumerate(self.c):
			# Compute -∇·f
			rhs_with_postproc(t + ci*dt, r2 if i > 0 else r1, r2)

			# Fetch the appropriate RK accumulation kernels
			kerns = self._get_rkvdh2_kerns(i, r1, r2, *rs)

			# Bind the arguments
			for k in kerns:
				k.bind(dt=dt)

			# Execute
			run_kernels(kerns)

			# Swap
			r1, r2 = r2, r1

		# Return
		return (r2, *rs) if len(rs) else r2


class StdRK34Stepper(StdRKVdH2RStepper):
	stepper_name = 'rk34'
	stepper_order = 3

	a = [
		11847461282814 / 36547543011857,
		3943225443063 / 7078155732230,
		-346793006927 / 4029903576067
	]

	b = [
		1017324711453 / 9774461848756,
		8237718856693 / 13685301971492,
		57731312506979 / 19404895981398,
		-101169746363290 / 37734290219643
	]

	bhat = [
		15763415370699 / 46270243929542,
		514528521746 / 5659431552419,
		27030193851939 / 9429696342944,
		-69544964788955 / 30262026368149
	]


class StdRK45Stepper(StdRKVdH2RStepper):
	stepper_name = 'rk45'
	stepper_order = 4

	a = [
		970286171893 / 4311952581923,
		6584761158862 / 12103376702013,
		2251764453980 / 15575788980749,
		26877169314380 / 34165994151039
	]

	b = [
		1153189308089 / 22510343858157,
		1772645290293 / 4653164025191,
		-1672844663538 / 4480602732383,
		2114624349019 / 3568978502595,
		5198255086312 / 14908931495163
	]

	bhat = [
		1016888040809 / 7410784769900,
		11231460423587 / 58533540763752,
		-1563879915014 / 6823010717585,
		606302364029 / 971179775848,
		1097981568119 / 3980877426909
	]
