import math
import numpy as np

from pyfr.integrators.implicit.base import BaseImplicitIntegrator
from pyfr.mpiutil import get_comm_rank_root, mpi

class BaseImplicitController(BaseImplicitIntegrator):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		# Solution filtering frequency
		self._fnsteps = self.cfg.getint('soln-filter', 'nsteps', '0')

		# Stats on the most recent step
		self.stepinfo = []

		# Fire off any event handlers if not restarting
		if not self.isrestart:
			self._run_plugins()

	def _accept_step(self, dt, err=None):
		comm, rank, root = get_comm_rank_root()
		self.tcurr += dt
		self.nacptsteps += 1
		self.nacptchain += 1
		self.nrjctchain = 0
		self.stepinfo.append((dt, 'accept', err))

		self._invalidate_caches()

		if rank == root:
			print(f'time is {self.tcurr}, dt is {dt}, adapt err is {err}', flush=True)

		# Run any plugins
		self._run_plugins()

		# Clear the step info
		self.stepinfo = []

	def _reject_step(self, dt, rold, err=None):
		comm, rank, root = get_comm_rank_root()
		if dt <= self.dtmin:
			raise RuntimeError('Minimum sized time step rejected')
		
		if rank == root:
			print('Time step rejected')

		if rank == root:
			print(f'time is {self.tcurr}, dt is {dt}, adapt err is {err}', flush=True)

		self.newtonsolver._idxcurr = rold

		self.nacptchain = 0
		self.nrjctsteps += 1
		self.nrjctchain += 1
		self.stepinfo.append((dt, 'reject', err))

class ImplicitNoneController(BaseImplicitController):
	controller_name = 'none'
	controller_has_variable_dt = True

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		sect = 'solver-time-integrator'
		f = self.cfg.getfloat
		self._newton_div_fac = f(sect, 'newton-div-fac', 0.4)
		self._growth_fac = f(sect, 'newton-growth-fac', 1.05)
		self.dtmax = f(sect, 'dt')

	@property
	def controller_needs_errest(self):
		return False

	def advance_to(self, t):
		if t < self.tcurr:
			raise ValueError('Advance time is in the past')

		while self.tcurr < t:
			dt = max(min(t - self.tcurr, self._dt), self.dtmin)

			# Take the physical step
			rcurr, rold, rerr, nerr = self.step(self.tcurr, dt)

			# Check for newton nans
			if not math.isnan(nerr):
				self._accept_step(dt, nerr)
				self._dt = min(self._growth_fac*self._dt, self.dtmax)

			else:
				self._reject_step(dt, rold, nerr)
				self._dt *= self._newton_div_fac

class ImplicitNewtonController(BaseImplicitController):
	controller_name = 'niters'
	controller_has_variable_dt = False

	@property
	def controller_needs_errest(self):
		return False

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		sect = 'solver-time-integrator'

		# PI control values
		self._alpha = self.cfg.getfloat(sect, 'pi-alpha', 0.7)
		self._beta = self.cfg.getfloat(sect, 'pi-beta', 0.4)

		# Estimate of previous error
		self._errprev = 1.0

		# Step size adjustment factors
		self._saffac = self.cfg.getfloat(sect, 'safety-fact', 0.8)
		self._maxfac = self.cfg.getfloat(sect, 'max-fact', 2.5)
		self._minfac = self.cfg.getfloat(sect, 'min-fact', 0.1)

		# Get max time step
		self.dtmax = self.cfg.getfloat(sect, 'dt-max', 1e3)

	def advance_to(self, t):
		if t < self.tcurr:
			raise ValueError('Advance time is in the past')

		while self.tcurr < t:

			# Constants
			maxf = self._maxfac
			minf = self._minfac
			saff = self._saffac

			expa = self._alpha / 2
			expb = self._beta / 2

			# Decide on the time step
			dt = max(min(t - self.tcurr, self._dt, self.dtmax), self.dtmin)

			# Take the physical step
			rcurr, rold, rerr, nerr = self.step(self.tcurr, dt)

			# Decide the error factor
			nerr = 100 if not math.isfinite(nerr) else nerr
			self._errprev = nerr if self.nsteps == 0 else self._errprev

			# Determine time step adjustment factor
			fac = nerr**-expa * self._errprev**expb
			fac = min(maxf, max(minf, saff*fac))

			# Compute the size of the next step
			self._dt = fac*dt

			# Decide if to accept or reject the step
			if nerr < 1.0:
				self._errprev = nerr
				self._accept_step(dt, nerr)
			else:
				self._reject_step(dt, rold, nerr)

class ImplicitSoderlindController(BaseImplicitController):
	controller_name = 'soderlind'
	controller_has_variable_dt = True

	@property
	def controller_needs_errest(self):
		return True

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		sect = 'solver-time-integrator'

		# Maximum time step
		self.dtmax = self.cfg.getfloat(sect, 'dt-max', 1e2)

		# Error tolerances
		self._atol = self.cfg.getfloat(sect, 'atol')
		self._rtol = self.cfg.getfloat(sect, 'rtol')

		# Error norm
		self._norm = self.cfg.get(sect, 'errest-norm', 'l2')

		if self._atol < 10*self.backend.fpdtype_eps:
			raise ValueError('Absolute tolerance too small')

		if self._rtol < 10*self.backend.fpdtype_eps:
			raise ValueError('Relative tolerance too small')
		
		# Truncation Error Constant
		self._errbias = self.cfg.getfloat(sect, 'err-bias', 1.0)

		# Step size safety factor
		self._saff = self.cfg.getfloat(sect, 'safety-fact', 0.8)

		# Number of failures after which eta < etamxf
		self._smallnef = self.cfg.getint(sect, 'small-etaef', 2)

		# Halt simulation at maxnef failure
		self._maxnef = self.cfg.getint(sect, 'max-etaef', 7)

		# Time step reduction if niters > maxiters
		self._etacf = self.cfg.getfloat(sect, 'etacf', 0.25)

		# Set initial errors and stepsize growth
		self._errpp, self._etapp = 1, 1
		self._errp, self._etap = 1, 1

		# Step size growth
		self._etamx1 = self.cfg.getfloat(sect, 'etamax-1', 100.0)
		self._etamax = self.cfg.getfloat(sect, 'etamax', 2.5)
		self._etamxf = self.cfg.getfloat(sect, 'etamaxf', 0.3)
		self._etamin = self.cfg.getfloat(sect, 'etamin', 0.3)

		# Controller constants
		self._k1 = 1.25
		self._k2 = 0.5
		self._k3 = -0.75
		self._k4 = 0.25
		self._k5 = 0.75

	def _errest(self, rcurr, rold, rerr):
		comm, rank, root = get_comm_rank_root()

		# Get a set of kernels to estimate the integration error
		ekerns = self._get_reduction_kerns(rcurr, rold, rerr, method='errest',
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
		
		return err*self._errbias if not np.isnan(err) else 100
	
	def _estimate_eta(self, dt, err):
		errpp, etapp = self._errpp, self._etapp
		errp, etap = self._errp, self._etap
		saff = self._saff
		phat = self.stepper_order - 1

		k1, k2, k3 = self._k1, self._k2, self._k3
		k4, k5 = self._k4, self._k5

		# Set etamin and etamax
		etamin = self.dtmin / dt
		etamax = self._etamax if err < 1 else 1

		# Handle the case with more than smallnef failures
		# if self.nrjctchain >= self._smallnef and err > 1:
		# 	etamax = self._etamxf
		# 	etamin = self._etamin
		# 	import pdb;pdb.set_trace()

		# Calculate eta from history
		eta = err**(-k1/phat) * errp**(-k2/phat) * errpp**(-k3/phat)
		eta *= etap**k4 * etapp**k5

		# Handle the case of insufficient time history
		if self.nacptsteps <= 1:
			# etamax = (self._etamx1 if err < 1 and 
			#  		 self.nacptsteps == 0 else etamax)

			# Fall back to an I controller
			eta = err**(-1/phat)

		eta = min(etamax, max(eta*saff, etamin))

		return eta

	def advance_to(self, t):
		if t < self.tcurr:
			raise ValueError('Advance time is in the past')

		sord = self.stepper_order
		expa = 0.7 / sord
		expb = 0.4 / sord

		etamin = self._etamin
		etamax = self._etamax
		saff = self._saff

		while self.tcurr < t:
			# Decide on the time step
			dt = max(min(t - self.tcurr, self._dt, self.dtmax), self.dtmin)

			# Take the physical step
			rcurr, rold, rerr = self.step(self.tcurr, dt)

			# Estimate the error
			err = self._errest(rcurr, rold, rerr)

			# Compute the size of the next step
			# eta = self._estimate_eta(dt, err)
			eta = err**-expa * self._errp**expb
			eta = min(etamax, max(eta*saff, etamin))
			self._dt = eta*dt

			# self._errpp, self._etapp = self._errp, self._etap

			# Decide if accept or reject step
			if err > 1.0:
				self._errp = err
				self._reject_step(dt, rold, err=err)
			else:
				self._accept_step(dt, err=err)