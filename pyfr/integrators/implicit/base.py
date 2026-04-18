from pyfr.integrators.base import BaseIntegrator
from pyfr.integrators.base import BaseCommon
from pyfr.integrators.implicit.newton import BaseNonLinearSolver
from pyfr.util import subclass_where


class BaseImplicitIntegrator(BaseCommon, BaseIntegrator):
	formulation = 'implicit'

	def __init__(self, backend, systemcls, rallocs, mesh, initsoln, cfg):
		super().__init__(backend, rallocs, mesh, initsoln, cfg)

		tstart = self.tstart if not self.isrestart else self.tcurr

		newtonsolver = subclass_where(BaseNonLinearSolver, solver_name='newton')
		self.newtonsolver = newtonsolver(backend, systemcls, 
										 rallocs, mesh, initsoln, cfg, 
										 self.nstages, self.stepper_nregs, 
										 tstart, self._dt)

		self.system = self.newtonsolver.system

		# Handles blowup
		self.ntblowup = True
		
		# Event handlers for advance_to
		self.plugins = self._get_plugins(initsoln)

		# Commit the system
		self.system.commit()

		# Pre-process solution if necessary
		self.system.preproc(self.tcurr,
							self.system.ele_scal_upts(self.newtonsolver._idxcurr))

	@property
	def soln(self):
		if not self._curr_soln:
			self.system.postproc(self.newtonsolver._idxcurr)
			self._curr_soln = self.system.ele_scal_upts(self.
											   newtonsolver._idxcurr)

		return self._curr_soln

	@property
	def grad_soln(self):
		system = self.system

		if not self._curr_grad_soln:
			system.postproc(self.newtonsolver._idxcurr)
			system.compute_grads(self.tcurr, self.newtonsolver._idxcurr)
			self._curr_grad_soln = [e.get() for e in system.eles_vect_upts]

		return self._curr_grad_soln

	@property
	def dt_soln(self):
		system = self.system

		if not self._curr_dt_soln:
			copy = self.soln
			system.rhs(self.tcurr, self.newtonsolver._idxcurr, 
			  			self.newtonsolver._idxcurr)
			self._curr_dt_soln = system.ele_scal_upts(self.
											 newtonsolver._idxcurr)

			# Reset current register with original contents
			for c, e in zip(copy, system.ele_banks):
				e[self.newtonsolver._idxcurr].set(c)

		return self._curr_dt_soln

	@property
	def controller_needs_errest(self):
		pass

	@property
	def entmin(self):
		return self.system.get_ele_entmin_int()

	@entmin.setter
	def entmin(self, value):
		self.system.set_ele_entmin_int(value)
