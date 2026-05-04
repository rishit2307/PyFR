import math

from pyfr.integrators.implicit.base import BaseImplicitIntegrator
from pyfr.mpiutil import get_comm_rank_root

class BaseImplicitStepper(BaseImplicitIntegrator):
	pass

class BaseESDIRKStepper(BaseImplicitStepper):
	bhat = []

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	@property
	def stepper_has_errest(self):
		return self.controller_needs_errest and len(self.bhat)
	
	@property
	def stepper_nregs(self):
		return 3 if self.stepper_has_errest else 2

	def step(self, t, dt):
		newtonsolver = self.newtonsolver
		stage_regidx = newtonsolver.register._stage_regidx
		newtonsolver = self.newtonsolver

		newtonsolver.store_current_solution()
		newtonsolver.init_step(t, nsteps=self.nsteps,
						       nrjctchain=self.nrjctchain)

		for i, (acoeffs, ccoeff) in enumerate(zip(self.a, self.c), start=1):

			ac = [acoeff*dt for acoeff in acoeffs]
			rcurr, rold, nerr = newtonsolver.solve(t+ccoeff*dt, ac, i)
			# if diverge:
			# 	self.ntblowup = False
			# 	for plugin in self.plugins:
			# 		pname = getattr(plugin, 'name', 'other')
			# 		if pname == 'writer':
			# 			print(f'Writing solution')
			# 			plugin(self)
			# 			raise RuntimeError('NNorm is nan')

		if self.stepper_has_errest:
			rerr = newtonsolver.register._err_regidx

			consts = [0.0] + [(b - bh)*dt for 
							  b, bh in zip(self.a[-1], self.bhat)]

			regidxs = [rerr] + stage_regidx[:len(self.bhat) + 1]
			self._addv(consts, regidxs)

		regerr = rerr if self.stepper_has_errest else None

		return rcurr, rold, regerr, nerr

class TrapeziumStepper(BaseESDIRKStepper):
	stepper_name = 'trapezium'
	stepper_order = 2
	nstages = 2
	fsal = True
	exp_first_stage = True

	a = [[1/2, 1/2]]
	c = [1]


class ESDIRK32Stepper(BaseESDIRKStepper):
	stepper_name = 'esdirk2'
	stepper_order = 2
	nstages = 3
	exp_first_stage = True

	gamma = 1 - 1/math.sqrt(2)
	b2 = math.sqrt(2)/4
	fsal = True
	beta = (4 - math.sqrt(2))/8
	delta = 1/(2*math.sqrt(2))

	a = [[gamma, gamma],
		 [b2, b2, gamma]]
	c = [2*gamma, 1]

	bhat = [beta, beta, delta]


class ESDIRK3Stepper(BaseESDIRKStepper):
	stepper_name = 'esdirk3'
	stepper_order = 3
	nstages = 4
	exp_first_stage = True
	fsal = True

	a = [[1767732205903/4055673282236, 1767732205903/4055673282236],
		 [2746238789719/10658868560708, -640167445237/6845629431997, 
		  1767732205903/4055673282236],
		  [1471266399579/7840856788654, -4482444167858/7529755066697, 
		   11266239266428/11593286722821, 1767732205903/4055673282236]]
	
	bhat = [2756255671327/12835298489170, -10771552573575/22201958757719, 
			9247589265047/10645013368117, 2193209047091/5459859503100]

	c = [1767732205903/2027836641118, 3/5, 1]


class ESDIRK4Stepper(BaseESDIRKStepper):
	stepper_name = 'esdirk4'
	stepper_order = 4
	nstages = 6
	exp_first_stage = True
	fsal = True

	a = [[1/4, 1/4], [8611/62500, -1743/31250, 1/4], 
		 [5012029/34652500, -654441/2922500, 174375/388108, 1/4], 
		 [15267082809/155376265600, -71443401/120774400, 
		  730878875/902184768, 2285395/8070912, 1/4], 
		 [82889/524892, 0 ,15625/83664, 69875/102672, -2260/8211, 1/4]]

	c = [1/2, 83/250, 31/50, 17/20, 1]

	bhat = [4586570599/29645900160, 0, 178811875/945068544,
			814220225/1159782912, -3700637/11593932, 
			61727/225920]
