import numpy as np
from pyfr.integrators.base import BaseCommon
from pyfr.integrators.implicit.jacobi import BlockJacobi

class GMRESSolver(BaseCommon):
	def __init__(self, backend, system, cfg, register, tstart):

		self.system = system
		self.backend = backend
		
		sect = 'solver-time-integrator'
		self.niters = cfg.getint(sect, 'gmres-niters', 10)
		self.ltol = cfg.getfloat(sect, 'gmres-tol', 1e-2)

		fpdtype = self.cfg.get('backend', 'precision')
		if fpdtype == 'double':
			self.epsmc = np.sqrt(np.finfo(float).eps)
		else:
			self.epsmc = np.sqrt(np.finfo(np.float32).eps)

		self.register = register

		self.prec = cfg.get(sect, 'precondition', None)
		if self.prec in ['left', 'right']:
			self.jacobi_prec = BlockJacobi(system, backend, register, self.epsmc)
			self.dtjac_out = cfg.get(sect, 'dtjac-out')
			self.dtjac_start = tstart

		self.dtmin = cfg.get(sect, 'dt-min', 1e-12)
		self.iter = 0

		self.e1 = np.zeros(self.niters+1)
		
		self.e1[0] = 1.0

		self.sn = np.zeros(self.niters)
		self.cs = np.zeros(self.niters)
		self.k = 0
		self.y = [[] for _ in range(len(self.system.ele_types))]	

	
	def solve(self, tcurr, dt):
		if self.prec and (tcurr - self.dtjac_out) >= (self.dtjac_start - dt):
			self.jacobi_prec._eval_jac(tcurr, dt)
			self.dtjac_start = tcurr





