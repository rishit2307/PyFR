import numpy as np

from pyfr.integrators.base import BaseCommon
from pyfr.mpiutil import mpi, get_comm_rank_root
from pyfr.util import memoize


class BlockJacobi(BaseCommon):
	def __init__(self, system, backend, register, cfg, epsmc):
		self.system = system
		self.backend = backend
		self.register = register
		self.cfg = cfg
		self.epsmc = epsmc

		sect = 'solver-time-integrator'
		precision = cfg.get('backend', 'precision')
		if precision == 'double':
			self.fpdtype = np.float64
		else:
			self.fpdtype = np.float32

		self.jac_fpdtype = cfg.get(sect, 'jacobi-prec', precision)
		self._set_jac_backend()

	def _bind_kerns(self, kerns, *args):
		for k in kerns:
			k.bind(*args)

	@memoize
	def _get_jacinit_kerns(self, etp, *rs):
		celes = self.system.celes[etp]
		etype_ix = self.system.ele_types.index(etp)
		jac = self.jac[etype_ix]
		em = self.system.ele_banks[etype_ix]
		kerns = [self.backend.kernel('jacinit', 
				 *[em[r] for r in rs] +[jac, celes])]

		return kerns

	def _init_jac(self, etp, rs, npt, v, col, afac, h):

		if etp not in self.system.ele_types:
			return

		kerns = self._get_jacinit_kerns(etp, *rs)
		self._bind_kerns(kerns, npt, v, col, afac, h)

		self.backend.run_kernels(kerns)

	def _shuff_jac(self):
		jacinv, jac = self.jacinv, self.jac
		kerns = [self.backend.kernel('jacshuffle', *[jacinv[i], jac[i]])
				 for i in range(len(self.system.ele_types))]

		return kerns
	
	def _set_jac_backend(self):
		backend = self.backend

		self.jac, self.jacinv = [], []
		self.Permut = []
		for i, etype in enumerate(self.system.ele_types):
			nupts, nvars, neles = self.system.ele_shapes[i]
			jac = np.empty((nupts*nvars, nupts, nvars, neles),
							dtype=self.fpdtype)
			
			P = np.zeros((neles, nupts*nvars), 
						 dtype=self.backend.ixdtype)
			
			jacinv = np.zeros((neles, (nupts*nvars)**2), 
							  dtype=self.fpdtype)

			self.jac.append(backend.matrix(jac.shape, jac))
			self.jacinv.append(backend.matrix(jacinv.shape, jacinv))

			self.Permut.append(backend.matrix(P.shape, P, 
											  dtype=self.backend.ixdtype))

	@memoize
	def mul_jac(self, *rs):
		jac = self.jac
		backend = self.backend

		kerns = [backend.kernel('jacmul', *[em[r] for r in rs]+[jac[i]])
				 for i, em in enumerate(self.system.ele_banks)]

		return kerns

	def _eval_jac(self, tc, a, currstg, rcurr):
		add, rhs = self._add, self.system.rhs
		backend = self.backend
		reg = self.register
		comm, rank, root = get_comm_rank_root()
		nvars = self.system.nvars

		raux = reg._aux_regidx
		rcurr_rhs = reg._stage_regidx[currstg]

		h = self.epsmc

		jac, jacinv = self.jac, self.jacinv
		P = self.Permut
		eupts = [(eshapes[0], etype) for eshapes, etype in 
				 zip(self.system.ele_shapes, self.system.ele_types)]

		geupts = set(comm.allreduce(eupts, op=mpi.SUM))

		for nupts, etype in sorted(geupts):
			for col in range(self.system.ncolours[etype]):
				for npt in range(nupts):
					for v in range(nvars):
						self._add(0.0, raux, 1.0, rcurr)
						self._addid([rcurr, raux], 
									npt, v, col, h, etype)

						rhs(tc, raux, raux)
						self._add(-a, raux, a, rcurr_rhs)
						self._init_jac(etype, [raux, rcurr], 
									   npt, v, col, 1.0, h)

		kerns = [self.backend.kernel('getf3', *[jac[i], jacinv[i], P[i]])
				 for i in range(len(self.system.ele_types))]

		self.backend.run_kernels(kerns)
		kerns = self._shuff_jac()
		self.backend.run_kernels(kerns, wait=True)

	def _update_precision(self):
		jac_temp = []
		backend = self.backend
		del self.jacinv
		comm, rank, root = get_comm_rank_root()

		for i, etype in enumerate(self.system.ele_types):
			jac_temp.append(self.jac[i].get())
		
		for jac in self.jac:
			del jac
		
		del self.jac
		
		self.jac = []

		for i, etype in enumerate(self.system.ele_types):
			self.jac.append(backend.matrix(jac_temp[i].shape, 
							jac_temp[i], dtype=self.jac_fpdtype))
		
		del jac_temp
		del self._memoize_cache_