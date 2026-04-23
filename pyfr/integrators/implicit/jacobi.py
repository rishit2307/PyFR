import numpy as np
from collections import defaultdict
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.preprocessing import StandardScaler

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

		self.jac_bsize = cfg.getint(sect, 'jac-bsize', 4000)
		self.jac_fpdtype = cfg.get(sect, 'jacobi-prec', precision)
		self.clustering = cfg.getbool(sect, 'clustering', False)
		if self.clustering:
			self.eleclust = cfg.getint(sect, 'eleclust', 128)
			self._init_kmeans(cfg, sect)
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

	def _init_jac(self, etp, rs, stele, npt, v, col, afac, h):

		if etp not in self.system.ele_types:
			return
		bsize = self.jac_bsize
		eix = self.system.ele_types.index(etp)
		neles = self.system.ele_shapes[eix][-1]

		kerns = self._get_jacinit_kerns(etp, *rs)
		self._bind_kerns(kerns, stele, bsize, neles, npt, v, col, afac, h)

		self.backend.run_kernels(kerns)

	def _shuff_jac(self):
		jacinv, jac = self.jacinv, self.jac
		kerns = [self.backend.kernel('jacshuffle', *[jacinv[i], jac[i]])
				 for i in range(len(self.system.ele_types))]

		return kerns

	def _set_jac_backend(self):
		backend = self.backend
		comm, rank, root = get_comm_rank_root()
		self.jac, self.jacinv = [], []
		self.Permut = []

		for i, etype in enumerate(self.system.ele_types):

			nupts, nvars, neles = self.system.ele_shapes[i]
			jac_bsize = min(neles, self.jac_bsize)

			jac = np.empty((jac_bsize, (nupts*nvars)**2),
							dtype=self.fpdtype)

			P = np.zeros((jac_bsize, nupts*nvars), 
						 dtype=self.backend.ixdtype)

			if self.clustering:
				jacinv = np.zeros((jac_bsize, (nupts*nvars)**2), 
							  dtype=self.fpdtype)
			else:
				jacinv = np.zeros((nupts*nvars, nupts, nvars, neles), 
								dtype=self.fpdtype)

			self.jac.append(backend.matrix(jac.shape, jac))
			self.jacinv.append(backend.matrix(jacinv.shape, jacinv))

			self.Permut.append(backend.matrix(P.shape, P, 
											  dtype=self.backend.ixdtype))

	@memoize
	def mul_jac(self, *rs):
		jac = self.jacinv
		backend = self.backend
		kerns = [backend.kernel('jacmul', *[em[r] for r in rs]+[jac[i]])
				 for i, em in enumerate(self.system.ele_banks)]

		return kerns

	@memoize
	def mul_jac_kmeans(self, *rs):
		jacinv = self.jacinv
		backend = self.backend
		emap, cluster_neles = self.emap, self.cluster_neles
		stidx = self.stidx

		kerns = [backend.kernel('jacmul_clust', *[em[r] for r in rs]+[jacinv[i]] +
						  		[emap[i]]+[cluster_neles[i], stidx[i]]) for i, em in 
								enumerate(self.system.ele_banks)]

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
		edofs = [(etype, eshapes[0], eshapes[-1]) for eshapes, etype in 
				 zip(self.system.ele_shapes, self.system.ele_types)]

		gedofs = comm.allreduce(edofs, op=mpi.SUM)

		geupts = set([(etp, upt) for etp, upt, _ in gedofs])
		gneles = defaultdict(list)
		for etp, upt, neles in gedofs:
			gneles[etp].append(neles)

		# Evaluate the Jacobians
		for etype, nupts in sorted(geupts):
			for eles in range(0, max(gneles[etype]), self.jac_bsize):
				for col in range(self.system.ncolours[etype]):
					for npt in range(nupts):
						for v in range(nvars):		
							self._add(0, raux, 1, rcurr)
							self._addid([rcurr, raux], 
										eles, self.jac_bsize,
										npt, v, col, h, etype)

							rhs(tc, raux, raux)
							self._add(-a, raux, a, rcurr_rhs)
							self._init_jac(etype, [raux, rcurr], 
											eles, npt, v, col, 1, h)

				if etype in self.system.ele_types:
					eix = self.system.ele_types.index(etype)
					neles = self.system.ele_shapes[eix][-1]

					kerns = [self.backend.kernel('getf3', *[jac[eix], jacinv[eix], P[eix]],
								                  kmeans=self.clustering)]

					self._bind_kerns(kerns, eles, self.jac_bsize, neles)
					self.backend.run_kernels(kerns)

					if self.clustering:
						jac_fpdtype = self.jac_fpdtype
						eix = self.system.ele_types.index(etype)
						jacinv_cpu = self.jacinv[eix].get().astype(np.dtype(jac_fpdtype))
						if eles < neles:
							six = min(neles - eles, self.jac_bsize)
							self._get_kmeans(eles, etype, jacinv_cpu[:six])
		if self.clustering:
			self._set_jac_kmeans()
		self.backend.wait()

	def _update_precision(self):
		jac_temp = []
		backend = self.backend
		del self.jac

		comm, rank, root = get_comm_rank_root()

		for i, etype in enumerate(self.system.ele_types):
			jac_temp.append(self.jacinv[i].get())
		
		for jacinv in self.jacinv:
			del jacinv
		
		del self.jacinv
		
		self.jacinv = []

		for i, etype in enumerate(self.system.ele_types):
			self.jacinv.append(backend.matrix(jac_temp[i].shape, 
							jac_temp[i], dtype=self.jac_fpdtype))
		
		del jac_temp
		del self._memoize_cache_

	def _set_jac_kmeans(self):
		self.jacinv = []
		backend = self.backend
		comm, rank, root = get_comm_rank_root()

		for i, jack in enumerate(self.jacinv_tmp):
			etp = self.system.ele_types[i]
			np.save(f'./npy/jac_{etp}_{rank}', np.vstack(jack))
			jackb = backend.matrix(np.vstack(jack).shape, 
						  		   np.vstack(jack), dtype=self.jac_fpdtype)

			# jack = np.load(f'./npy/jac_{etp}_{rank}.npy')
			# jackb = backend.matrix(jack.shape, 
			# 			  		   jack, dtype=self.jac_fpdtype)

			ci = np.array(self.cluster_neles[i])[None]
			ei = np.array(self.emap[i])[None]
			stix = np.cumsum(ci) - ci
			# ci = np.load(f'./npy/ci_{etp}_{rank}.npy')
			# stix = np.load(f'./npy/stix_{etp}_{rank}.npy')
			# ei = np.load(f'./npy/ei_{etp}_{rank}.npy')
			self.cluster_neles[i] = backend.matrix(ci.shape, 
										  		ci,dtype=backend.ixdtype)
			
			self.stidx[i] = backend.matrix(stix.shape, stix, 
											 dtype=backend.ixdtype)

			self.jacinv.append(jackb)
			self.emap[i] = backend.matrix(ei.shape, ei, dtype=backend.ixdtype)

			np.save(f'./npy/ci_{etp}_{rank}', ci)
			np.save(f'./npy/stix_{etp}_{rank}', stix)
			np.save(f'./npy/ei_{etp}_{rank}', ei)

		del self.jacinv_tmp

	def _init_kmeans(self, cfg, sect):
		self.eleclust = cfg.getint(sect, 'eleclust', 128)
		self.cluster_neles = [[] for _ in range(len(self.system.ele_types))]
		self.emap = [[] for _ in range(len(self.system.ele_types))]
		self.stidx = [[] for _ in range(len(self.system.ele_types))]
		self.jacinv_tmp = [[] for _ in range(len(self.system.ele_types))]

	def _get_kmeans(self, eles, etype, jacinv_cpu):
		neles = jacinv_cpu.shape[0]
		eix = self.system.ele_types.index(etype)
		comm, rank, root = get_comm_rank_root()

		if neles < self.jac_bsize:
			eleclust = 1
			self.jacinv_tmp[eix].append(jacinv_cpu)
			self.cluster_neles[eix] += np.ones((neles), dtype=np.int32).tolist()
			self.emap[eix] += (np.arange(neles, dtype=np.int32) + eles).tolist()
			return

		print(f'rank is {rank}, etype is {etype}', flush=True)

		cluster_neles_tmp = []
		emap_tmp = []
		jactp = []
		eleclust = self.eleclust

		nclust = max(neles//eleclust, 1)
		scaler = StandardScaler()
		jacinvsc = scaler.fit_transform(jacinv_cpu)
		kmeans = KMeans(n_clusters=nclust).fit(jacinvsc)
		del jacinvsc
		print(f'rank is {rank}, kmeans b4 final done', flush=True)

		cluster_neles_tmp += np.bincount(kmeans.labels_, minlength=nclust).tolist()

		jactp_sum = np.zeros((nclust, jacinv_cpu.shape[1]))
		np.add.at(jactp_sum, kmeans.labels_, jacinv_cpu)
		jactp = (jactp_sum / np.array(cluster_neles_tmp)[:, None]).tolist()
		del jactp_sum
		self.jacinv_tmp[eix] += jactp

		#Handling emap_tmp (indices + offset)
		idx_sort = np.argsort(kmeans.labels_)
		self.emap[eix] += (idx_sort + eles).tolist()

		# for clust in range(nclust):
		# 	ele_ix = np.where(kmeans.labels_ == clust)[0]
		# 	cluster_neles_tmp.append(len(ele_ix))
		# 	emap_tmp += (ele_ix+eles).tolist()
		# 	jactp.append(np.mean(jacinv_cpu[ele_ix], axis=0))

		self.cluster_neles[eix] += cluster_neles_tmp
