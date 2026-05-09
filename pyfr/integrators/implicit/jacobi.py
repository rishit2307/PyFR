import numpy as np
from collections import defaultdict
# from sklearn.cluster import MiniBatchKMeans, KMeans
# from sklearn.preprocessing import StandardScaler
from cuml.cluster import KMeans
import cupy as cp
import time

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
		comm, rank, root = get_comm_rank_root()

		sect = 'solver-time-integrator'
		precision = cfg.get('backend', 'precision')
		if precision == 'double':
			self.fpdtype = np.float64
		else:
			self.fpdtype = np.float32

		self.jac_bsize = {}
		getypes = set(comm.allreduce(self.system.ele_types, op=mpi.SUM))

		for etype in getypes:
			self.jac_bsize[etype]=  cfg.getint(sect, f'jac-bsize-{etype}', 5000)

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
	def _get_jacinit_kerns(self, etp, col, *rs):
		celes = self.system.celes[etp, col]
		etype_ix = self.system.ele_types.index(etp)
		jac = self.jac[etype_ix]

		em = self.system.ele_banks[etype_ix]
		kerns = [self.backend.kernel('jacinit', 
				 *[em[r] for r in rs] +[jac, celes])]

		return kerns

	def _init_jac(self, etp, rs, stele, npt, v, col, afac, h):

		celes = self.system.celes
		if (etp, col) not in celes.keys():
			return
		bsize = self.jac_bsize[etp]

		kerns = self._get_jacinit_kerns(etp, col, *rs)
		self._bind_kerns(kerns, stele, bsize, npt, v, col, afac, h)

		self.backend.run_kernels(kerns)

	def _shuff_jac(self):
		jacinv, jac = self.jacinv, self.jac
		kerns = [self.backend.kernel('jacshuffle', *[jacinv[i], jac[i]])
				 for i in range(len(self.system.ele_types))]

		return kerns
	
	@memoize
	def _get_invjac_kerns(self, *arr, kmeans=False):

		kerns = [self.backend.kernel('getf3', *arr, kmeans=kmeans)]
		return kerns
		         
	def _inv_jac(self, etp, col, kmeans=False):

		eix = self.system.ele_types.index(etp)
		jacinv, jac = self.jacinv, self.jac
		P = self.Permut
		celes = self.system.celes[etp, col]

		kerns = self._get_invjac_kerns(*[jac[eix], jacinv[eix], 
								         P[eix], celes], kmeans=kmeans)
		
		return kerns

	def _set_jac_backend(self):
		backend = self.backend
		comm, rank, root = get_comm_rank_root()
		self.jac, self.jacinv = [], []
		self.Permut = []
		ecount = self.system.ecount

		for i, etype in enumerate(self.system.ele_types):

			nupts, nvars, neles = self.system.ele_shapes[i]
			jac_bsize = min(neles, self.jac_bsize[etype])

			jac = np.empty((ecount[etype], (nupts*nvars)**2),
							dtype=self.fpdtype)

			P = np.zeros((ecount[etype], nupts*nvars), 
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
	def mul_jac(self, *rs, inscales, outscales):
		jac = self.jacinv
		backend = self.backend
		kerns = [backend.kernel('jacmul', *[em[r] for r in rs]+[jac[i]], 
						        inscales=inscales, outscales=outscales)
				 for i, em in enumerate(self.system.ele_banks)]

		return kerns

	@memoize
	def mul_jac_kmeans(self, *rs, inscales, outscales):
		jacinv = self.jacinv
		backend = self.backend
		emap, cluster_neles = self.emap, self.cluster_neles
		stidx = self.stidx

		kerns = [backend.kernel('jacmul_clust', *[em[r] for r in rs]+[jacinv[i]]
						  		+ [emap[i]]+[cluster_neles[i], stidx[i]],
								inscales=inscales, outscales=outscales) 
								for i, em in enumerate(self.system.ele_banks)]

		return kerns

	def _eval_jac(self, tc, a, currstg, rcurr):
		rhs = self.system.rhs
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
			for eles in range(0, max(gneles[etype]), self.jac_bsize[etype]):
				for col in range(self.system.ncolours[etype]):
					for npt in range(nupts):
						for v in range(nvars):
							self._add(0, raux, 1, rcurr)
							self._addid([rcurr, raux],
										eles, self.jac_bsize[etype],
										npt, v, col, h, etype)

							rhs(tc, raux, raux)
							self._add(-a, raux, a, rcurr_rhs)
							self._init_jac(etype, [raux, rcurr], 
											eles, npt, v, col, 1, h)

					if (etype, col) in self.system.celes.keys():
						eix = self.system.ele_types.index(etype)
						neles = self.system.ele_shapes[eix][-1]
						kerns = self._inv_jac(etype, col, 
							                  kmeans=self.clustering)

						self._bind_kerns(kerns, eles, neles, self.jac_bsize[etype])
						self.backend.run_kernels(kerns)

				if self.clustering and etype in self.system.ele_types:
						eix = self.system.ele_types.index(etype)
						neles = self.system.ele_shapes[eix][-1]
						jacinv_cp = self._pyfr_to_cupy(self.jacinv[eix])
						if eles < neles:
							six = min(neles - eles, self.jac_bsize[etype])
							self._get_kmeans(eles, etype,
											 jacinv_cp[:six])
		if self.clustering:
			self._set_jac_kmeans()
		self.backend.wait()

	def _update_precision(self):
		jac_temp = []
		backend = self.backend
		del self.jac

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
		clust_neles = self.cluster_neles
		stidx, emap = self.stidx, self.emap
		comm, rank, root = get_comm_rank_root()

		for i, jack in enumerate(self.jacinv_tmp):
			# etp = self.system.ele_types[i]

			# np.save(f'./npy/jac_{etp}_{rank}', np.vstack(jack))
			jackb = backend.matrix(np.vstack(jack).shape, 
						  		   np.vstack(jack), 
								   dtype=self.jac_fpdtype)

			ci = np.array(self.cluster_neles[i])[None]
			ei = np.array(self.emap[i])[None]
			stix = np.cumsum(ci) - ci

			clust_neles[i] = backend.matrix(ci.shape, 
										    ci,
											dtype=backend.ixdtype)
			stidx[i] = backend.matrix(stix.shape, stix, 
									  dtype=backend.ixdtype)

			self.jacinv.append(jackb)
			emap[i] = backend.matrix(ei.shape, ei, 
							         dtype=backend.ixdtype)

			# np.save(f'./npy/ci_{etp}_{rank}', ci)
			# np.save(f'./npy/stix_{etp}_{rank}', stix)
			# np.save(f'./npy/ei_{etp}_{rank}', ei)

		del self.jacinv_tmp

	def _init_kmeans(self, cfg, sect):
		self.eleclust = cfg.getint(sect, 'eleclust', 128)
		self.cluster_neles = [[] for _ in range(len(self.system.ele_types))]
		self.emap = [[] for _ in range(len(self.system.ele_types))]
		self.stidx = [[] for _ in range(len(self.system.ele_types))]
		self.jacinv_tmp = [[] for _ in range(len(self.system.ele_types))]

	def _pyfr_to_cupy(self, parr):
		parr.__cuda_array_interface__  = pcarr_dict = {}
		pcarr_dict['shape'] = tuple(parr.datashape[1:])
		pcarr_dict['data'] = (parr.data, False)
		pcarr_dict['typestr'] = '<f8'

		carr = cp.asarray(parr)
		return carr

	def _get_kmeans(self, eles, etype, jacinv_cp):
		neles = jacinv_cp.shape[0]
		eix = self.system.ele_types.index(etype)
		comm, rank, root = get_comm_rank_root()

		if neles < self.jac_bsize[etype] and eles > 0:
			jacinv_cpu = cp.asnumpy(jacinv_cp)
			self.jacinv_tmp[eix].append(jacinv_cpu)
			self.cluster_neles[eix] += np.ones((neles), dtype=np.int32).tolist()
			self.emap[eix] += (np.arange(neles, dtype=np.int32) + eles).tolist()
			return

		print(f'rank is {rank}, etype is {etype}', flush=True)

		cluster_neles_tmp = []
		jactp = []
		eleclust = self.eleclust

		nclust = max(neles//eleclust, 1)

		mean = cp.mean(jacinv_cp, axis=0)
		std = cp.std(jacinv_cp, axis=0)
		jacinv_cp -= mean
		jacinv_cp /= std

		kmeans = KMeans(n_clusters=nclust, 
				        init='k-means++', 
						n_init='auto', 
						max_iter=300, 
						tol=0.0001, 
						random_state=None).fit(jacinv_cp)
		
		jacinv_cp *= std
		jacinv_cp += mean

		kml = cp.asnumpy(kmeans.labels_)
		jacinv_cpu = cp.asnumpy(jacinv_cp)

		print(f'rank is {rank}, kmeans b4 final done', flush=True)

		cluster_neles_tmp += np.bincount(kml, minlength=nclust).tolist()

		jactp_sum = np.zeros((nclust, jacinv_cpu.shape[1]))
		np.add.at(jactp_sum, kml, jacinv_cpu)
		jactp = (jactp_sum / np.array(cluster_neles_tmp)[:, None]).tolist()
		del jactp_sum
		self.jacinv_tmp[eix] += jactp

		#Handling emap_tmp (indices + offset)
		idx_sort = np.argsort(kml)
		self.emap[eix] += (idx_sort + eles).tolist()

		self.cluster_neles[eix] += cluster_neles_tmp

