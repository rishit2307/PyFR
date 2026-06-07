import numpy as np
from collections import defaultdict

from cuml.cluster import KMeans
import cupy as cp

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

	def _init_jac(self, etp, rs, npt, v, col, afac, h):

		celes = self.system.celes
		if (etp, col) not in celes.keys():
			return

		kerns = self._get_jacinit_kerns(etp, col, *rs)
		self._bind_kerns(kerns, npt, v, col, afac, h)

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

			jac = np.empty((ecount[etype], (nupts*nvars)**2),
							dtype=self.fpdtype)

			P = np.zeros((ecount[etype], nupts*nvars), 
						 dtype=self.backend.ixdtype)

			if self.clustering:
				jacinv = np.zeros((neles, (nupts*nvars)**2), 
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
	def _get_shuff_kerns(self, etype, rin):
		backend = self.backend
		eix = self.system.ele_types.index(etype)
		em = self.system.ele_banks[eix]

		kerns = [backend.kernel('shuffle', 
						       *[em[rin], self.shuff_arr[etype]]
								 + [self.emap[eix], 
								self.stidx[eix], self.stidx_pad[eix],
								self.cluster_id[eix]])
		        ]

		return kerns

	@memoize
	def mul_jac_kmeans(self, ruc, rout,
					  inscales, outscales):

		comm, rank, root = get_comm_rank_root()

		jacinv_clust = self.jacinv_clust
		be = self.backend

		emap, cneles_pad = self.emap, self.cluster_neles_pad
		stidx_pad = self.stidx_pad
		stidx, cneles = self.stidx, self.cluster_neles
		clustk = []

		for etype, jacc in jacinv_clust.items():
			eix = self.system.ele_types.index(etype)
			# np.save(f'/home/grads/r/rishit/code/Pyfr_venv_2/PyFR-Test-Cases/gpu_exp/npy/emap_{rank}_{etype}', self.emap[eix].get())
			# np.save(f'/home/grads/r/rishit/code/Pyfr_venv_2/PyFR-Test-Cases/gpu_exp/npy/stidx_{rank}_{etype}', self.stidx[eix].get())
			# np.save(f'/home/grads/r/rishit/code/Pyfr_venv_2/PyFR-Test-Cases/gpu_exp/npy/stidx_pad_{rank}_{etype}', self.stidx_pad[eix].get())
			# np.save(f'/home/grads/r/rishit/code/Pyfr_venv_2/PyFR-Test-Cases/gpu_exp/npy/cid_{rank}_{etype}', self.cluster_id[eix].get())
			# np.save(f'/home/grads/r/rishit/code/Pyfr_venv_2/PyFR-Test-Cases/gpu_exp/npy/cluster_neles_{rank}_{etype}', self.cluster_neles[eix].get())
			# np.save(f'/home/grads/r/rishit/code/Pyfr_venv_2/PyFR-Test-Cases/gpu_exp/npy/cluster_neles_pad_{rank}_{etype}', self.cluster_neles_pad[eix].get())
			shuffk = self._get_shuff_kerns(etype, ruc)
			clustk += shuffk

			em = self.system.ele_banks[eix]
			regs = [self.shuff_arr[etype], em[rout]]
			arr = regs + [jacc]

			# Get kernels
			kern = be.kernel('jacmul_clust', *arr, 
							emap=emap[eix],
							clust_neles=cneles[eix],
							clust_neles_pad=cneles_pad[eix],
							stidx_pad=stidx_pad[eix],
							stidx=stidx[eix],
							inscales=inscales, 
							outscales=outscales)

			clustk.append(kern)
		return clustk

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
			for col in range(self.system.ncolours[etype]):
				for npt in range(nupts):
					for v in range(nvars):
						self._add(0, raux, 1, rcurr)
						self._addid([rcurr, raux],
									npt, v, col, h, etype)

						rhs(tc, raux, raux)
						self._add(-a, raux, a, rcurr_rhs)
						self._init_jac(etype, [raux, rcurr], 
										npt, v, col, 1, h)

				if (etype, col) in self.system.celes.keys():
					eix = self.system.ele_types.index(etype)
					neles = self.system.ele_shapes[eix][-1]
					kerns = self._inv_jac(etype, col, 
											kmeans=self.clustering)

					self.backend.run_kernels(kerns)

			if self.clustering and etype in self.system.ele_types:
				eix = self.system.ele_types.index(etype)
				jacinv_cp = self._pyfr_to_cupy(self.jacinv[eix])
				self._get_kmeans(etype, jacinv_cp)

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
		backend = self.backend
		clust_neles = self.cluster_neles
		clust_neles_pad = self.cluster_neles_pad
		stidx, emap = self.stidx, self.emap
		stidx_pad = self.stidx_pad
		cid = self.cluster_id
		comm, rank, root = get_comm_rank_root()

		commit = lambda arr: backend.matrix(arr.shape,
									        arr, 
											dtype=backend.ixdtype)

		for i, etype in enumerate(self.system.ele_types):
			eix = self.system.ele_types.index(etype)
			jacclust = self.jacinv_clust[etype]
			jackb = backend.matrix(jacclust.shape, 
									jacclust, 
									dtype=self.jac_fpdtype)

			ci = np.array(self.cluster_neles[i])[None]
			ei = np.array(self.emap[i])[None]
			stix = np.cumsum(ci) - ci
			cidi = np.array(cid[i])[None]


			ci_pad = np.array(self.cluster_neles_pad[i])[None]
			stix_pad = np.cumsum(ci_pad) - ci_pad

			clust_neles[i] = commit(ci)
			clust_neles_pad[i] = commit(ci_pad)

			stidx[i] = commit(stix)
			stidx_pad[i] = commit(stix_pad)
			emap[i] = commit(ei)
			cid[i] = commit(cidi)

			self.jacinv_clust[etype] = jackb


			# np.save(f'./npy/ci_{etp}_{rank}', ci)
			# np.save(f'./npy/stix_{etp}_{rank}', stix)
			# np.save(f'./npy/ei_{etp}_{rank}', ei)

	def _init_kmeans(self, cfg, sect):
		self.eleclust = cfg.getint(sect, 'eleclust', 128)
		self.cluster_neles = [[] for _ in range(len(self.system.ele_types))]
		self.emap = [[] for _ in range(len(self.system.ele_types))]
		self.cluster_neles_pad = [[] for _ in range(len(self.system.ele_types))]
		self.stidx = [[] for _ in range(len(self.system.ele_types))]
		self.stidx_pad = [[] for _ in range(len(self.system.ele_types))]
		self.shuff_arr = {}
		self.cluster_id = [[] for _ in range(len(self.system.ele_types))]
		self.jacinv_clust = {}

		# List of candidate element size
		self.eblk = [16, 32, 64]

	def _pyfr_to_cupy(self, parr):
		parr.__cuda_array_interface__  = pcarr_dict = {}
		pcarr_dict['shape'] = tuple(parr.datashape[1:])
		pcarr_dict['data'] = (parr.data, False)
		pcarr_dict['typestr'] = '<f8'

		carr = cp.asarray(parr)
		return carr

	def _get_kmeans(self, etype, jacinv_cp):

		eix = self.system.ele_types.index(etype)
		comm, rank, root = get_comm_rank_root()
		nupts, nvars, neles = self.system.ele_shapes[eix]

		print(f'rank is {rank}, etype is {etype}', flush=True)

		jactp = []
		eblk = self.eblk
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
						tol=1e-4, 
						random_state=None).fit(jacinv_cp)

		jacinv_cp *= std
		jacinv_cp += mean

		kml = cp.asnumpy(kmeans.labels_)
		kmlas = np.argsort(kml)
		jacinv_cpu = cp.asnumpy(jacinv_cp)

		print(f'rank is {rank}, kmeans b4 final done', flush=True)

		# Sorted KMeans
		bc = np.bincount(kml, minlength=nclust)
		bcsix = np.argsort(bc)
		bcs = bc[bcsix]

		# Clustered Jacobians
		jactp_sum = np.zeros((nclust, jacinv_cpu.shape[1]))
		np.add.at(jactp_sum, kml, jacinv_cpu)
		jactp = (jactp_sum / bc[:, None])
		jactp = jactp[bcsix]
		del jactp_sum
		self.jacinv_clust[etype] = jactp

		# Split labels and its indices
		csum = np.cumsum(bc)[:-1]
		karggroup = np.split(kmlas, csum)

		# Assign emaps and cluster ids
		for i, (carg, c) in enumerate(zip(bcsix, bcs)):
			self.cluster_id[eix].extend([i]*c)
			self.emap[eix].extend(karggroup[carg].tolist())

		self.cluster_neles[eix] += bcs.tolist()

		# Get padded element counts
		ix = np.searchsorted(eblk, self.cluster_neles[eix])
		ix[ix >= len(eblk)] = len(eblk) - 1
		pad = -(-bcs // np.array(eblk)[ix])
		self.cluster_neles_pad[eix] += (pad*np.array(eblk)[ix]).tolist()

		# Assign storage for shuffled array
		pad_eles = np.sum(self.cluster_neles_pad[eix])
		shufarr = np.zeros((nupts*nvars, pad_eles))
		self.shuff_arr[etype] = self.backend.matrix(shufarr.shape, 
											       shufarr)


