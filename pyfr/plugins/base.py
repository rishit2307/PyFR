import functools as ft
import re
import shlex
import os
import numpy as np
from pytools import prefork
from collections import defaultdict
import pickle
from pyfr.inifile import Inifile
from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.quadrules import get_quadrule
from pyfr.regions import parse_region_expr
from pyfr.util import memoize, match_paired_paren, subclass_where
from pyfr.regions import BoundaryRegion, ConstructiveRegion
from pyfr.shapes import BaseShape
from pyfr.writers.native import NativeWriter

def cli_external(meth):
    @ft.wraps(meth)
    def newmeth(cls, args):
        return meth(cls(), args)

    return classmethod(newmeth)


def init_csv(cfg, cfgsect, header, *, filekey='file', headerkey='header'):

    from configparser import NoOptionError

    # Determine the file path
    try:
        fname = cfg.get(cfgsect, filekey)
    except NoOptionError:
        basedir = cfg.getpath(cfgsect, 'basedir', '.', abs=True)
        basename = cfg.get(cfgsect, 'basename')
        fname = os.path.join(basedir, basename)

    # Append the '.csv' extension
    if not fname.endswith('.csv'):
        fname += '.csv'

    # Open for appending
    outf = open(fname, 'a')

    # Output a header if required
    if outf.tell() == 0 and cfg.getbool(cfgsect, headerkey, True):
        print(header, file=outf)

    # Return the file
    return outf


def region_data(cfg, cfgsect, mesh, rallocs):
    region = cfg.get(cfgsect, 'region', '*')

    # Determine the element types in our partition
    sptptn = f'spt_(.+?)_p{rallocs.prank}$'
    etypes = [m[1] for f in mesh if (m := re.match(sptptn, f))]

    # All elements
    if region == '*':
        return {etype: slice(None) for etype in etypes}
    # All elements inside some region
    else:
        comm, rank, root = get_comm_rank_root()

        # Parse the region expression and obtain the element set
        rgn = parse_region_expr(region)
        eset = rgn.interior_eles(mesh, rallocs)

        # Ensure the region is not empty
        if not comm.reduce(bool(eset), op=mpi.LOR, root=root) and rank == root:
            raise ValueError(f'Empty region {region}')

        return {etype: np.unique(eidxs).astype(np.int32)
                for etype, eidxs in sorted(eset.items())}


def surface_data(cfg, cfgsect, mesh, rallocs, intg):
    surf = cfg.get(cfgsect, 'surface')
    region_eset = defaultdict(list)
    surfbcnames = []
    comm, rank, root = get_comm_rank_root()
    # Geometric region
    if '(' in surf:
        rgncls = ConstructiveRegion(surf)
        region_eset = rgncls.interior_eles(mesh, rallocs)
    # Boundary region
    else:
        for rg in surf.split(','):
            bndnm = rg.strip()
            rgncls = BoundaryRegion(bndnm)
            surfbcnames.append(bndnm)
            lrgeset = rgncls.interior_eles(mesh, rallocs)
            for etype, eidlist in lrgeset.items():
                for eid in eidlist:
                    if not eid in region_eset[etype]: 
                        region_eset[etype].append(eid)
    
    
    print(f'region_eset is {region_eset}')
    expr = cfg.get(cfgsect, 'endcaps', None)
    endcaps = []
    exclbcs = cfg.get(cfgsect, 'exclbcs', 'none').split(',')
    exclbcs = [bc.strip() for bc in exclbcs]

    # Factor out the individual endcap region expressions
    from ast import literal_eval
    expr = re.sub(
        r'(\w+)\((' + match_paired_paren('()') + r')\)',
        lambda m: endcaps.append(literal_eval(m.groups()[1])) or 
        f'e{len(endcaps) - 1}',
        expr
    )
    elemap = intg.system.ele_map
    eset, surf_eset = _prep_surface(mesh, rallocs, elemap, region_eset, exclbcs=exclbcs, 
                    inclbcs=surfbcnames, endcaps=endcaps,
                    inclperiodic=False)


    # Parse the surface expression and obtain the element set
    # if '+' in surf and '(' not in surf:
    #     surf = surf.split('+')
    #     eset = {}
    #     for s in surf:
    #         rgn = parse_region_expr(s)
    #         for k, v in rgn.surface_faces(mesh, rallocs):
    #             if k in eset.keys():
    #                 eset[k] += v
    #             else:
    #                 eset[k] = v
            
        

        
    # else:
    #     rgn = parse_region_expr(surf)
    # rgn = parse_region_expr(surf)

    # eset = rgn.surface_faces(mesh, rallocs)
    

    # Ensure the surface is not empty
    # if not comm.reduce(bool(eset), op=mpi.LOR, root=root) and rank == root:
    #     raise ValueError(f'Empty surface {surf}')

    # _write_fwhsurf_geo(intg, surf_eset, eset)

    return {etype: np.unique(eidxs).astype(np.int32)
            for etype, eidxs in sorted(eset.items())}

def _write_fwhsurf_geo(intg, eset, inters_eidxs):
    comm, rank, root = get_comm_rank_root()

    # If we are the root rank then prepare the metadata
    if rank == root:
        stats = Inifile()
        stats.set('data', 'prefix', 'soln')
        intg.collect_stats(stats)
        metadata = dict(intg.cfgmeta,
                        stats=stats.tostr(),
                        mesh_uuid=intg.mesh_uuid)
    else:
        metadata = None

    # Fetch and (if necessary) subset the solution
    ele_regions, ele_region_data = [], {}
    for etype, eidxs in sorted(eset.items()):
        doff = intg.system.ele_types.index(etype)
        darr = np.unique(eidxs).astype(np.int32)

        ele_regions.append((doff, etype, darr))
        ele_region_data[f'{etype}_idxs'] = darr

    data = dict(ele_region_data)
    for idx, etype, rgn in ele_regions:
        data[etype] = intg.soln[idx][..., rgn].astype(np.float64)

    # Write out the file
    vtuwriter = NativeWriter(intg, '.', 'fwhsurf', 'soln')
    fwhmeshfname = vtuwriter.write(data, intg.tcurr, metadata)

    #Debug write surface faces using vtk
    vtumfname = os.path.splitext(fwhmeshfname)[0]
    VTUSurfWriter(intg, inters_eidxs).write(vtumfname)
    #end debug


def _prep_surface(mesh, rallocs, elemap, eset, 
                  exclbcs=[], inclbcs=[], endcaps=[], 
                  inclperiodic=False):
    
    sfaces = set()
    eidxs = defaultdict(list)
    surfeset = set()
    # Begin by assuming all faces of all elements are on the surface
    for etype, eidls in eset.items():
        nfaces = len(subclass_where(BaseShape, name=etype).faces)
        sfaces.update((etype, i, j) for i in eidls for j in range(nfaces))

    #Eliminate Endcap faces if any
    endcap_sfaces = _sfaces_endcaps(elemap, sfaces, endcaps)
    sfaces.difference_update(endcap_sfaces)
    
    # Process interior faces
    con = mesh[f'con_p{rallocs.prank}'].T
    for ll, rr in con.astype('U4,i4,i1,i2').tolist():
        l, r = ll[:-1], rr[:-1]

        if l in sfaces and r in sfaces:
            #Add periodic
            if ll[-1] != 0 and inclperiodic:
                # add the first element info
                etype, eidx, fidx = l
                retype, reidx, rfidx = r
                eidxs[etype, fidx].append(eidx)
                surfeset.add((etype, eidx))
                # add the second element info as well
                etype, eidx, fidx = r
                retype, reidx, rfidx = l
                eidxs[etype, fidx].append(eidx)
                surfeset.add((etype, eidx))
            else: #eliminate internal conn faces
                sfaces.difference_update([l, r])

        #Add interior   
        elif (l in sfaces or r in sfaces) and not inclbcs:
            etype, eidx, fidx = l if l in sfaces else r
            retype, reidx, rfidx = r if l in sfaces else l
            eidxs[etype, fidx].append(eidx)
            surfeset.add((etype, eidx))

    # Process physical boundaries
    for f in mesh:
        if (m := re.match(f'bcon_(.+?)_p{rallocs.prank}$', f)):
            bcon = mesh[f][['f0', 'f1', 'f2']].astype('U4,i4,i1')
            bcname = m.group(1)
            # Group included physical boundary faces 
            if (bcname in inclbcs) or not (inclbcs or bcname in exclbcs): 
                for f in bcon.tolist():
                    if f in sfaces:
                        etype, eidx, fidx = f
                        eidxs[etype, fidx].append(eidx)
                        surfeset.add((etype, eidx))
            elif bcname in exclbcs: #eliminate excluded boundary faces
                sfaces.difference_update(bcon.tolist())

    comm, rank, root = get_comm_rank_root()
    reqs, bufs = [], []

    # Next, consider faces on partition boundaries
    for p in rallocs.prankconn[rallocs.prank]:
        con = mesh[f'con_p{rallocs.prank}p{p}']
        con = con.astype('U4,i4,i1,i2').tolist()

        # See which of these faces are on the surface boundary
        lb = np.array([c[:-1] in sfaces for c in con])
        rb = np.empty_like(lb)

        # Exchange this information with our neighbour
        reqs.append(comm.Isend(lb, rallocs.pmrankmap[p]))
        reqs.append(comm.Irecv(rb, rallocs.pmrankmap[p]))

        bufs.append((con, lb, rb, rallocs.pmrankmap[p]))

    # Wait for the exchanges to finish
    mpi.Request.Waitall(reqs)

    # Process mpi faces
    for cons, lconds, rconds, rhsmrank in bufs:
        for lfacecond, rfacecond, iface in zip(lconds, rconds, cons):
            etype, eidx, fidx, fflag = iface
        
            if lfacecond & rfacecond:
                #handle mpi & periodic faces
                if fflag != 0 and inclperiodic:
                    eidxs[etype, fidx].append(eidx)
                    surfeset.add((etype, eidx))
                else: #eliminate shared faces
                    sfaces.difference_update(iface[:-1])
            
            elif not inclbcs:
                #handle fwh active ranks
                if lfacecond:
                    eidxs[etype, fidx].append(eidx)
                    surfeset.add((etype, eidx))
                #handle fwh sharing/edge ranks
    
    _surf_eset = defaultdict(list)

    for etype, eidx in surfeset:
        _surf_eset[etype].append(eidx)

    return {k: sorted(v) for k, v in eidxs.items()}, _surf_eset
def _sfaces_endcaps(elemap, sfaces, endcaps):
    endeqn = []
    endtol = []
    mind = [[]]*len(endcaps) #debug
    
    for e in endcaps:
        #Construct the endcaps equations
        ec = np.array(e[:3])
        nv = np.cross(ec[2] - ec[0], ec[1] - ec[0])
        endtol.append(e[3]*np.linalg.norm(ec[0]))
        nv /= np.linalg.norm(nv)
        endd = - nv*ec[0]
        endeqn.append([nv, endd])

    ecpsfaces= set()

    #Identify end-cap faces
    for etype, eidx, fidx in sfaces:
        eles = elemap[etype]
        
        facefpts = eles.basis.facefpts[fidx]
        nfacefpts = eles.basis.nfacefpts[fidx]
        fplcs = eles.plocfpts[facefpts, eidx] 
        fcent = np.sum(fplcs, axis=0)/nfacefpts

        for i, (eq, etol) in enumerate(zip(endeqn, endtol)):
            nv, endd = eq
            #distance to endcap plane
            d = np.linalg.norm(nv*fcent + endd)
            mind[i].append(d)
            #add to endcap faces if close to the endcap
            if d <= etol:
                ecpsfaces.add((etype, eidx, fidx))
    #debug
    if endcaps:
        comm, rank, root = get_comm_rank_root()
        nfaces0 = len(sfaces) #debug
        sfaces.difference_update(ecpsfaces)
        nfaces1 = len(sfaces)
        ntotfaces0 = comm.reduce(nfaces0, root=root)
        ntotfaces1 = comm.reduce(nfaces1, root=root)

        mind = [min(dd) for dd in mind if dd]
        if mind:
            print(f'rank {rank}, min distance: {np.round(mind, 5)},'
                f'\tnsetfaces: '
                f'{nfaces0} - {len(ecpsfaces)} =  {nfaces1}')
        if rank == root:
            print('\n')
            print(f'nfaces {ntotfaces0} --> {ntotfaces1}')
            for i, (eq, ec, etol) in enumerate(
                zip(endeqn, endcaps, endtol)):
                print(f'end-cap eq{i}: {eq[0][0]}(x - {ec[0][0]})'
                    f' + {eq[0][1]}(y - {ec[0][1]})'
                    f' + {eq[0][2]}(z - {ec[0][2]}), '
                    f'\tdistance tol = {np.round(etol,5)}')
            print('\n')
    #enddebug

    return ecpsfaces

class BasePlugin:
    name = None
    systems = None
    formulations = None

    def __init__(self, intg, cfgsect, suffix=None):
        self.cfg = intg.cfg
        self.cfgsect = cfgsect

        self.suffix = suffix

        self.ndims = intg.system.ndims
        self.nvars = intg.system.nvars

        # Tolerance for time comparisons
        self.tol = 5*intg.dtmin

        # Check that we support this particular system
        if not ('*' in self.systems or intg.system.name in self.systems):
            raise RuntimeError(f'System {intg.system.name} not supported by '
                               f'plugin {self.name}')

        # Check that we support this particular integrator formulation
        if intg.formulation not in self.formulations:
            raise RuntimeError(f'Formulation {intg.formulation} not '
                               f'supported by plugin {self.name}')

        # Check that we support dimensionality of simulation
        if intg.system.ndims not in self.dimensions:
            raise RuntimeError(f'Dimensionality of {intg.system.ndims} not '
                               f'supported by plugin {self.name}')

    def __call__(self, intg):
        pass

    def serialise(self, intg):
        return {}


class BaseSolnPlugin(BasePlugin):
    prefix = 'soln'


class BaseSolverPlugin(BasePlugin):
    prefix = 'solver'


class BaseCLIPlugin:
    name = None

    @classmethod
    def add_cli(cls, parser):
        pass


class PostactionMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.postact = None
        self.postactaid = None
        self.postactmode = None

        if self.cfg.hasopt(self.cfgsect, 'post-action'):
            self.postact = self.cfg.getpath(self.cfgsect, 'post-action')
            self.postactmode = self.cfg.get(self.cfgsect, 'post-action-mode',
                                            'blocking')

            if self.postactmode not in {'blocking', 'non-blocking'}:
                raise ValueError('Invalid post action mode')

    def __del__(self):
        if getattr(self, 'postactaid', None) is not None:
            prefork.wait(self.postactaid)

    def _invoke_postaction(self, intg, **kwargs):
        comm, rank, root = get_comm_rank_root()

        # If we have a post-action and are the root rank then fire it
        if rank == root and self.postact:
            # If a post-action is currently running then wait for it
            if self.postactaid is not None:
                prefork.wait(self.postactaid)

            # Prepare the command line
            cmdline = shlex.split(self.postact.format_map(kwargs))

            # Invoke
            if self.postactmode == 'blocking':
                intg.abort |= bool(prefork.call(cmdline))
            else:
                self.postactaid = prefork.call_async(cmdline)


class RegionMixin:
    def __init__(self, intg, *args, **kwargs):
        super().__init__(intg, *args, **kwargs)

        # Parse the region
        ridxs = region_data(self.cfg, self.cfgsect, intg.system.mesh,
                            intg.rallocs)

        # Generate the appropriate metadata arrays
        self._ele_regions, self._ele_region_data = [], {}
        for etype, eidxs in ridxs.items():
            doff = intg.system.ele_types.index(etype)
            self._ele_regions.append((doff, etype, eidxs))

            if not isinstance(eidxs, slice):
                self._ele_region_data[f'{etype}_idxs'] = eidxs


class SurfaceMixin:
    def _surf_region(self, intg):
        # Parse the region
        sidxs = surface_data(intg.cfg, self.cfgsect, intg.system.mesh,
                             intg.rallocs, intg)

        # Generate the appropriate metadata arrays
        ele_surface, ele_surface_data = [], {}
        for (etype, face), eidxs in sidxs.items():
            doff = intg.system.ele_types.index(etype)
            ele_surface.append((doff, etype, face, eidxs))

            if not isinstance(eidxs, slice):
                ele_surface_data[f'{etype}_f{face}_idxs'] = eidxs
        return ele_surface, ele_surface_data

    @memoize
    def _surf_quad(self, itype, proj, flags=''):
        # Obtain quadrature info
        rname = self.cfg.get(f'solver-interfaces-{itype}', 'flux-pts')

        # Quadrature rule (default to that of the solution points)
        qrule = self.cfg.get(self.cfgsect, f'quad-pts-{itype}', rname)
        try:
            qdeg = self.cfg.getint(self.cfgsect, f'quad-deg-{itype}')
        except NoOptionError:
            qdeg = self.cfg.getint(self.cfgsect, 'quad-deg')

        # Get the quadrature rule
        q = get_quadrule(itype, qrule, qdeg=qdeg, flags=flags)

        # Project its points onto the provided surface
        pts = np.atleast_2d(q.pts.T)
        return np.vstack(np.broadcast_arrays(*proj(*pts))).T, q.wts


class VTUSurfWriter(object):

    vtkfile_version = '2.1'
    vtk_types = dict(line=3, tri=5, quad=9, tet=10, pyr=14, pri=13, hex=12)
    vtk_nodes = dict(tri=3, quad=4, tet=4, pyr=5, pri=6, hex=8)
    # number of first order nodes/faces per element
    _petype_focount_map = {'line': 2, 'tri': 3, 'quad': 4,
                       'tet': 4, 'pyr': 5, 'pri': 6, 'hex': 8} 
    # map of first order face nodes
    # To generate this fnmap, we use vtk_nodemaps and then apply gmsh_fnmap 
    # to extract the correct first order face node ordering.
    # Some fnmap face points may need to be flipped to maintain 
    # a c.c.w or c.c. 

    # face node counting
    _petype_fnmap = {
        ('tri',  3 ):  {'line': [[0, 1], [1, 2], [2, 0]]},
        ('tri',  6 ):  {'line': [[5, 0], [0, 2], [2, 5]]},
        ('tri',  10 ):  {'line': [[9, 0], [0, 3], [3, 9]]},
        ('tri',  15 ):  {'line': [[14, 0], [0, 4], [4, 14]]},
        ('quad',  4 ):  {'line': [[0, 1], [1, 3], [3, 2], [2, 0]]},
        ('quad',  9 ):  {'line': [[0, 2], [2, 8], [8, 6], [6, 0]]},
        ('quad',  16 ):  {'line': [[0, 3], [3, 15], [15, 12], [12, 0]]},
        ('quad',  25 ):  {'line': [[0, 4], [4, 24], [24, 20], [20, 0]]},
        ('tet',  4 ):  {'tri': [[1, 0, 3], [3, 0, 2], [2, 1, 3], [0, 1, 2]]},
        ('tet',  10 ):  {'tri': [[2, 0, 9], [9, 0, 5], [5, 2, 9], [0, 2, 5]]},
        ('tet',  20 ):  
            {'tri': [[3, 0, 19], [19, 0, 9], [9, 3, 19], [0, 3, 9]]},
        ('tet',  35 ): 
            {'tri': [[4, 0, 34], [34, 0, 14], [14, 4, 34], [0, 4, 14]]},
        ('hex',  8 ):  
            {'quad': [[0, 1, 3, 2], [0, 1, 5, 4], [1, 3, 7, 5], [3, 2, 6, 7],
                     [0, 2, 6, 4], [4, 5, 7, 6]]},
        ('hex',  27 ):  
            {'quad': [[0, 2, 8, 6], [0, 2, 20, 18], [2, 8, 26, 20], 
                     [8, 6, 24, 26], [0, 6, 24, 18], [18, 20, 26, 24]]},
        ('hex',  64 ):  
            {'quad': [[0, 3, 15, 12], [0, 3, 51, 48], [3, 15, 63, 51], 
                     [15, 12, 60, 63], [0, 12, 60, 48], [48, 51, 63, 60]]},
        ('hex',  125 ):  
            {'quad': [[0, 4, 24, 20], [0, 4, 104, 100], [4, 24, 124, 104],
                     [24, 20, 120, 124], [0, 20, 120, 100], 
                     [100, 104, 124, 120]]},
        ('pri',  6 ):  
            {'quad': [[0, 1, 4, 3], [1, 2, 5, 4], [0, 3, 5, 2]], 
             'tri': [[0, 2, 1], [3, 4, 5]]},
        ('pri',  18 ): 
            {'quad': [[0, 2, 14, 12], [2, 5, 17, 14], [0, 12, 17, 5]], 
             'tri': [[0, 5, 2], [12, 14, 17]]},
        ('pri',  40 ):  
            {'quad': [[0, 3, 33, 30], [3, 9, 39, 33], [0, 30, 39, 9]],
             'tri': [[0, 9, 3], [30, 33, 39]]},
        ('pri',  75 ):  
            {'quad': [[0, 4, 64, 60], [4, 14, 74, 64], [0, 60, 74, 14]],
             'tri': [[0, 14, 4], [60, 64, 74]]},
        ('pyr',  5 ):  
            {'quad': [[2, 3, 1, 0]], 'tri': [[0, 1, 4], [1, 3, 4], [3, 2, 4],
                     [0, 4, 2]]},
        ('pyr',  14 ):  
            {'quad': [[6, 8, 2, 0]], 'tri': [[0, 2, 13], [2, 8, 13], 
                     [8, 6, 13], [0, 13, 6]]},
        ('pyr',  30 ): 
            {'quad': [[12, 15, 3, 0]], 'tri': [[0, 3, 29], [3, 15, 29], 
                     [15, 12, 29], [0, 29, 12]]},
        ('pyr',  55 ):  
            {'quad': [[20, 24, 4, 0]], 'tri': [[0, 4, 54], [4, 24, 54], 
                     [24, 20, 54], [0, 54, 20]]},
    }
    # offset to get local fidx/fnum inside each facetype map
    _fnum_offset = {'tri' : {'line': 0},
                    'quad': {'line': 0},
                    'tet' : {'tri': 0},
                    'hex' : {'quad': 0},
                    'pri' : {'quad': -2, 'tri': 0}, 
                    'pyr' : {'quad': 0, 'tri': -1}}
    # reverse map from face index to face type
    fnum_pftype_map = {
            'tri' : { 0: 'line', 1: 'line', 2: 'line'}, 
            'quad': { 0: 'line', 1: 'line', 2: 'line', 3: 'line'}, 
            'tet' : { 0: 'tri', 1: 'tri', 2: 'tri', 3: 'tri'}, 
            'hex' : { 0: 'quad', 1: 'quad', 2: 'quad', 3: 'quad', 4: 'quad',
                                                                5: 'quad'},
            'pri' : { 0: 'tri', 1: 'tri', 2: 'quad', 3: 'quad', 4: 'quad'}, 
            'pyr' : { 0: 'quad', 1: 'tri', 2: 'tri', 3: 'tri', 4: 'tri'} 
    }

    def __init__(self, intg, eidxs, fieldvars=None, fielddata=None):

        self._vtufnodes = defaultdict(list) # fwh face to vtu nodes set

        self.mesh = intg.system.mesh
        self.rallocs = intg.rallocs
        self._eidxs = eidxs
        self.ndims = intg.system.ndims
        # Output data type
        self.fpdtype = intg.backend.fpdtype

        if fieldvars:
            self._vtk_vars = fieldvars.extend(('Partition', 'r'))
        else:
            self._vtk_vars = [('Partition', 'r')]
        self._vtk_fields = fielddata if fielddata else []

        self._prepare_vtufnodes()

    def write(self,fname):
        fname = f'{fname}.vtu'
        self._write_vtu_out(fname)

    def _prepare_vtufnodes(self): 
        for (etype, fidx), eidxlist in self._eidxs.items():
            lmesh = self.mesh[f'spt_{etype}_p{self.rallocs.prank}']
            pts = np.swapaxes(lmesh, 0, 1)
            pftype = self.fnum_pftype_map[etype][fidx]
            fidx = int(fidx) + self._fnum_offset[etype][pftype]
            for eidx in eidxlist:
                nelemnodes = pts[eidx].shape[0]
                nidx = self._petype_fnmap[etype, nelemnodes][pftype][fidx]
                self._vtufnodes[pftype].append(pts[eidx][nidx, :])

    def _prepare_vtufnodes_info(self):
        self.partsdata = {}
        comm, rank, root = get_comm_rank_root()
        info = defaultdict(dict)
        for k, v in self._vtufnodes.items():
            npts, ncells, names, types, comps, sizes = self._get_array_attrs(k)
            info[k]['vtu_attr'] = [names, types, comps, sizes]
            info[k]['mesh_attr'] = [npts, ncells]
            info[k]['shape'] = np.asarray(v).shape 
            info[k]['dtype'] = np.asarray(v).dtype.str
            #fileddata
            psize = info[k]['shape'][0]*info[k]['shape'][1]
            self.partsdata[k] = np.tile(rank, psize) 
            info[k]['parts_shape'] = self.partsdata[k].shape
            info[k]['parts_dtype'] = self.partsdata[k].dtype.str
        return info

    def _write_vtu_out(self,fname):

        comm, rank, root = get_comm_rank_root()
        # prepare nodes info for each rank
        info = self._prepare_vtufnodes_info()

        # Communicate and prepare data for writing
        if rank != root:
            # Send the info about our data points to the root rank
            comm.gather(info, root=root)
            # Send the data points itself
            for etype, arrs in self._vtufnodes.items():
                comm.Send(np.array(arrs).astype(info[etype]['dtype']), 
                                                                root, tag=52)
            # Send field data one by one
            for etype in self.partsdata:
                comm.Send(self.partsdata[etype], root, tag=53)
        #root
        else:
            # Collect info about what remote ranks want to write 
            ginfo = comm.gather({}, root)
            # Update the info and receive the node arrays
            vpts_global = {}
            parts_global = {}
            # root data first
            for etype in info:
                vpts_global[etype] = np.array(self._vtufnodes[etype])
                parts_global[etype] = self.partsdata[etype]

            # update info and receive/stack nodes from other ranks
            for mrank, minfo in enumerate(ginfo):
                for etype, vinfo in minfo.items():
                    if etype in info:
                        info[etype]['vtu_attr'][3] = [sum(x) for x in 
                                        zip(info[etype]['vtu_attr'][3], 
                                                    vinfo['vtu_attr'][3])]
                        info[etype]['mesh_attr'] = [sum(x) for x in
                                        zip(info[etype]['mesh_attr'],
                                                    vinfo['mesh_attr'])]
                        shapes = [x for x in info[etype]['shape']]
                        shapes[0] += vinfo['shape'][0]
                        info[etype]['shape'] = tuple(shapes)
                        pshapes = [x for x in info[etype]['parts_shape']]
                        pshapes[0] += vinfo['parts_shape'][0]
                        info[etype]['parts_shape'] = tuple(pshapes)
                    else:
                        info[etype] = vinfo
                    varr = np.empty(vinfo['shape'], dtype=vinfo['dtype'])
                    comm.Recv(varr, mrank, tag=52)
                    if etype in vpts_global:
                        vgpts = vpts_global[etype]
                        vpts_global[etype] = np.vstack((vgpts, varr)) 
                    else:
                        vpts_global[etype] = varr

                    parr = np.empty(vinfo['parts_shape'], 
                                            dtype=vinfo['parts_dtype'])
                    comm.Recv(parr, mrank, tag=53)
                    if etype in parts_global:
                        gprts = parts_global[etype]
                        parts_global[etype] = np.hstack((gprts, parr))
                    else:
                        parts_global[etype] = parr

            # Writing
            write_s_to_fh = lambda s: fh.write(s.encode())

            with open(fname, 'wb') as fh:
                write_s_to_fh('<?xml version="1.0" ?>\n<VTKFile '
                            'byte_order="LittleEndian" '
                            'type="UnstructuredGrid" '
                            f'version="{self.vtkfile_version}">\n'
                            '<UnstructuredGrid>\n')

                # Running byte-offset for appended data
                off = 0
                # Header
                for etype in info:
                    off = self._write_serial_header(fh, info[etype], off)
                write_s_to_fh('</UnstructuredGrid>\n'
                            '<AppendedData encoding="raw">\n_')
                # Data
                for etype in info:
                    self._write_data(fh, etype, vpts_global[etype], 
                                                        parts_global[etype])

                write_s_to_fh('\n</AppendedData>\n</VTKFile>')

        # Wait for the root rank to finish writing
        comm.barrier()

    def _write_darray(self, array, vtuf, dtype):
        array = array.astype(dtype)
        np.uint32(array.nbytes).tofile(vtuf)
        array.tofile(vtuf)

    def _process_name(self, name):
        return re.sub(r'\W+', '_', name)

    def _write_serial_header(self, vtuf, info, off):
        names, types, comps, sizes = info['vtu_attr']
        npts, ncells = info['mesh_attr']

        write_s = lambda s: vtuf.write(s.encode())

        write_s(f'<Piece NumberOfPoints="{npts}" NumberOfCells="{ncells}">\n')
        write_s('<Points>\n')

        # Write vtk DaraArray headers
        for i, (n, t, c, s) in enumerate(zip(names, types, comps, sizes)):
            write_s(f'<DataArray Name="{self._process_name(n)}" type="{t}" '
                    f'NumberOfComponents="{c}" '
                    f'format="appended" offset="{off}"/>\n')

            off += 4 + s

            # Write ends/starts of vtk file objects
            if i == 0:
                write_s('</Points>\n<Cells>\n')
            elif i == 3:
                write_s('</Cells>\n<PointData>\n')

        # Write end of vtk element data
        write_s('</PointData>\n</Piece>\n')

        # Return the current offset
        return off

    def _write_data(self, vtuf, mk, vpts, parts):
        fpdtype = self.fpdtype 
        vpts = vpts.astype(fpdtype)
        neles = vpts.shape[0]
        nfopts = self._petype_focount_map[mk]
        fopts = np.arange(nfopts)

        # Append dummy z dimension for points in 2D
        if self.ndims == 2:
            vpts = np.pad(vpts, [(0, 0), (0, 0), (0, 1)], 'constant')
        # Write mesh points
        self._write_darray(vpts, vtuf, fpdtype) # simple nodes writer

        # Prepare VTU cell-node connectivity arrays
        vtu_con = np.tile(fopts, (neles, 1))
        vtu_con += (np.arange(neles)*nfopts)[:, None]

        # Generate offset into the connectivity array
        vtu_off = np.tile(nfopts, (neles, 1))
        vtu_off += (np.arange(neles)*len(fopts))[:, None]

        # Tile VTU cell type numbers
        types = self.vtk_types[mk]
        vtu_typ = np.tile(types, neles)

        # Write VTU node connectivity, connectivity offsets and cell types
        self._write_darray(vtu_con, vtuf, np.int32) 
        self._write_darray(vtu_off, vtuf, np.int32)
        self._write_darray(vtu_typ, vtuf, np.uint8)

        # Writing additional field data:
        self._write_darray(parts, vtuf, np.int32)


    def _get_npts_ncells(self, mk): 
        ncells = np.asarray(self._vtufnodes[mk]).shape[0]
        npts = ncells * self._petype_focount_map[mk]
        return npts, ncells

    def _get_array_attrs(self, mk):
        fpdtype = self.fpdtype 
        vdtype = 'Float32' if fpdtype == np.float32 else 'Float64'
        dsize = np.dtype(fpdtype).itemsize 

        names = ['', 'connectivity', 'offsets', 'types']
        types = [vdtype, 'Int32', 'Int32', 'UInt8']
        comps = ['3', '', '', '']

        vvars = self._vtk_vars
        for fname, varnames in vvars:
            names.append(fname.title())
            types.append('Int32')
            comps.append(str(len(varnames)))

        npts, ncells = self._get_npts_ncells(mk)
        nb = npts*dsize
        sizes = [3*nb, 4*npts, 4*ncells, ncells]
        dsize = np.dtype(np.int32).itemsize
        nb = npts*dsize
        sizes.extend(len(varnames)*nb for _, varnames in vvars)

        return npts, ncells, names, types, comps, sizes
