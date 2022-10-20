import re

import numpy as np

from pyfr.inifile import Inifile
from pyfr.mpiutil import get_comm_rank_root
from pyfr.nputil import npeval
from pyfr.plugins.base import BasePlugin, PostactionMixin, RegionMixin
from pyfr.writers.native import NativeWriter


class TavgPlugin(PostactionMixin, RegionMixin, BasePlugin):
    name = 'tavg'
    systems = ['*']
    formulations = ['dual', 'std']

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        # Underlying elements class
        self.elementscls = intg.system.elementscls

        # Averaging mode
        self.mode = self.cfg.get(cfgsect, 'mode', 'windowed')
        if self.mode not in {'continuous', 'windowed'}:
            raise ValueError('Invalid averaging mode')
        
        # Std deviation mode
        self.dev_mode = self.cfg.get(cfgsect, 'std-dev', 'none')
        if self.dev_mode not in {'summarise', 'all', 'none'}:
            raise ValueError('Invalid standard deviation mode')
        self.dev_mode = None if self.dev_mode == 'none' else self.dev_mode

        # Expressions pre-processing
        self._prepare_exprs()

        # Output data type
        fpdtype = self.cfg.get(cfgsect, 'precision', 'single')
        if fpdtype == 'single':
            self.fpdtype = np.float32
            self.eps = (np.finfo(np.float32).eps)**0.5
        elif fpdtype == 'double':
            self.fpdtype = np.float64
            self.eps = (np.finfo(np.float64).eps)**0.5
        else:
            raise ValueError('Invalid floating point data type')

        # Base output directory and file name
        basedir = self.cfg.getpath(self.cfgsect, 'basedir', '.', abs=True)
        basename = self.cfg.get(self.cfgsect, 'basename')

        # Construct the file writer
        self._writer = NativeWriter(intg, basedir, basename, 'tavg')

        # Gradient pre-processing
        self._init_gradients()

        # Time averaging parameters
        self.tstart = self.cfg.getfloat(cfgsect, 'tstart', 0.0)
        self.dtout = self.cfg.getfloat(cfgsect, 'dt-out')
        self.nsteps = self.cfg.getint(cfgsect, 'nsteps')

        # Register our output times with the integrator
        intg.call_plugin_dt(self.dtout)

        # Mark ourselves as not currently averaging
        self._started = False

    def _prepare_exprs(self):
        cfg, cfgsect = self.cfg, self.cfgsect
        c = self.cfg.items_as('constants', float)
        self.anames, self.aexprs = [], []
        self.outfields, self.fexprs = [], []
        self.vnames, self.fnames = [], []
        
        # Iterate over accumulation expressions first
        for k in cfg.items(cfgsect, prefix='avg-'):
            self.anames.append(k.removeprefix('avg-'))
            self.aexprs.append(cfg.getexpr(cfgsect, k, subs=c))
            self.outfields.append(k)

        # Followed by any functional expressions
        for k in cfg.items(cfgsect, prefix='fun-avg-'):
            self.fnames.append(k.removeprefix('fun-avg-'))
            self.fexprs.append(cfg.getexpr(cfgsect, k, subs=c))
            self.outfields.append(k)

        # Create fields for std deviations        
        if self.dev_mode == 'all':
            for k in cfg.items(cfgsect, prefix='avg-'):
                self.outfields.append(f'stdev-{k[4:]}')
                 
        # Followed by functional deviations
        if self.dev_mode == 'all':
            for k in cfg.items(cfgsect, prefix='fun-avg-'):
                self.outfields.append(f'stdev-{k[4:]}')
        
    def _init_gradients(self):
        # Determine what gradients, if any, are required
        gradpnames = set()
        for ex in self.aexprs:
            gradpnames.update(re.findall(r'\bgrad_(.+?)_[xyz]\b', ex))

        privarmap = self.elementscls.privarmap[self.ndims]
        self._gradpinfo = [(pname, privarmap.index(pname))
                           for pname in gradpnames]

    def _init_accumex(self, intg):
        comm, rank, root = get_comm_rank_root()
        self.tstart_acc = self.prevt = self.tout_last = intg.tcurr
        self.prevex = self._eval_acc_exprs(intg)
        self.accex = [np.zeros_like(p, dtype=np.float64) for p in self.prevex]
        self.vaccex = [np.zeros_like(a, dtype=np.float64) for a in self.accex]
        neles = 0
        import pdb;pdb.set_trace()
        for idx, etype, rgn in self._ele_regions:
            neles += intg.system.ele_shapes[idx][0]*len(rgn)
        self.ct = comm.reduce(neles, root=root)
   
    def _eval_acc_exprs(self, intg):
        exprs = []
        
        # Get the primitive variable names
        pnames = self.elementscls.privarmap[self.ndims]

        # Compute the gradients
        if self._gradpinfo:
            grad_soln = intg.grad_soln

        # Iterate over each element type in the simulation
        for idx, etype, rgn in self._ele_regions:
            soln = intg.soln[idx][..., rgn].swapaxes(0, 1)

            # Convert from conservative to primitive variables
            psolns = self.elementscls.con_to_pri(soln, self.cfg)

            # Prepare the substitutions dictionary
            subs = dict(zip(pnames, psolns))

            # Prepare any required gradients
            if self._gradpinfo:
                grads = np.rollaxis(grad_soln[idx], 2)[..., rgn]

                # Transform from conservative to primitive gradients
                pgrads = self.elementscls.grad_con_to_pri(soln, grads,
                                                          self.cfg)

                # Add them to the substitutions dictionary
                for pname, idx in self._gradpinfo:
                    for dim, grad in zip('xyz', pgrads[idx]):
                        subs[f'grad_{pname}_{dim}'] = grad

            # Evaluate the expressions
            exprs.append([npeval(v, subs) for v in self.aexprs])

        # Stack up the expressions for each element type and return
        return [np.dstack(exs).swapaxes(1, 2) for exs in exprs]

    def _eval_fun_exprs(self, intg, accex):
        exprs = []
        
        # Iterate over each element type our averaging region
        for avals in accex:
            # Prepare the substitution dictionary
            subs = dict(zip(self.anames, avals.swapaxes(0, 1)))

            exprs.append([npeval(v, subs) for v in self.fexprs])

        # Stack up the expressions for each element type and return
        return [np.dstack(exs).swapaxes(1, 2) for exs in exprs]

    def _eval_fun_var(self, dev, accex):
        dfexpr = []
        eps = self.eps
        an = self.anames
        fexp = self.fexprs
        devs = [d.swapaxes(0,1) for d in dev]
 
        # Iterate over each element type our averaging region
        for avals in accex:
            df = []
            av = avals.swapaxes(0,1)

            # # Prepare the substitution dictionary
            # subs = dict(zip(an, av))

            # # Evaluate the function
            # fv = [npeval(f, subs) for f in fexp]
            
            # Iterate over averaged variables
            for idx, avar in enumerate(an):
                h = np.zeros_like(av, dtype=np.float64)
                # Calculate step size  
                h[idx] = eps * av[idx]
                h[idx][h[idx] == 0] = eps
                
                # Prepare the substitution dictionary for step
                subs = dict(zip(an, av))
                subsh = dict(zip(an, av+h)) 
                
                # # Calculate derivatives for functional averages
                # df.append([(npeval(v, subsh) - fv[i]) / h[idx]
                #              for i, v in enumerate(fexp)])

                # Calculate derivatives for functional averages
                df.append([(npeval(v, subsh) - npeval(v, subs)) / h[idx]
                             for v in fexp])
            
            
            dfexpr.append(np.stack(df).swapaxes(0,1))
        import pdb;pdb.set_trace()
        tp = [np.sqrt(np.sum(df**2 * sd[None, :]**2), axis = 1).swapaxes(0,1)\
             for df, sd in zip(dfexpr, devs)]
        # Multiply with variance and take the RMS value
        return [np.sqrt(np.einsum('ij..., j... -> i...', 
            df**2, sd**2)).swapaxes(0,1) for df, sd in zip(dfexpr, devs)]
    
    def _acc_avg_var(self, intg, currex):  
        prevex, vaccex, accex = self.prevex, self.vaccex, self.accex  
                
        # Weights for online variance and average
        Wmp1mpn = intg.tcurr - self.prevt
        W1mpn = intg.tcurr - self.tstart_acc
        W1m = W1mpn - Wmp1mpn

        # If initialising or first step
        if self.tstart_acc == self.prevt:
            # Iterate over element type
            for v, p, c in zip(vaccex, prevex, currex):

                # Accumulate variance
                v += Wmp1mpn*(c**2 + p**2) - 0.5*Wmp1mpn*(p + c)**2

        # After first step
        else:
            # Iterate over element type
            for v, a, p, c in zip(vaccex, accex, prevex, currex):

                # Accumulate variance
                v += (Wmp1mpn*(c**2 + p**2) - 0.5*Wmp1mpn*(p + c)**2 + 
                     ((W1m / (2 * Wmp1mpn * W1mpn)) * 
                     (Wmp1mpn * a/W1m - (c + p) * Wmp1mpn)**2))

        # Accumulate average
        for a, p, c in zip(accex, prevex, currex):
            a += Wmp1mpn*(p + c)

    def __call__(self, intg):
        # If we are not supposed to be averaging yet then return
        if intg.tcurr < self.tstart:
            return
        
        # If necessary, run the start-up routines
        if not self._started:
            self._init_accumex(intg)
            self._started = True

        # See if we are due to write and/or accumulate this step
        dowrite = intg.tcurr - self.tout_last >= self.dtout - self.tol
        doaccum = intg.nacptsteps % self.nsteps == 0

        if dowrite or doaccum:
            # Evaluate the time averaging expressions
            currex = self._eval_acc_exprs(intg)
            
            # Accumulate them; always do this even when just writing
            self._acc_avg_var(intg, currex)
            
            # Save the time and solution
            self.prevt = intg.tcurr
            self.prevex = currex

            if dowrite:
                comm, rank, root = get_comm_rank_root()
                accex = self.accex
                vaccex = self.vaccex
                ct = self.ct
               
                # Normalise the accumulated expressions
                tavg = [a / (2*(intg.tcurr - self.tstart_acc)) for a in accex]
                
                if self.dev_mode:
                    ct = self.ct
    
                    # Calculate standard deviation
                    dev = [np.sqrt(np.abs(v / (2*(intg.tcurr - self.tstart_acc))))
                          for v in vaccex]

                    dexpr = [d.swapaxes(0, 1) for d in dev]

                    # Maximum and sum of standard deviations
                    mdat = np.zeros((len(self.anames)))
                    acdat = np.zeros((len(self.anames)))

                    for v in dev:
                        


                    # Maximum and sum of functional deviations
                    mfdat = np.zeros((len(self.fnames)))
                    afdat = np.zeros((len(self.fnames)))


                    for v in dexpr:

                    
                    # Calculate max standard deviation on each rank
                    mdat = [max(map(np.amax, v)) for v in zip(*dexpr)]

                    # Sum standard deviation for average
                    acdat = [np.hstack(list(map(np.ravel, v))).sum()
                               for v in zip(*dexpr)]
            
                    # Gather max and summmed values 
                    md = comm.gather(mdat, root=root)
                    ad = comm.gather(acdat, root=root)
                    
                    if rank==root:
                        
                        # Calculate max and average deviations on root
                        max_dev = [max(v) for v in zip(*list(filter(None, md)))]
                        avg_dev = [sum(v)/ct for v in
                                    zip(*list(filter(None, ad)))]

                    if self.fexprs:                    
                        # Evaluate functional variance
                        fvar = self._eval_fun_var(dev, tavg)
                        fexpr = list(map(lambda v: v.swapaxes(0,1), fvar))

                        # Calculate max functional deviation on each rank
                        mfdat = [max(map(np.amax, v)) for v in zip(*fexpr)]

                        # Sum functional deviation over element types
                        afdat = [np.hstack(list(map(np.ravel, v))).sum()
                                        for v in zip(*fexpr)]

                        # Gather max and summmed values 
                        mfd = comm.gather(mfdat, root=root)
                        afd = comm.gather(afdat, root=root)

                        if rank == root:
                            # Calculate max and average functional deviations
                            max_fdev = [max(v) for v in zip(*list(filter(None, mfd)))]
                            avg_fdev = [sum(v)/sum(ct) for v in
                                            zip(*list(filter(None, afd)))]
                    
                if self.fexprs:                      
                    # Evaluate functional averages
                    funex = self._eval_fun_exprs(intg, tavg)

                    # Stack the average expressions
                    tavg = [np.hstack([a, f]) for a, f in zip(tavg, funex)]
   
                if self.dev_mode == 'all':
                    # Stack std deviations
                    tavg = [np.hstack([a, d]) for a, d in zip(tavg, dev)]
                    
                    if self.fexprs:
                        # Stack functional deviations
                        tavg = [np.hstack([a, df]) for a, df in zip(tavg, fvar)]
                    
                # Form the output records to be written to disk
                data = dict(self._ele_region_data)

                for (idx, etype, rgn), d in zip(self._ele_regions, tavg):
                    data[etype] = d.astype(self.fpdtype)
                

                stats = Inifile()
                stats.set('data', 'prefix', 'tavg')
                stats.set('data', 'fields', ','.join(self.outfields))
                stats.set('tavg', 'tstart', self.tstart_acc)
                stats.set('tavg', 'tend', intg.tcurr)

                # Write summarised stats   
                if self.dev_mode:
                    if rank == root:
                        # Write std deviations
                        for an, vm, va in zip(self.anames, max_dev, avg_dev):
                            stats.set('tavg', f'avg-std-dev-{an}', va)
                            stats.set('tavg', f'max-std-dev-{an}', vm)
                        
                        # Followed by functional deviations
                        if self.fexprs:
                            for fn, fm, fa in zip(self.fnames, max_fdev, avg_fdev):
                                stats.set('tavg', f'fun-avg-std-dev-{fn}', fa)
                                stats.set('tavg', f'fun-max-std-dev-{fn}', fm)
                                
                intg.collect_stats(stats)

                # If we are the root rank then prepare the metadata
                if rank == root:
                    metadata = dict(intg.cfgmeta,
                                    stats=stats.tostr(),
                                    mesh_uuid=intg.mesh_uuid)
                else:
                    metadata = None

                # Write to disk
                solnfname = self._writer.write(data, intg.tcurr, metadata)

                # If a post-action has been registered then invoke it
                self._invoke_postaction(intg=intg, mesh=intg.system.mesh.fname,
                                        soln=solnfname, t=intg.tcurr)

                # Reset the accumulators
                if self.mode == 'windowed':
                    for a,v in zip(self.accex, self.vaccex):
                        a.fill(0)
                        v.fill(0)
                    self.tstart_acc = intg.tcurr

                self.tout_last = intg.tcurr
            
