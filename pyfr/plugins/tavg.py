# -*- coding: utf-8 -*-

from ast import operator
import re

import numpy as np

import csv
from collections import defaultdict
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
        
        self.dev_mode = self.cfg.get(cfgsect, 'std-dev', 'none')
        

        # Expressions pre-processing
        self._prepare_exprs()

        # Output data type
        fpdtype = self.cfg.get(cfgsect, 'precision', 'single')
        if fpdtype == 'single':
            self.fpdtype = np.float32
        elif fpdtype == 'double':
            self.fpdtype = np.float64
        else:
            raise ValueError('Invalid floating point data type')

        # Base output directory and file name
        basedir = self.cfg.getpath(self.cfgsect, 'basedir', '.', abs=True)
        basename = self.cfg.get(self.cfgsect, 'basename')

        # Construct the file writer
        self._writer = NativeWriter(intg, basedir, basename, 'tavg')

        # Gradient pre-processing
        self._init_gradients(intg)

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
        self.vnames = []
        
        

        # Iterate over accumulation expressions first
        for k in cfg.items(cfgsect):
            if k.startswith('avg-'):
                self.anames.append(k[4:])
                self.aexprs.append(cfg.getexpr(cfgsect, k, subs=c))
                self.outfields.append(k)
                if self.dev_mode == 'all':
                    self.outfields.append(re.sub('avg-', 'std-dev-', k))
        
        # Followed by any functional expressions
        for k in cfg.items(cfgsect):
            if k.startswith('fun-avg-'):
                self.fexprs.append(cfg.getexpr(cfgsect, k, subs=c))
                self.outfields.append(k)
                        



    def _init_gradients(self, intg):
        # Determine what gradients, if any, are required
        gradpnames = set()
        for ex in self.aexprs:
            gradpnames.update(re.findall(r'\bgrad_(.+?)_[xyz]\b', ex))

        privarmap = self.elementscls.privarmap[self.ndims]
        self._gradpinfo = [(pname, privarmap.index(pname))
                           for pname in gradpnames]

    def _init_accumex(self, intg):
        self.tstart_acc = self.prevt = self.tout_last = intg.tcurr
        self.prevex = self._eval_acc_exprs(intg)
        self.accex = [np.zeros_like(p, dtype=np.float64) for p in self.prevex]
        self.vaccex = [np.zeros_like(a, dtype=np.float64) for a in self.accex]
        self.lamda = 0
        self.W1m = 1
        

        header = ['prevex', 'currex', 'time','doinit',  'doaccum', 'dowrite', 'var', 'avg', 'W1m', 'W1mpn']

        
        
       
        if np.amax(self.prevex[0] ) > np.amax(self.prevex[1] ):
            self.ab = np.where(self.prevex[0] == np.amax(self.prevex[0]))
            self.idx = 0
        else:
            self.ab = np.where(self.prevex[1]== np.amax(self.prevex[1]))
            self.idx = 1

        self.a = self.ab[0][0]
        self.b = self.ab[1][0]
        self.c = self.ab[2][0]

        
        
        with open('testing.csv', 'w') as f:
            self.writer = csv.writer(f)
            self.writer.writerow(header)

        # data = [self.prevex[0][self.ab[0][0], self.ab[1][0], self.ab[2][0]], '-', intg.tcurr,'yes', '-', '-', '-', '-']
        # Extra state for continuous accumulation
        # if self.mode == 'continuous':
        #     self.vcaccex = [np.zeros_like(v) for v in self.vaccex]
        #     self.tstart_actual = intg.tcurr

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
            import pdb;pdb.set_trace()
            exprs.append([npeval(v, subs) for v in self.fexprs])

        # Stack up the expressions for each element type and return
        return [np.dstack(exs).swapaxes(1, 2) for exs in exprs]

    def _eval_vexprs(self, var):
        
        

        pass



   

     

       
        
        # eavg_dev = []
        # emax_dev = []
        # for v in var:
        #     emax_dev.append([np.amax(val) for val in v])
        #     eavg_dev.append([np.average(val) for val in v])

        
        # exprs = defaultdict(list)
        # for vval in vaccex:
            
        #     subs = dict(zip(self.vnames, vval.swapaxes(0,1)))

        #     for v,a in zip(self.vexprs, self.vnames):
                 
        #         exprs[a].append(npeval(v, subs))
        
        
        # return [npeval(v, dict(exprs)) for v in self.vexprs]



            


    def _acc_avg_var(self, intg, currex, dowrite, doaccum):

        # if self.mode == 'windowed':
        #     # accex = self.accex
        #     tstart = self.tout_last
        #     vaccex = self.vaccex
       
        # else:
        #     # accex = self.caccex
        #     tstart = self.tstart_actual
        #     vaccex = self.vcaccex
            
        prevex = self.prevex
        vaccex = self.vaccex
        accex =  self.accex
        

        
        
        Wmp1mpn = (intg.tcurr - self.prevt)
        W1mpn = (intg.tcurr - self.tstart_acc)
        W1m = self.W1m

    
      

        for v, a, p, c in zip(vaccex, accex, prevex, currex):

            
            v += Wmp1mpn*(c ** 2 + p** 2) - 0.5*Wmp1mpn*(p+ c)**2 + \
                self.lamda*((W1m / (2 * Wmp1mpn * W1mpn)) * \
                (Wmp1mpn * a/W1m - (c + p) * Wmp1mpn)**2)
        

                                   
            a += Wmp1mpn*(p + c)

        self.W1m = W1mpn
        self.lamda = 1

        
        # print((vaccex[0][self.ab[0][0], self.ab[1][0], self.ab[2][0]])/(intg.tcurr - self.tout_last))
        a,b,c = self.a, self.b, self.c
        # tp = self.prevex[self.idx][a,b,c]
        

        if doaccum and not dowrite:
           

            data = [self.prevex[self.idx][a,b,c], \
                currex[self.idx][a,b,c], intg.tcurr, '-', \
                    doaccum, dowrite, \
                    self.vaccex[self.idx][a,b,c]/ (2*(intg.tcurr - self.tstart_acc)),\
                        accex[self.idx][a,b,c] / (2*(intg.tcurr - self.tstart_acc))\
                            ,self.W1m, W1mpn]
            with open('testing.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(data)

  
        
            

    def __call__(self, intg):
        # If we are not supposed to be averaging yet then return
        if intg.tcurr < self.tstart:
            return

        # If necessary, run the start-up routines
        if not self._started:
            self._init_accumex(intg)
            self._started = True
            return

        # See if we are due to write and/or accumulate this step
        dowrite = intg.tcurr - self.tout_last >= self.dtout - self.tol
        doaccum = intg.nacptsteps % self.nsteps == 0

        if dowrite or doaccum:
            # Evaluate the time averaging expressions
           
            currex = self._eval_acc_exprs(intg)
            
            
            
                
            self._acc_avg_var(intg,currex, dowrite, doaccum)

           

            if dowrite:
                comm, rank, root = get_comm_rank_root()

                accex = self.accex
                vaccex = self.vaccex
                               
                # Normalise the accumulated expressions
                tavg = [a / (2*(intg.tcurr - self.tstart_acc)) for a in accex]
                
               

                # Evaluate any functional expressions
                if self.fexprs:
                    
                    funex = self._eval_fun_exprs(intg, tavg)
                    tavg = [np.hstack([a, f]) for a, f in zip(tavg, funex)]
                
                if self.dev_mode in {'summarise', 'all'}:
                    
                    dev = [np.sqrt(np.abs(v / (2*(intg.tcurr - self.tstart_acc))))
                          for v in vaccex]
                    dexpr = list(map(lambda v: v.swapaxes(0,1), dev))

        
                    max_dev = [max(map(np.amax, v)) for v in zip(*dexpr)]
                    avg_dev = [np.hstack(list(map(np.ravel, v))).mean()
                               for v in zip(*dexpr)]

                    if self.dev_mode == 'all':
                        tavg = [np.hstack([a, v]) for a, v in zip(tavg, dev)]
                    

                if dowrite:
                    a,b,c  = self.a, self.b, self.c
                    data = [self.prevex[self.idx][a,b,c], currex[self.idx][a,b,c], intg.tcurr, '-', doaccum, dowrite, var[self.idx][a,b,c],tavg[self.idx][a,b,c] ]
                    with open('testing.csv', 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(data)
            
                
                
                # Form the output records to be written to disk
                data = dict(self._ele_region_data)

                for (idx, etype, rgn), d in zip(self._ele_regions, tavg):
                    data[etype] = d.astype(self.fpdtype)

                stats = Inifile()
                stats.set('data', 'prefix', 'tavg')
                stats.set('data', 'fields', ','.join(self.outfields))
                stats.set('tavg', 'tstart', self.tstart_acc)
                stats.set('tavg', 'tend', intg.tcurr)
                
                if self.dev_mode == 'summarise':
                    for an, vm, va in zip(self.anames, max_dev, avg_dev):
                        stats.set('tavg', f'avg-std-dev-{an}',va)
                        stats.set('tavg', f'max-std-dev-{an}',vm)


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
                    self.lamda = 0
                    self.tstart_acc = intg.tcurr



                self.tout_last = intg.tcurr
            

            # Save the time and solution
            self.prevt = intg.tcurr
            self.prevex = currex
