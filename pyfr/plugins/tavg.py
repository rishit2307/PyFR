import re
import itertools as it
import numpy as np

from pyfr.inifile import Inifile
from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.nputil import npeval
from pyfr.plugins.base import BasePlugin, PostactionMixin, RegionMixin
from pyfr.writers.native import NativeWriter


class TavgPlugin(PostactionMixin, RegionMixin, BasePlugin):
    name = 'tavg'
    systems = ['*']
    formulations = ['dual', 'std']

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)
        
        comm, rank, root = get_comm_rank_root()
        # Underlying elements class
        self.elementscls = intg.system.elementscls

        # Averaging mode
        self.mode = self.cfg.get(cfgsect, 'mode', 'windowed')
        if self.mode not in {'continuous', 'windowed'}:
            raise ValueError('Invalid averaging mode')
        
        # Std deviation mode
        self.std_mode = self.cfg.get(cfgsect, 'std-mode', 'summary')
        if self.std_mode not in {'summary', 'all'}:
            raise ValueError('Invalid standard deviation mode')

        # Expressions pre-processing
        self._prepare_exprs()

        # Floating point precision
        self.eps = np.finfo(np.float64).eps

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
        self._init_gradients()

        # Time averaging parameters
        self.tstart = self.cfg.getfloat(cfgsect, 'tstart', 0.0)
        self.dtout = self.cfg.getfloat(cfgsect, 'dt-out')
        self.nsteps = self.cfg.getint(cfgsect, 'nsteps')

        # Register our output times with the integrator
        intg.call_plugin_dt(self.dtout)

        # Mark ourselves as not currently averaging
        self._started = False

        # Get total solution points for the region
        em = intg.system.ele_map
        rgn = self.cfg.get(self.cfgsect, 'region')
        ergn = self._ele_regions
        if rgn == '*':
            self.tpts = sum([em[e].nupts * em[e].neles for _, e, _ in ergn])
        else:
            self.tpts = sum([len(r) * em[e].nupts for _, e, r in ergn])

        # Reduce
        self.tpts = comm.reduce(self.tpts, root=root)
        # UNCOMMENT FOR DEBUGGING
        for idx, etype,rgn in self._ele_regions:

            print(f'rank is {rank} , {intg.soln[idx][..., rgn].shape}')
            
        
        if rank == 0:
            print(f'rank is {rank}, self.tpts is {self.tpts} ')

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
        if self.std_mode == 'all':
            for k in cfg.items(cfgsect, prefix='avg-'):
                self.outfields.append(f'stdev-{k[4:]}')
            for k in cfg.items(cfgsect, prefix='fun-avg-'):
                self.outfields.append(f'fun-stdev-{k[4:]}')
        
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


        # # Code for testing
        # header = ['prevex', 'currex', 'time','doinit',  'doaccum', 'dowrite', 'var', 'avg', 'W1m', 'W1mpn']

        
        
       
        # if np.amax(self.prevex[0] ) > np.amax(self.prevex[1] ):
        #     self.ab = np.where(self.prevex[0] == np.amax(self.prevex[0]))
        #     self.idx = 0
        # else:
        #     self.ab = np.where(self.prevex[1]== np.amax(self.prevex[1]))
        #     self.idx = 1

        # self.a = self.ab[0][0]
        # self.b = self.ab[1][0]
        # self.c = self.ab[2][0]
        
        # with open('testing.csv', 'w') as f:
        #     self.writer = csv.writer(f)
        #     self.writer.writerow(header)


   
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
        return [np.array(exs) for exs in exprs]

    def _eval_fun_exprs(self, avars):

        # Prepare the substitution dictionary
        subs = dict(zip(self.anames, avars))

        # Evaluate the function and return
        return np.array([npeval(v, subs) for v in self.fexprs])

    def _eval_fun_var(self, dev, accex):
        dfexpr = []
        exprs = []
        eps, an = np.sqrt(self.eps), self.anames

        # Iterate over each element type our averaging region
        for av in accex:
            df = []

            # Evaluate the function
            fx = self._eval_fun_exprs(av)
            exprs.append(fx)
            
            for i, _ in enumerate(an):
                
                # Calculate step size

                h = eps * np.maximum(abs(av[i]), eps, where=abs(av[i])>eps, out=np.ones_like(av[i]))   
                # tp = h = eps * abs(av[i]) 
                # np.maximum(tp, eps, where=h<eps**2, out=tp)
                av[i] += h
                
                # Evaluate function for the step
                fxh = self._eval_fun_exprs(av)
                
                # Calculate derivatives for functional averages
                df.append((fxh - fx) / h)
                av[i] -= h

            # Stack derivatives     
            dfexpr.append(np.array(df))

        # Multiply by variance and take RMS value
        fv = [np.linalg.norm(df * sd[:, None], axis=0) for df, sd in zip(dfexpr, dev)]
        
        return exprs, fv 

    # def _eval_fun_var(self, dev, accex):
    #     dfexpr = []
    #     eps = self.eps
    #     an = self.anames
    #     fexp = self.fexprs
    #     devs = [d.swapaxes(0,1) for d in dev]
 
    #     # Iterate over each element type our averaging region
    #     for avals in accex:
    #         df = []
    #         av = avals.swapaxes(0,1)

    #         # # Prepare the substitution dictionary
    #         # subs = dict(zip(an, av))

    #         # # Evaluate the function
    #         # fv = [npeval(f, subs) for f in fexp]
            
    #         # Iterate over averaged variables
    #         for idx, avar in enumerate(an):
    #             h = np.zeros_like(av, dtype=np.float64)
    #             # Calculate step size  
    #             h[idx] = np.sqrt(eps) * av[idx]
    #             h[idx][abs(h[idx]) < eps] = np.sqrt(eps)
                
    #             # Prepare the substitution dictionary for step
    #             subs = dict(zip(an, av))
    #             subsh = dict(zip(an, av+h)) 
                
    #             # # Calculate derivatives for functional averages
    #             # df.append([(npeval(v, subsh) - fv[i]) / h[idx]
    #             #              for i, v in enumerate(fexp)])

    #             # Calculate derivatives for functional averages
    #             df.append([(npeval(v, subsh) - npeval(v, subs)) / h[idx]
    #                          for v in fexp])
            
            
    #         dfexpr.append(np.stack(df).swapaxes(0,1))

    #     # tp = [np.sqrt(np.sum(df**2 * sd[None, :]**2), axis = 1).swapaxes(0,1)\
    #     #      for df, sd in zip(dfexpr, devs)]
    #     # Multiply with variance and take the RMS value
    #     return [np.sqrt(np.einsum('ij..., j... -> i...', 
    #         df**2, sd**2)).swapaxes(0,1) for df, sd in zip(dfexpr, devs)]
    
    def _acc_avg_var(self, intg, currex, doaccum, dowrite):  
        prevex, vaccex, accex = self.prevex, self.vaccex, self.accex  
                
        # Weights for online variance and average
        Wmp1mpn = intg.tcurr - self.prevt
        W1mpn = intg.tcurr - self.tstart_acc
        W1m = W1mpn - Wmp1mpn

        # Iterate over element type
        for v, a, p, c in zip(vaccex, accex, prevex, currex):

            # Accumulate average
            a += Wmp1mpn*(p + c)

            # Accumulate variance
            if self.tstart_acc == self.prevt:
                v += Wmp1mpn*(c**2 + p**2) - 0.5*Wmp1mpn*(p + c)**2

            else:
                v += (Wmp1mpn*(c**2 + p**2) - 0.5*Wmp1mpn*(p + c)**2 + 
                     ((W1m / (2 * Wmp1mpn * W1mpn)) * 
                     (Wmp1mpn * a/W1m - (c + p) * Wmp1mpn * W1mpn / W1m)**2))

        # # print((vaccex[0][self.ab[0][0], self.ab[1][0], self.ab[2][0]])/(intg.tcurr - self.tout_last))
        # a,b,c = self.a, self.b, self.c
        # # tp = self.prevex[self.idx][a,b,c]
        
        # # Code for testing
        # if doaccum and not dowrite and intg.tcurr != self.prevt:
           

        #     data = [self.prevex[self.idx][a,b,c], \
        #         currex[self.idx][a,b,c], intg.tcurr, '-', \
        #             doaccum, dowrite, \
        #             self.vaccex[self.idx][a,b,c]/ (2*(intg.tcurr - self.tstart_acc)),\
        #                 accex[self.idx][a,b,c] / (2*(intg.tcurr - self.tstart_acc))\
        #                     ,W1m, W1mpn]
        #     with open('testing.csv', 'a') as f:
        #         writer = csv.writer(f)
        #         writer.writerow(data)

  
        # Code for testing ends

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
            self._acc_avg_var(intg, currex, doaccum, dowrite)

            # if dowrite:
            #     a,b,c  = self.a, self.b, self.c
            #     data = [self.prevex[self.idx][a,b,c], currex[self.idx][a,b,c], intg.tcurr, '-', doaccum, dowrite, self.vaccex[self.idx][a,b,c]/ (2*(intg.tcurr - self.tstart_acc)), self.accex[self.idx][a,b,c] / (2*(intg.tcurr - self.tstart_acc)) ]
            #     with open('testing.csv', 'a') as f:
            #         writer = csv.writer(f)
            #         writer.writerow(data)
            
            # Save the time and solution
            self.prevt = intg.tcurr
            self.prevex = currex

            if dowrite:
                comm, rank, root = get_comm_rank_root()
                accex, vaccex = self.accex, self.vaccex
                fdev, funex = [], []   
                tpts = self.tpts
                wts = 2*(intg.tcurr - self.tstart_acc)
               
                # Normalise the accumulated expressions
                tavg = [a / wts for a in accex]

                # Calculate standard deviation
                dev = [np.sqrt(np.abs(v / wts)) for v in vaccex]

                # # Maximum and sum of standard deviations
                # max_dev = np.zeros(len(self.anames + self.fnames), dtype=np.float64)
                # acc_dev = np.zeros(len(self.anames + self.fnames),  dtype=np.float64)
                # for dx in dev:
                #     max_dev = np.maximum(np.amax(dx, axis=(1, 2)), max_dev)
                #     acc_dev += np.sum(dx, axis=(1, 2)) 
 
                # # Reduce and output if we're the root rank
                # if rank != root:
                #     comm.Reduce(max_dev, None, op=mpi.MAX, root=root)
                #     comm.Reduce(acc_dev, None, root=root)
                # else:
                #     comm.Reduce(mpi.IN_PLACE, max_dev, op=mpi.MAX, root=root)
                #     comm.Reduce(mpi.IN_PLACE, acc_dev, op=mpi.SUM, root=root)

                
                if self.fexprs:  
                    # #UNCOMMENT FOR DEBUGGING            
                    # funex = self._eval_fun_exprs(intg, tavg)
                    # fdev = self._eval_fun_var(dev, tavg)

                    # Evaluate functional expressions and deviations
                    
                    
                    # UNCOMMENT FOR DEBUGGING
                    av0 = tavg[1][0, 3, 29]
                    av1 = tavg[1][1, 3, 29]
                    av2 = tavg[1][2, 3, 29]
                    if abs(av0) > np.sqrt(self.eps):
                        av0h = abs(av0)*np.sqrt(self.eps)
                    else:
                        av0h = np.sqrt(self.eps)

                    if abs(av1) > np.sqrt(self.eps):
                        av1h = abs(av1)*np.sqrt(self.eps)
                    else:
                        av1h = np.sqrt(self.eps)

                    if abs(av2) > np.sqrt(self.eps):
                        av2h = abs(av2)*np.sqrt(self.eps)
                    else:
                        av2h = np.sqrt(self.eps)

                    print(av0h, av1h, av2h)
                    

                    fav   = av0*av1 + av2
                    fav0h = (av0h+av0)*av1+av2
                    fav1h = av0*(av1h+av1) + av2
                    fav2h = av0*av1 + av2h+av2
                    dfdav0 = (fav0h - fav)/av0h
                    dfdav1 = (fav1h - fav)/av1h
                    dfdav2 = (fav2h - fav)/av2h
                    print(fav, fav0h, fav1h, fav2h)
                    print(dfdav0, dfdav1, dfdav2)

                    dfdav = np.sqrt((dfdav0*dev[1][0,3, 29])**2 + (dfdav1*dev[1][1, 3, 29])**2 + (dfdav2*dev[1][2, 3, 29])**2)

                    import pdb;pdb.set_trace()
                    funex, fdev = self._eval_fun_var(dev, tavg)
                    


                    # Stack the functional expressions
                    tavg = [np.vstack([a, f]) for a, f in zip(tavg, funex)]
                

                
                
                # print(f'rank is {rank}, fd diff is {[np.all(f - f1 == 0) for (f, f1) in zip(fdev, fdev1)]}')
                # print(f'rank is {rank}, f diff is {[np.all(f - f1 == 0) for (f, f1) in zip(funex, funex1)]}')
                
                # Maximum and sum of deviations
                maxd = np.zeros_like(self.fnames + self.anames, dtype=np.float64)
                accd = np.zeros_like(self.fnames + self.anames, dtype=np.float64) 

                for dx, fx in it.zip_longest(dev, fdev):
                    fdv = np.vstack((dx, fx)) if fdev else dx
                    maxd = np.maximum(np.amax(fdv, axis=(1,2)), maxd)
                    accd += fdv.sum((1, 2))
                
        
                #UNCOMMENT FOR DEBUGGING    
                acc_d = []
                m_d = []
                for d in dev:
                    acc_d.append(np.sum(d, axis=(1, 2)))
                    m_d.append(np.amax(d, axis = (1, 2)))
                print(f'rank is {rank}, accd is {acc_d}')
                print(f'rank is {rank}, maxfd is {m_d}')
                # Reduce and output if we're the root rank
                if rank != root:
                    comm.Reduce(maxd, None, op=mpi.MAX, root=root)
                    comm.Reduce(accd, None, op=mpi.SUM, root=root)
                else:
                    comm.Reduce(mpi.IN_PLACE, maxd, op=mpi.MAX, root=root)
                    comm.Reduce(mpi.IN_PLACE, accd, op=mpi.SUM, root=root)
                    # UNCOMMENT FOR DEBUGGING    
                    print(f'rank is root, {accd}')
                    print(f'rank is root, {maxd}')

                if self.std_mode == 'all' and self.fexprs:
                    # Stack std deviations and functional deviations
                    tavg = [np.vstack([a, d, df]) for a, d, df in zip(tavg, dev, fdev)]
                
                elif self.std_mode == 'all':
                    # Stack std deviations only
                    tavg = [np.vstack([a, d]) for a, d in zip(tavg, dev)]

                # Form the output records to be written to disk
                data = dict(self._ele_region_data)

                for (idx, etype, rgn), d in zip(self._ele_regions, tavg):
                    data[etype] = d.swapaxes(0, 1).astype(self.fpdtype)
                
                stats = Inifile()
                stats.set('data', 'prefix', 'tavg')
                stats.set('data', 'fields', ','.join(self.outfields))
                stats.set('tavg', 'tstart', self.tstart_acc)
                stats.set('tavg', 'tend', intg.tcurr)

                # Write summarised stats   
                if rank == root:
                    anm, fnm = self.anames, self.fnames

                    # Write std deviations
                    for an, vm, vc in zip(anm, maxd[:len(anm)], accd[:len(anm)]):
                        stats.set('tavg', f'avg-std-{an}', vc/tpts)
                        stats.set('tavg', f'max-std-{an}', vm)
                    
                    # Followed by functional deviations
                    for fn, fm, fc in zip(fnm, maxd[len(anm):], accd[len(anm):]):
                        stats.set('tavg', f'fun-avg-std-{fn}', fc/tpts)
                        stats.set('tavg', f'fun-max-std-{fn}', fm)
                            
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

                    for a, v in zip(self.accex, self.vaccex):
                        a.fill(0)
                        v.fill(0)

                    self.tstart_acc = intg.tcurr

                self.tout_last = intg.tcurr
