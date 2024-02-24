import numpy as np
import copy
from pyfr.integrators.std.base import BaseStdIntegrator
from pyfr.util import memoize
from pyfr.mpiutil import get_comm_rank_root, mpi
from collections import defaultdict
class BaseStdStepper(BaseStdIntegrator):
    def collect_stats(self, stats):
        super().collect_stats(stats)

        # Total number of RHS evaluations
        stats.set('solver-time-integrator', 'nfevals', self._stepper_nfevals)

    # def _init_loworder(self):
    #     r0, r1, r2, r3, *r4 = self._regidx
    #     r5 = r4[1]

    #     for i in range(len(self.system.ele_types)):
    #         proj = self.system.projmat[i]
    #         b = self.system.ele_banks[i][0]
    #         c = self.system.lele_banks[i][0]
    #         self.backend.run_kernels([self.backend.kernel('mul', proj, b, out=c)])

    # def eval_jac(self, t):
    #     add, rhs = self._add, self.system.rhs
    #     nupts = self.system.ele_shapes[0][0]
    #     jac = defaultdict(list)

    #     for col in self.system.celes.keys():
    #         for v in range(self.system.nvars):
    #             for npt in range(nupts):
    #                 eidx = self.system.celes[col]
    #                 ur0 = self.system.lele_banks[0][0].get()
    #                 eps = np.zeros_like(ur0)
    #                 eps[npt, v, eidx] = 1e-8

    #                 ur = ur0+eps
    #                 self.system.lele_banks[0][1].set(ur)
    #                 rhs(t, 0, 0)
    #                 rhs(t, 1, 1)

    #                 dr1 = self.system.lele_banks[0][0].get()
    #                 dr0 = self.system.lele_banks[0][1].get()

    #                 dr = (dr1-dr0)/eps

    #                 for e in eidx:
    #                     jac[e].append(dr[..., e])
                        

    #     for e in range(len(self.system.neles)):
    #         jac[e] = np.array(jac[e])

                         
                

    def _init_gmres(self,t, dt, dtfac=1, lclass=None):
        self.m = 400
        self.rnorm= dict()

        self.ltol = 1e-13
        self.eletype = dict()

        comm, rank, root = get_comm_rank_root()

        self.eletype = set(comm.allreduce(self.system.ele_types, op=mpi.SUM))
        self.eletype = comm.bcast(self.eletype, root=root)
        prec = self.cfg.get('backend', 'precision')
        if prec == 'double':
            self.epsmc = np.sqrt(np.finfo(float).eps)
        else:
            self.epsmc = np.sqrt(np.finfo(np.float32).eps)
        # for etype in self.eletype:
        #     self.rnorm[etype] = 0.0
        #     self.l2u[etype] = 0.0
        #     self.err[etype] = 0.0
        

        self.e1 = np.zeros(self.m+1)
        self.e1[0] = 1.0
        
        # for i in range(len(self.system.ele_types)):
        #     self.e1[i][0] =  1.0
            
        self.sn = np.zeros(self.m)
        self.cs = np.zeros(self.m)
        
        self.y = [[] for _ in range(len(self.system.ele_types))]

        if lclass:
            self._init_loworder(lclass)
            self._eval_jac(lclass, t, dt, dtfac)
    
    def solve_gmres(self, t, dt, x, dtfac=1.0, lclass=None):
        comm, rank, root = get_comm_rank_root()
        y = self.y
        cs, sn = self.cs, self.sn
        ltol = self.ltol
        rnorm, m = self.rnorm, self.m
        add, rhs_with_postproc = self._add, self.system.rhs
        l2u, eletype = self.l2u, self.eletype
        r0, r1, r2, r3, *r4 = self._regidx

        # for i, etype in enumerate(self.system.ele_types):
        #         rnorm[etype] = (np.linalg.norm(self.system.ele_banks[i][r1].get()))**2
            
        # for etype in eletype:
        #         rnorm[etype] = np.sqrt(comm.allreduce(rnorm[etype], op=mpi.SUM))

        rnorm = sum([np.linalg.norm(self.system.ele_banks[i][r1].get())**2
                              for i in range(len(self.system.ele_types))])
        rnorm = np.sqrt(comm.allreduce(rnorm, op=mpi.SUM))

        Q = [[] for i in range(len(self.system.ele_types))]
            
        for i, etype in enumerate(self.system.ele_types):
            nupts, nvars, neles = self.system.ele_shapes[i]
            Q[i] = np.zeros((nupts, nvars, neles, m+1))
            Q[i][..., 0] = self.system.ele_banks[i][r1].get() / rnorm

        H = np.zeros((m+1, m))
        beta = rnorm*self.e1

        for k in range(m):
            H[:k+2, k], q = self.arnoldi(Q, k, t, dt, dtfac, lclass=lclass)

            for i in range(len(self.system.ele_types)):
                Q[i][..., k+1] = q[i]

            H[:k+2, k], cs[k], sn[k] = self.giv_rot(H[:k+2, k], 
                                                    cs, sn ,k)

            beta[k+1] = -sn[k] * beta[k]
            beta[k] = cs[k] * beta[k]

            err = abs(beta[k+1]) / rnorm
            
            if err < ltol:
                if rank == root:
                    print(f'GMRES converged in {k} iterations, error is {err}')
                    
                break
        
        if k == m-1 and err > ltol:
            if rank == root:

                print(f'GMRES did not converge in {m} iterations, error is {err}')
        
        y =  np.linalg.solve(H[:k+1, :k+1], beta[:k+1])

        # import pdb;pdb.set_trace()
        for i in range(len(self.system.ele_types)):
            
            
            if lclass:
                if rank == root:
                    print('Hi, inside lclass')
                self.system.ele_banks[i][r3].set(Q[i][..., :k+1] @ y)
                self.restrict(lclass, r3)
                self.jac_mult(lclass)
                self.prolongate(lclass, r3)
                x[i] += self.system.ele_banks[i][r3].get()
            else:
                x[i] += Q[i][..., :k+1] @ y

            self.system.ele_banks[i][r3].set(x[i])

        # y = np.linalg.lstsq(H[:m+1, :m], beta, rcond=None)[0]
        # for i in range(len(self.system.ele_types)):
        #     x[i] += Q[i][..., :k+1] @ y
        #     self.system.ele_banks[i][r3].set(x[i])
       

    def arnoldi(self, Q, k, t, dt, dtfac, lclass=None):
        add, rhs_with_postproc = self._add, self.system.rhs
        # h, qnorm = dict(), dict()
        h = np.zeros(k+2)
        netype = len(self.system.ele_types)
        eletype = self.eletype
        
        


        r0, r1, r2, r3, *r4 = self._regidx
        r4 = r4[0]

        comm, rank, root = get_comm_rank_root()

        for i in range(netype):
            self.system.ele_banks[i][r3].set(Q[i][..., k])
        
        Un = sum([np.linalg.norm(self.system.ele_scal_upts(r2)[i])**2
                for i in range(netype)])
        Un = np.sqrt(comm.allreduce(Un, op=mpi.SUM))

        Qn = sum([np.linalg.norm(self.system.ele_scal_upts(r3)[i])**2
                  for i in range(netype)])
        Qn = comm.allreduce(Qn, op=mpi.SUM)
        
        eps =  self.epsmc*np.sqrt(Un + 1)/np.sqrt(Qn)

        if lclass:
            self.restrict(lclass, r3)
            self.jac_mult(lclass)
            self.prolongate(lclass, r3)

        
        # r1 = Un+1,k + eps*Q
        add(0.0, r1, eps, r3, 1.0, r2)

        # r1 = R(Un+eps*Q)
        rhs_with_postproc(t+dt, r1, r1)      

        # r1 = R(Un+eps*Q)/eps + Q/dt
        add(-1.0/eps, r1, dtfac/dt, r3)

        # r1 = R(Un+eps*Q)/eps + Q/dt  - R(Un)/eps 
        add(1.0, r1, 1.0/eps, r4)
        
        q = [self.system.ele_banks[i][r1].get() for i in range(netype)]

        for j in range(k+1):
            h[j] = sum([np.dot(q[i].reshape(-1), Q[i][..., j].reshape(-1))
                        for i in range(len(self.system.ele_types))])
            h[j] = comm.allreduce(h[j], op=mpi.SUM)

            for i in range(netype):
                q[i] -= h[j] * Q[i][..., j]
            

        qnorm = sum([np.linalg.norm(q[i])**2 for i in range(netype)])
        qnorm = np.sqrt(comm.allreduce(qnorm, op=mpi.SUM))
        h[k+1] = qnorm
        # for i, etype in enumerate(self.system.ele_types):
        #     qnorm[etype] = (np.linalg.norm(q[i]))**2
        
        # for etype in eletype:
        #     qnorm[etype] = np.sqrt(comm.allreduce(qnorm[etype], mpi.SUM))

        for i in range(netype):
            q[i] /= h[k+1]

        return h, q
    
    def giv_rot(self, h, cs, sn, k):
        for i in range(k):
            temp = cs[i] * h[i] + sn[i] * h[i+1]

            h[i+1] = -sn[i] * h[i] + cs[i] * h[i+1]
            h[i] = temp
        
        cs_k, sn_k = self.giv(h[k], h[k+1])

        h[k] = cs_k * h[k] + sn_k * h[k+1]
        h[k+1] = 0.0


        return h, cs_k, sn_k

    def giv(self, v1, v2):
        tt = np.sqrt(v1**2 + v2**2)
        cs = v1/tt
        sn = v2/tt

        return cs, sn

class StdEulerStepper(BaseStdStepper):
    stepper_name = 'euler'
    stepper_has_errest = False
    stepper_nregs = 2
    stepper_order = 1

    @property
    def _stepper_nfevals(self):
        return self.nsteps

    def step(self, t, dt):
        add, rhs_with_postproc = self._add, self.system.rhs
        ut, f = self._regidx
        rhs_with_postproc(t, ut, f)
        add(1.0, ut, dt, f)

        return ut
    
class Euler(BaseStdStepper):
    stepper_name = 'backward-euler'
    stepper_has_errest = False
    stepper_nregs = 5
    stepper_order = 1

    @property
    def _stepper_nfevals(self):
        return self.nsteps

    def _res(self, t, dt, dtfac=1.0, eps=None):
        add, rhs_with_postproc = self._add, self.system.rhs
        eletype =  self.eletype
        comm, rank, root = get_comm_rank_root()

        r0, r1, r2, r3, *r4 = self._regidx
        r4 = r4[0]

        self.l2u = sum([np.linalg.norm(self.system.ele_scal_upts(r2)[i])**2 for 
                        i in range(len(self.system.ele_types))])
        
        self.l2u = np.sqrt(comm.allreduce(self.l2u, mpi.SUM))
        
        dUn = sum([np.linalg.norm(self.system.ele_scal_upts(r3)[i])**2
                for i in range(len(self.system.ele_types))])
        dUn = comm.allreduce(dUn, op=mpi.SUM)
        dUn = np.sqrt(dUn)
        
        # max_etp = max(l2u, key=l2u.get)
        eps = self.epsmc*np.sqrt(self.l2u+1)/(dUn + self.epsmc**2)

        # r1 = Un+1,k + eps*DUn+1,k
        add(0.0, r1, eps, r3, 1.0, r2)

        # r1 = R(Un+1,k+1)
        rhs_with_postproc(t+dt, r1, r1)

        # r1 = R(Un+1,k+1)/eps + dUnk/dt
        add(-1.0/eps, r1, dtfac/dt, r3)

        # # r3 = R(Un)
        # rhs_with_postproc(t, r0, r3)

        # r4 = R(Un+1, k)
        rhs_with_postproc(t+dt, r2, r4)

        # r3 = r4 = R(Un+1, k)
        add(0.0, r3, 1.0, r4)

        # r1 = R(Un+1,k+1)/eps + dUnk/dt - R(Un+1,k)/eps
        add(1.0, r1, 1.0/eps, r4)

        # # r3 = R(Un)/2 + R(Un+1,k)/2 
        # add(dtfac/2., r3, dtfac/2., r4)

        # r3 = R(Un+1,k)  + Un+1,k/dt - Un/dt
        add(1.0, r3, -dtfac/dt, r2, dtfac/dt, r0)

        # r1 = b - Ax
        add(-1.0, r1, 1.0, r3)

    def newton_res(self, t, dt):
        add, rhs_with_postproc = self._add, self.system.rhs
        self.normele = dict()
        comm, rank, root = get_comm_rank_root()

        r0, r1, r2, r3, *r4 = self._regidx
        r4 = r4[0]

        # r1 = Du/Dt
        add(0.0, r1, 1/dt, r2, -1/dt, r0)

        # r4 = R(Un+1)
        rhs_with_postproc(t+dt, r2, r4)

        # r1 = Du/dt + R(Un+1)
        add(1.0, r1, -1, r4)
        # import pdb;pdb.set_trace()
        self.normele = sum([np.linalg.norm(self.system.ele_banks[i][r1]
                            .get())**2 for i in range(len(self.system.ele_types))])
        
        self.normele = np.sqrt(comm.allreduce(self.normele, op=mpi.SUM))
        return self.normele/self._get_gndofs()
        # return self.normele
    
    def step(self, t, dt):
        r0, r1, r2, r3, *r4 = self._regidx
        r4 = r4[0]
        add = self._add
        nnorm = np.inf
        s = 1.0
        ntol = self.ntol = 1e-4

        comm, rank, root = get_comm_rank_root()
        print(f't is {t}, dt is {dt}')
        nonlin_iter = 0

        while nnorm > ntol:
            self._init_gmres()

            x = copy.deepcopy(self.system.ele_scal_upts(r3))

            self._res(t, dt, dtfac=1.0)

            self.solve_gmres(t, dt, x, dtfac=1.0)

            # r2 = Un+1,k+1 = Un+1,k + s*dUk
            add(1.0, r2, s, r3)

            nnorm = self.newton_res(t, dt)

            nonlin_iter+= 1
            if rank == root:
                print(f'Newton residual is {nnorm}')
                print(nonlin_iter)

        # r0 = Un+1 = r2
        add(0.0, r0, 1.0, r2)

        print("Step completed")

        return r0


class Trapezoidal(BaseStdStepper):
    stepper_name = 'trapezium'
    stepper_has_errest = False
    stepper_nregs = 6
    stepper_order = 1

    @property
    def _stepper_nfevals(self):
        return self.nsteps
    

    def _res(self, t, dt, dtfac=1.0):

        add, rhs_with_postproc = self._add, self.system.rhs
        eletype =  self.eletype
        comm, rank, root = get_comm_rank_root()

        r0, r1, r2, r3, *r4 = self._regidx
        r4 = r4[0]
        # for i, etype in enumerate(self.system.ele_types):
        #     l2u[etype] = (np.linalg.norm(self.soln[i]))**2
 
        # for etype in eletype:
        #     l2u[etype] = np.sqrt(comm.allreduce(l2u[etype],
        #                   op=mpi.SUM))

        self.l2u = sum([np.linalg.norm(self.system.ele_scal_upts(r2)[i])**2 for 
                        i in range(len(self.system.ele_types))])
        
        self.l2u = np.sqrt(comm.allreduce(self.l2u, mpi.SUM))
        
        dUn = sum([np.linalg.norm(self.system.ele_scal_upts(r3)[i])**2
                for i in range(len(self.system.ele_types))])
        dUn = comm.allreduce(dUn, op=mpi.SUM)
        dUn = np.sqrt(dUn)
        
        # max_etp = max(l2u, key=l2u.get)
        eps = self.epsmc*np.sqrt(self.l2u+1)/(dUn + self.epsmc**2)

        # r1 = Un+1,k + eps*DUn+1,k
        add(0.0, r1, eps, r3, 1.0, r2)

        # r1 = R(Un+1,k+1)
        rhs_with_postproc(t+dt, r1, r1)

        # r1 = R(Un+1,k+1)/eps + dUnk/dt
        add(-1.0/eps, r1, dtfac/dt, r3)

        # r3 = R(Un)
        rhs_with_postproc(t, r0, r3)

        # r4 = R(Un+1, k)
        rhs_with_postproc(t+dt, r2, r4)

        # r1 = R(Un+1,k+1)/eps + dUnk/dt - R(Un+1,k)/eps
        add(1.0, r1, 1.0/eps, r4)

        # r3 = R(Un)/2 + R(Un+1,k)/2 
        add(dtfac/2., r3, dtfac/2., r4)

        # r3 = R(Un)/2 + R(Un+1,k)/2  + Un+1,k/dt - Un/dt
        add(1.0, r3, -dtfac/dt, r2, dtfac/dt, r0)

        # r1 = b - Ax
        add(-1.0, r1, 1.0, r3)



    def newton_res(self, t, dt, tp=0):
        add, rhs_with_postproc = self._add, self.system.rhs
        self.normele = dict()
        comm, rank, root = get_comm_rank_root()

        r0, r1, r2, r3, *r4 = self._regidx
        r4 = r4[0]

        # for etp in self.eletype:
        #     self.normele[etp] = 0

        # if tp == 1.0:
        #     rhs_with_postproc(t-dt, r0, r1)
        # else:
        #     rhs_with_postproc(t, r0, r1)

        
        # r1 = R(Un)
        rhs_with_postproc(t, r0, r1)

        # r1 = R(Un)/2 + Du/Dt
        add(-1/2, r1, 1/dt, r2, -1/dt, r0)
        
        # if tp == 1.0:
        # # r4 = R(Un+1)
        #     rhs_with_postproc(t, r2, r4)
        
        # else:
        #     rhs_with_postproc(t, r2, r4)

        # r4 = R(Un+1)
        rhs_with_postproc(t+dt, r2, r4)

        # r1 = R(Un)/2 + Du/dt + R(Un+1)/2
        add(1.0, r1, -1/2, r4)

        self.normele = sum([np.linalg.norm(self.system.ele_banks[i][r1]
                            .get())**2 for i in range(len(self.system.ele_types))])
        
        self.normele = np.sqrt(comm.allreduce(self.normele, op=mpi.SUM))
        return self.normele/self._get_gndofs()
        # return self.normele

    def step(self, t, dt, lclass=None):
        r0, r1, r2, r3, *r4 = self._regidx
        r4 = r4[0]
        add = self._add
        nnorm = np.inf
        s = 1.0
        ntol = self.ntol = 0.1
        comm, rank, root = get_comm_rank_root()

        nonlin_iter = 0
        
       
        # prev_err = self.newton_res(t, dt, tp=1)

        while nnorm > ntol:
            
            self._init_gmres(t, dt, dtfac=2.0, lclass=lclass)


            # print(f'Prev err is {prev_err}')
            x = copy.deepcopy(self.system.ele_scal_upts(r3))
            
            self._res(t, dt, dtfac=2.0)

            
            self.solve_gmres(t, dt, x, dtfac=2.0, lclass=lclass)

            y = copy.deepcopy(self.system.ele_scal_upts(r2))

            # r2 = Un+1,k+1 = Un+1,k + s*dUk
            add(1.0, r2, s, r3)

            nnorm = self.newton_res(t, dt)
            

            # while nnorm >= prev_err:
            #     for i in range(len(self.system.ele_types)):
            #         self.system.ele_banks[i][r2].set(y[i])
                
            #     s /= 2
            #     print("Line search activated")
            #     # r2 = Un+1,k+1 = Un+1,k + s*dUk
            #     add(1.0, r2, s, r3)

            #     nnorm = self.newton_res(t, dt)

            #     print(f'Line search nnorm is {nnorm}')

            
            # prev_err = nnorm
            
            nonlin_iter+= 1
            if rank == root:
                print(nnorm)
                print(nonlin_iter)

        # print(f'x is {np.sum(self.system.ele_scal_upts(r3))}')
        # r0 = Un+1 = r2
        add(0.0, r0, 1.0, r2)
        if rank == root:
            print("Step completed")

        return r0

class BDF2(Trapezoidal):
    stepper_name = 'bdf2'
    stepper_has_errest = False
    stepper_nregs = 6
    stepper_order = 3

    @property
    def _stepper_nfvals(self):
        return 4*self.nsteps
    
    def _res(self, t, dt, dtfac=1.0):
        add, rhs_with_postproc = self._add, self.system.rhs
        l2u, eletype = self.l2u, self.eletype
        comm, rank, root = get_comm_rank_root()

        r0, r1, r2, r3, r4, r5 = self._regidx

        self.l2u = np.array(sum([np.linalg.norm(self.soln[i])**2 for 
                        i in range(len(self.system.ele_types))]))
        
        comm.Allreduce(mpi.IN_PLACE, self.l2u, mpi.SUM)
        self.l2u = np.sqrt(self.l2u)

        eps = 1e-7*l2u

        # r1 = Un+2,k + eps*DUn+2,k
        add(0.0, r1, 1.0, r2, eps, r3)

        # r1 = R(Un+2,k+1)
        rhs_with_postproc(t+dt, r1, r1)

        # r1 = R(Un+2,k+1)/eps + dUnk/dt
        add(-1.0/eps, r1, dtfac/dt, r3)

        # r4 = R(Un+2, k)
        rhs_with_postproc(t+dt, r2, r4)

        # r1 = R(Un+2,k+1)/eps + dUnk/dt - R(Un+2,k)/eps
        add(1.0, r1, 1.0/eps, r4)

        # r3 = 3*Un+2,k/2*dt - 2*Un+1/dt + Un/2*dt
        add(0.0, r3, -dtfac/dt, r2, dtfac*4/(3*dt), r5, -dtfac/(dt*3), r0)

        # r3 = r3 + R(Un+2, k, tn+2)
        add(1.0, r3, 1.0, r4)

        # r1 = b - Ax
        add(-1.0, r1, 1.0, r3)

        return eps
    
    def newton_res(self, t, dt):
        add, rhs_with_postproc = self._add, self.system.rhs
        # self.normele = dict()
        comm, rank, root = get_comm_rank_root()

        r0, r1, r2, r3, r4, r5 = self._regidx

        # for etp in self.eletype:
        #     self.normele[etp] = 0

        # r1 = R(Un+2,k+1)
        rhs_with_postproc(t+dt, r2, r1)

        # r1 = Un+2,k - 4*Un+1/3 + Un/3 + 2*dt*R(Un+2,tn+2)/3
        add(-2*dt/3, r1, 1.0, r2, -4/3, r5, 1/3, r0)

        self.normele = sum([np.linalg.norm(self.system.ele_banks[i][r1]
                            .get())**2 for i in range(len(self.system.ele_types))])
        
        self.normele = np.sqrt(comm.allreduce(self.normele, op=mpi.SUM))
        return self.normele/self._get_gndofs()



    def step(self, t, dt):
        r0, r1, r2, r3, r4, r5 = self._regidx
        ntol = self.ntol = 0.001
        add = self._add
        nnorm = 1.0
        nonlin_iter = 0
        comm, root, rank = get_comm_rank_root()
        y = copy.deepcopy(self.system.ele_scal_upts(r0))

        if t == 0.0:
            dt /= 5.
            for j in range(5):
                while np.amax(nnorm) > ntol:
                    self._init_gmres()

                    x = copy.deepcopy(self.system.ele_scal_upts(r3))

                    eps = super()._res(t, dt, dtfac=2.0)
            
                    self.solve_gmres(t, dt, eps, x, dtfac=2.0)

                    nnorm = super().newton_res(t, dt)
                    
                    nonlin_iter+= 1
                    if rank == root:
                        print(f'Trapezium rule {j} step')
                        print(nnorm)
                        print(nonlin_iter)
                
                # r0 = Un+1 = Un + dUn
                add(0.0, r0, 1.0, r2)
                # print(f'x is {np.sum(self.system.ele_scal_upts(r3))}')

                nnorm = 1.0

                t += dt/5.
            
            add(0.0, r5, 1.0, r2)
            for i in range(len(self.system.ele_types)):
                self.system.ele_banks[i][r0].set(y[i])

            return r2

        

        
        print(f'time is {t}')
        while np.amax(nnorm) > ntol:
            self._init_gmres()

            x = copy.deepcopy(self.system.ele_scal_upts(r3))

            eps = self._res(t, dt, dtfac=1.5)

            self.solve_gmres(t, dt, eps, x, dtfac=1.5)

            nnorm = self.newton_res(t, dt)

            nonlin_iter+= 1
            if rank == root:
                print(nnorm)
                print(nonlin_iter)

        add(0.0, r0, 1.0, r5)
        add(0.0, r5, 1.0, r2)

        return r2




        




class StdTVDRK3Stepper(BaseStdStepper):
    stepper_name = 'tvd-rk3'
    stepper_has_errest = False
    stepper_nregs = 3
    stepper_order = 3

    @property
    def _stepper_nfevals(self):
        return 3*self.nsteps

    def step(self, t, dt):
        add, rhs_with_postproc = self._add, self.system.rhs

        # Get the bank indices for each register (n, n+1, rhs)
        r0, r1, r2 = self._regidx

        # Ensure r0 references the bank containing u(t)
        if r0 != self._idxcurr:
            r0, r1 = r1, r0

        # First stage; r2 = -∇·f(r0); r1 = r0 + dt*r2
        rhs_with_postproc(t, r0, r2)
        add(0.0, r1, 1.0, r0, dt, r2)

        # Second stage; r2 = -∇·f(r1); r1 = 0.75*r0 + 0.25*r1 + 0.25*dt*r2
        rhs_with_postproc(t + dt, r1, r2)
        add(0.25, r1, 0.75, r0, 0.25*dt, r2)

        # Third stage; r2 = -∇·f(r1);
        #              r1 = 1.0/3.0*r0 + 2.0/3.0*r1 + 2.0/3.0*dt*r2
        rhs_with_postproc(t + 0.5*dt, r1, r2)
        add(2.0/3.0, r1, 1.0/3.0, r0, 2.0/3.0*dt, r2)

        # Return the index of the bank containing u(t + dt)
        return r1

class StdRK2Stepper(BaseStdStepper):
    stepper_name = 'rk2'
    stepper_has_errest = False
    stepper_nregs = 3
    stepper_order = 4

    @property
    def _stepper_nfevals(self):
        return 2*self.nsteps

    def step(self, t, dt):
        add, rhs_with_postproc = self._add, self.system.rhs

        r0, r1, r2 = self._regidx

        # Ensure r0 references the bank containing u(t)
        if r0 != self._idxcurr:
            r0, r1 = r1, r0

        rhs_with_postproc(t, r0, r1)

        add(0.0, r2, dt, r1, 1.0, r0)

        rhs_with_postproc(t+dt, r2, r2)

        add(dt/2, r1, dt/2, r2)

        add(1.0, r1, 1.0, r0)

        return r1

class StdRK4Stepper(BaseStdStepper):
    stepper_name = 'rk4'
    stepper_has_errest = False
    stepper_nregs = 3
    stepper_order = 4

    @property
    def _stepper_nfevals(self):
        return 4*self.nsteps

    def step(self, t, dt):
        add, rhs_with_postproc = self._add, self.system.rhs

        # Get the bank indices for each register
        r0, r1, r2 = self._regidx
        # Ensure r0 references the bank containing u(t)
        if r0 != self._idxcurr:
            r0, r1 = r1, r0

        # First stage; r1 = -∇·f(r0)
        rhs_with_postproc(t, r0, r1)

        # Second stage; r2 = r0 + dt/2*r1; r2 = -∇·f(r2)
        add(0.0, r2, 1.0, r0, dt/2.0, r1)
        rhs_with_postproc(t + dt/2.0, r2, r2)

        # As no subsequent stages depend on the first stage we can
        # reuse its register to start accumulating the solution with
        # r1 = r0 + dt/6*r1 + dt/3*r2
        add(dt/6.0, r1, 1.0, r0, dt/3.0, r2)

        # Third stage; here we reuse the r2 register
        # r2 = r0 + dt/2*r2
        # r2 = -∇·f(r2)
        add(dt/2.0, r2, 1.0, r0)
        rhs_with_postproc(t + dt/2.0, r2, r2)

        # Accumulate; r1 = r1 + dt/3*r2
        add(1.0, r1, dt/3.0, r2)

        # Fourth stage; again we reuse r2
        # r2 = r0 + dt*r2
        # r2 = -∇·f(r2)
        add(dt, r2, 1.0, r0)
        rhs_with_postproc(t + dt, r2, r2)

        # Final accumulation r1 = r1 + dt/6*r2 = u(t + dt)
        add(1.0, r1, dt/6.0, r2)

        # Return the index of the bank containing u(t + dt)
        return r1


class StdRKVdH2RStepper(BaseStdStepper):
    # Coefficients
    a = []
    b = []
    bhat = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Register our pointwise kernel
        self.backend.pointwise.register('pyfr.integrators.std.kernels.rkvdh2')

        # Compute the coefficients
        self.c = [0.0] + [sum(self.b[:i]) + ai for i, ai in enumerate(self.a)]
        self.e = [b - bh for b, bh in zip(self.b, self.bhat)]

        self._nstages = len(self.c)

    @memoize
    def _get_rkvdh2_kerns(self, stage, r1, r2, rold=None, rerr=None):
        kerns = []
        tplargs = {
            'a': self.a, 'b': self.b, 'e': self.e,
            'stage': stage, 'nstages': self._nstages,
            'nvars': self.system.nvars, 'errest': rold is not None
        }

        for dims, em in zip(self.system.ele_shapes, self.system.ele_banks):
            if rold is not None:
                kern = self.backend.kernel(
                    'rkvdh2', tplargs=tplargs, dims=[dims[0], dims[2]],
                    r1=em[r1], r2=em[r2], rold=em[rold], rerr=em[rerr],
                )
            else:
                kern = self.backend.kernel(
                    'rkvdh2', tplargs=tplargs, dims=[dims[0], dims[2]],
                    r1=em[r1], r2=em[r2],
                )

            kerns.append(kern)

        return kerns

    @property
    def stepper_has_errest(self):
        return self.controller_needs_errest and len(self.bhat)

    @property
    def _stepper_nfevals(self):
        return len(self.b)*self.nsteps

    @property
    def stepper_nregs(self):
        return 4 if self.stepper_has_errest else 2

    def step(self, t, dt):
        run_kernels = self.backend.run_kernels
        rhs_with_postproc = self.system.rhs

        r1 = self._idxcurr
        r2, *rs = set(self._regidx) - {r1}

        # Evaluate the stages in the scheme
        for i, ci in enumerate(self.c):
            # Compute -∇·f
            rhs_with_postproc(t + ci*dt, r2 if i > 0 else r1, r2)

            # Fetch the appropriate RK accumulation kernels
            kerns = self._get_rkvdh2_kerns(i, r1, r2, *rs)

            # Bind the arguments
            for k in kerns:
                k.bind(dt=dt)

            # Execute
            run_kernels(kerns)

            # Swap
            r1, r2 = r2, r1

        # Return
        return (r2, *rs) if len(rs) else r2


class StdRK34Stepper(StdRKVdH2RStepper):
    stepper_name = 'rk34'
    stepper_order = 3

    a = [
        11847461282814 / 36547543011857,
        3943225443063 / 7078155732230,
        -346793006927 / 4029903576067
    ]

    b = [
        1017324711453 / 9774461848756,
        8237718856693 / 13685301971492,
        57731312506979 / 19404895981398,
        -101169746363290 / 37734290219643
    ]

    bhat = [
        15763415370699 / 46270243929542,
        514528521746 / 5659431552419,
        27030193851939 / 9429696342944,
        -69544964788955 / 30262026368149
    ]


class StdRK45Stepper(StdRKVdH2RStepper):
    stepper_name = 'rk45'
    stepper_order = 4

    a = [
        970286171893 / 4311952581923,
        6584761158862 / 12103376702013,
        2251764453980 / 15575788980749,
        26877169314380 / 34165994151039
    ]

    b = [
        1153189308089 / 22510343858157,
        1772645290293 / 4653164025191,
        -1672844663538 / 4480602732383,
        2114624349019 / 3568978502595,
        5198255086312 / 14908931495163
    ]

    bhat = [
        1016888040809 / 7410784769900,
        11231460423587 / 58533540763752,
        -1563879915014 / 6823010717585,
        606302364029 / 971179775848,
        1097981568119 / 3980877426909
    ]
