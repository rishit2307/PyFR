import numpy as np
import copy
from pyfr.integrators.std.base import BaseStdIntegrator
from pyfr.util import memoize
from pyfr.mpiutil import get_comm_rank_root, mpi

class BaseStdStepper(BaseStdIntegrator):
    def collect_stats(self, stats):
        super().collect_stats(stats)

        # Total number of RHS evaluations
        stats.set('solver-time-integrator', 'nfevals', self._stepper_nfevals)


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

class Trapezoidal(BaseStdStepper):
    stepper_name = 'trapezium'
    stepper_has_errest = False
    stepper_nregs = 3
    stepper_order = 1

    @property
    def _stepper_nfevals(self):
        return self.nsteps
    
    def arnoldi(self, Q, k, eps, dt, t, eletype):
        add, rhs_with_postproc = self._add, self.system.rhs
        r0, r1, r2 = self._regidx
        netype = len(self.system.ele_types)
        h, qnorm = dict(), dict()

        comm, rank, root = get_comm_rank_root()

        for i in range(netype):
            self.system.ele_banks[i][r2].set(Q[i][..., k])

        # r1 = Un + eps*Q
        add(0.0, r1, eps, r2, 1.0, r0)

        # r1 = R(Un+eps*Q)
        rhs_with_postproc(t, r1, r1)      

        # r1 = R(Un+eps*Q)/eps + Q/dt
        add(-1.0/(2*eps), r1, 1.0/dt, r2)

        # r2 = R(Un)
        rhs_with_postproc(t, r0, r2)
        # r1 = R(Un+eps*Q)/eps + Q/dt  - R(Un)/eps 
        add(1.0, r1, 1.0/(2*eps), r2)

        for etype in eletype:
            h[etype] = np.zeros(k+2)
            qnorm[etype] = 0
        
        q = [self.system.ele_banks[i][r1].get() for i in range(netype)]

        for j in range(0, k+1):
            for i, etype in enumerate(self.system.ele_types):
                h[etype][j] = np.dot(q[i].flatten(), Q[i][..., j].flatten())
                h[etype][j] = comm.allreduce(h[etype][j], op=mpi.SUM)
                q[i] -= h[etype][j] * Q[i][..., j]
                

        for i, etype in enumerate(self.system.ele_types):
            qnorm[etype] = (np.linalg.norm(q[i]))**2
        
        for etype in eletype:
            qnorm[etype] = np.sqrt(comm.allreduce(qnorm[etype], mpi.SUM))

        for i, etype in enumerate(self.system.ele_types):
            h[etype][k+1] = qnorm[etype]
            q[i] /= h[etype][k+1]

        return h, q
    
    def apply_givens_rotation(self, h, cs, sn, k):
        for i in range(k):
            temp = cs[i] * h[i] + sn[i] * h[i+1]

            h[i+1] = -sn[i] * h[i] + cs[i] * h[i+1]
            h[i] = temp
        
        cs_k, sn_k = self.givens_rotation(h[k], h[k+1])

        h[k] = cs_k * h[k] + sn_k * h[k+1]
        h[k+1] = 0.0


        return h, cs_k, sn_k

    def givens_rotation(self, v1, v2):
        tt = np.sqrt(v1**2 + v2**2)
        cs = v1/tt
        sn = v2/tt

        return cs, sn


    def step(self, t, dt):
        add, rhs_with_postproc = self._add, self.system.rhs
        r0, r1, r2 = self._regidx
        ntol, m = 0.01, 10

        # MPI info
        comm, rank, root = get_comm_rank_root()

        # Initialise
        netype = len(self.system.ele_types)
        nnorm = 1.0
        rnorm, l2u, normele = dict(), dict(), dict()
        err = dict()
        if rank == 0:
            eletype = self.system.ele_types
        else:
            eletype = None

        ltol = 3e-8
        eletype = comm.bcast(eletype, root=root)
        for etp in eletype:
            rnorm[etp] = 0
            l2u[etp] = 0
            normele[etp] = 0
            err[etp] = 0.0


        nonlin_iter = 0

        while np.amax(nnorm) > ntol:
            e1 = [np.zeros(m+1) for i in range(netype)]
            for i in range(netype):
                e1[i][0] =  1.0
            
            sn = [np.zeros(m) for i in range(netype)]
            cs = [np.zeros(m) for i in range(netype)]

            y = [[] for _ in range(netype)]

            x = copy.deepcopy(self.system.ele_scal_upts(r2))

            for i, etype in enumerate(self.system.ele_types):
                l2u[etype] = (np.linalg.norm(self.soln[i]))**2

            for etype in eletype:
                l2u[etype] = np.sqrt(comm.allreduce(l2u[etype], op=mpi.SUM))
            
            max_etp = max(l2u, key=l2u.get)
            
            eps = 1e-7*l2u[max_etp]
            
            # r1 = Un+1 = Un + eps*dU0
            add(0.0, r1, eps, r2, 1.0, r0)

            # r1 = R(Un+1)
            rhs_with_postproc(t, r1, r1)
            # r1 = R(Un+1)/eps + dU0/dt
            add(-1.0/(2*eps), r1, 1/dt, r2)

            # r2 = R(Un)
            rhs_with_postproc(t, r0, r2)

            # r1 = R(Un+1)/eps + dU0/dt - R(Un)/eps
            add(1.0, r1, 1.0/(2*eps), r2)

            # r1 = b - Adu0: first step of GMRES
            add(-1.0, r1, 1.0, r2)
            
            for i, etype in enumerate(self.system.ele_types):
                rnorm[etype] = (np.linalg.norm(self.system.ele_banks[i][r1].get()))**2
            
            for etype in eletype:
                rnorm[etype] = np.sqrt(comm.allreduce(rnorm[etype], op=mpi.SUM))

            Q = [[] for i in range(netype)]
            
            for i, etype in enumerate(self.system.ele_types):
                nupts, nvars, neles = self.system.ele_shapes[i]
                Q[i] = np.zeros((nupts, nvars, neles, m+1))
                Q[i][..., 0] = self.system.ele_banks[i][r1].get() / rnorm[etype]
            
            
            H = [np.zeros((m+1, m)) for i in range(netype)]

            beta = [rnorm[etype]*e1[i] for i, etype
                    in enumerate(self.system.ele_types)]
            
            for k in range(m):
                h, q = self.arnoldi(Q, k, eps, dt, t, eletype)
                
                for i, etype in enumerate(self.system.ele_types):
                    H[i][:k+2, k], Q[i][..., k+1] = h[etype], q[i]

                    H[i][:k+2, k], cs[i][k], sn[i][k] = self.apply_givens_rotation(H[i][:k+2, k], cs[i], sn[i] ,k)

                    beta[i][k+1] = -sn[i][k] * beta[i][k]
                    beta[i][k] = cs[i][k] * beta[i][k]

                    err[etype] = abs(beta[i][k+1]) / l2u[etype]

                for etype in eletype:
                    error = np.amax(comm.allreduce(err[etype], op=mpi.SUM))
                
                if error < ltol:
                    print(f'GMRES converged in {k} iterations, error is {err}')
                    break

            if k == m-1:
                print(f'GMRES did not converge in {m} iterations, error is {err}')
            for i, etype in enumerate(self.system.ele_types):
                y[i] =  np.linalg.solve(H[i][:k+1, :k+1], beta[i][:k+1])
                x[i] += Q[i][..., :k+1] @ y[i]
                self.system.ele_banks[i][r2].set(x[i])

            # r1 = R(Un)
            rhs_with_postproc(t, r0, r1)

            # r0 = Un+1 = Un + dUn
            add(1.0, r0, 1.0, r2)

            # r1 = R(Un)/2 + Du/Dt
            add(-1/2, r1, 1/dt, r2)

            # r2 = R(Un+1)
            rhs_with_postproc(t, r0, r2)

            # r1 = R(Un)/2 + Du/dt + R(Un+1)/2
            add(1.0, r1, -1/2, r2)
            
            for i, etype in enumerate(self.system.ele_types):
                normele[etype] = (np.linalg.norm(self.system.ele_banks[i][r1].get()))**2

            for etype in eletype:
                normele[etype] = np.sqrt(comm.allreduce(normele[etype], mpi.SUM))

            nnorm = normele[max(normele, key=normele.get)]
            
            for i in range(netype):
                self.system.ele_banks[i][r2].set(x[i])

            nonlin_iter+= 1
            if rank == root:
                print(nnorm)
                print(nonlin_iter)

        return r0
    
# class BDF2(Trape):
#     stepper_name = 'bdf2'
#     stepper_has_errest = False
#     stepper_nregs = 4
#     stepper_order = 2


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
