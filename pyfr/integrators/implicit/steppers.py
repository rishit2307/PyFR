import math

from pyfr.integrators.implicit.base import BaseImplicitIntegrator
from pyfr.mpiutil import get_comm_rank_root

class BaseImplicitStepper(BaseImplicitIntegrator):
    pass

class BaseESDIRKStepper(BaseImplicitStepper):
    stepper_nregs = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def step(self, t, dt):
        self.newtonsolver.init_step(t)
        comm, rank, root = get_comm_rank_root()
        

        for i, (acoeffs, ccoeff) in enumerate(zip(self.a, self.c), start=1):
            acn = [acoeffs[-1]*dt]
            acn += [ac/acoeffs[-1] for ac in acoeffs]

            self.newtonsolver.solve(t + ccoeff*dt, acn, i)
        
        if not self.fsal:
            bcoeffs = [bt*dt for bt in self.b]
            self.newtonsolver.obtain_solution(bcoeffs)
        
        self._idxcurr = self.newtonsolver.store_current_solution()
        if rank == root:
            print(f'Physical time is {t}')

class ESDIRK32Stepper(BaseESDIRKStepper):
    stepper_name = 'esdirk2'
    nstages = 3
    gamma = (2 - math.sqrt(2))/2
    b2 = math.sqrt(2)/4
    fsal = True

    a = [[gamma, gamma],
         [1 - b2 - gamma, b2, gamma]]
    c = [2*gamma, 1]


class TrapeziumStepper(BaseESDIRKStepper):
    stepper_name = 'trapezium'
    nstages = 2
    fsal = True

    a = [[0.5, 0.5]]
    c = [1]


class ESDIRK3Stepper(BaseESDIRKStepper):
    stepper_name = 'esdirk3'
    nstages = 4
    fsal = True

    a = [[1767732205903/4055673282236, 1767732205903/4055673282236],
         [2746238789719/10658868560708, -640167445237/6845629431997, 
          1767732205903/4055673282236],
          [1471266399579/7840856788654, -4482444167858/7529755066697, 
           11266239266428/11593286722821, 1767732205903/4055673282236]]
    
    b = a[-1]

    c = [1767732205903/2027836641118, 3/5, 1]


class ESDIRK4Stepper(BaseESDIRKStepper):
    stepper_name = 'esdirk4'
    nstages = 5
    fsal = True

    a = [[1/4, 1/4], [8611/62500, -1743/31250, 1/4], 
         [5012029/34652500, -654441/2922500, 174375/388108, 1/4], 
         [15267082809/155376265600, -71443401/120774400, 730878875/902184768, 2285395/8070912, 1/4], 
         [82889/524892, 0 ,15625/83664, 69875/102672, -2260/8211, 1/4]]
    b = a[-1]
    c = [1/2, 83/250, 31/50, 17/20, 1]