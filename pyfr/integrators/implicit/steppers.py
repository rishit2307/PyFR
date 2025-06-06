import math

from pyfr.integrators.implicit.base import BaseImplicitIntegrator

class BaseImplicitStepper(BaseImplicitIntegrator):
    pass

class BaseESDIRKStepper(BaseImplicitStepper):
    stepper_nregs = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def step(self, t, dt):
        self.newtonsolver.init_step(t)

        for i, (acoeffs, ccoeff) in enumerate(zip(self.a, self.c), start=1):
            acoeffs = [acoeff*dt for acoeff in acoeffs]
            self.newtonsolver.solve(t + ccoeff*dt, acoeffs, i)
        
        if not self.fsal:
            bcoeffs = [bt*dt for bt in self.b]
            self.newtonsolver.obtain_solution(bcoeffs)
        print(f'time is {t}')

class ESDIRK32Stepper(BaseESDIRKStepper):
    stepper_name = 'esdirk32'
    nstages = 3
    gamma = (2 - math.sqrt(2))/2
    b2 = math.sqrt(2)/4
    fsal = False

    a = [[gamma, gamma],
         [1 - b2 - gamma, b2, gamma]]
    
    b = [1 - b2 - gamma, b2, gamma]

    c = [2*gamma, 1]


class TrapeziumStepper(BaseESDIRKStepper):
    stepper_name = 'trapezium'
    nstages = 2
    fsal = True

    a = [[0.5, 0.5]]
    c = [1]

