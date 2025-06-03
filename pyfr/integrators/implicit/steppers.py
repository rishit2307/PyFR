import math

from pyfr.integrators.implicit.base import BaseImplicitIntegrator

class BaseImplicitStepper(BaseImplicitIntegrator):
    pass

class BaseESDIRKStepper(BaseImplicitStepper):
    stepper_nregs = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def step(self, t, dt):
        for i, acoeff, ccoeff in enumerate(zip(self.a, self.c)):
            self.newtonsolver.init_stage(i, t + ccoeff*dt)
            self.newtonsolver.solve(i, acoeff, dt)

        

class ESDIRK32Stepper(BaseESDIRKStepper):
    stepper_name = 'esdirk32'
    nstages = 3
    gamma = (2 - math.sqrt(2))/2
    b2 = math.sqrt(2)/4

    a = [[gamma, gamma],
         [1 - b2 - gamma, b2, gamma]]
    
    b = [1 - b2 - gamma, b2, gamma]

    c = [2*gamma, 1]


