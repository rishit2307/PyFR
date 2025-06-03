import re

from pyfr.integrators.dual.phys import BaseDualController, BaseDualStepper
from pyfr.integrators.std import BaseStdController, BaseStdStepper
from pyfr.integrators.std.gmres import GMRESmultip
from pyfr.integrators.implicit import BaseImplicitController, BaseImplicitStepper
from pyfr.integrators.implicit.newton import BaseNonLinearSolver
from pyfr.util import subclass_where


def get_integrator(backend, systemcls, rallocs, mesh, initsoln, cfg):
    form = cfg.get('solver-time-integrator', 'formulation', 'std')
    
    if cfg.get('solver-time-integrator', 'multip', 'false') == 'true':
        print('Hi from multip')
        return GMRESmultip(backend, systemcls, rallocs, mesh, initsoln, cfg)


    # if cfg.get('solver-time-integrator', 'scheme') in implicit:
    #     return GMRESmultip(backend, systemcls, rallocs, mesh, initsoln, cfg)

    if form == 'std':
        cn = cfg.get('solver-time-integrator', 'controller')
        sn = cfg.get('solver-time-integrator', 'scheme')

        cc = subclass_where(BaseStdController, controller_name=cn)
        sc = subclass_where(BaseStdStepper, stepper_name=sn)
    elif form == 'dual':
        cn = cfg.get('solver-time-integrator', 'controller')
        sn = cfg.get('solver-time-integrator', 'scheme')

        cc = subclass_where(BaseDualController, controller_name=cn)
        sc = subclass_where(BaseDualStepper, stepper_name=sn)
    elif form == 'implicit':
        cn = cfg.get('solver-time-integrator', 'controller')
        sn = cfg.get('solver-time-integrator', 'scheme')

        cc = subclass_where(BaseImplicitController, controller_name=cn)
        sc = subclass_where(BaseImplicitStepper, stepper_name=sn)    

    # Determine the integrator name
    name = '_'.join([form, cn, sn, 'integrator'])
    name = re.sub('(?:^|_|-)([a-z])', lambda m: m[1].upper(), name)

    # Composite the base classes together to form a new type
    integrator = type(name, (cc, sc), dict(name=name))

    # Construct and return an instance of this new integrator class
    return integrator(backend, systemcls, rallocs, mesh, initsoln, cfg)
