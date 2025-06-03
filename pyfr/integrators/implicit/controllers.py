from pyfr.integrators.implicit.base import BaseImplicitIntegrator

class BaseImplicitController(BaseImplicitIntegrator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Solution filtering frequency
        self._fnsteps = self.cfg.getint('soln-filter', 'nsteps', '0')

        # Fire off any event handlers if not restarting
        if not self.isrestart:
            self._run_plugins()

        
    def _accept_step(self):
        self.tcurr += self._dt
        self.nacptsteps += 1
        self.nacptchain += 1

        self._invalidate_caches()

        # Run any plugins
        self._run_plugins()

        # Clear the step info
        self.stepinfo = []

class ImplicitNoneController(BaseImplicitController):
    controller_name = 'none'
    controller_has_variable_dt = False

    @property
    def controller_needs_errest(self):
        return False

    def advance_to(self, t):
        if t < self.tcurr:
            raise ValueError('Advance time is in the past')
        
        while self.tcurr < t:
            # Take the physical step
            self.step(self.tcurr, self._dt)

            # We are not adaptive, so accept every step
            self._accept_step()
