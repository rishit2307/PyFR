from math import prod

from pyfr.backends.base.generator import BaseKernelGenerator


class HIPKernelGenerator(BaseKernelGenerator):
    block1d = None
    block2d = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Specialise
        if self.ndim == 1:
            self._limits = 'if (_x < _nx)'
        else:
            self._limits = '''
                int _ysize = (_ny + blockDim.y - 1) / blockDim.y;
                int _ystart = threadIdx.y*_ysize;
                int _yend = (_ystart + _ysize > _ny) ? _ny : _ystart + _ysize;
                for (int _y = _ystart; _x < _nx && _y < _yend; _y++)
            '''

    def render(self):
        spec = self._render_spec()

        return f'''{spec}
               {{
                   int _x = blockIdx.x*blockDim.x + threadIdx.x;
                   #define X_IDX (_x)
                   #define X_IDX_AOSOA(v, nv) SOA_IX(X_IDX, v, nv)
                   #define BLK_IDX 0
                   #define BCAST_BLK(r, c, ld)  c
                   {self._limits}
                   {{
                       {self.body}
                   }}
                   #undef X_IDX
                   #undef X_IDX_AOSOA
                   #undef BLK_IDX
                   #undef BCAST_BLK
               }}'''

    def _render_spec(self):
        # We first need the argument list; starting with the dimensions
        kargs = [f'int {d}' for d in self._dims]

        # Now add any scalar arguments
        kargs.extend(f'{sa.dtype} {sa.name}' for sa in self.scalargs)

        # Finally, add the vector arguments
        for va in self.vectargs:
            # Views
            if va.isview:
                kargs.append(f'{va.dtype}* __restrict__ {va.name}_v')
                kargs.append(f'const int* __restrict__ {va.name}_vix')

                if va.ncdim == 2:
                    kargs.append(f'const int* __restrict__ {va.name}_vrstri')
            # Arrays
            else:
                # Intent in arguments should be marked constant
                const = 'const' if va.intent == 'in' else ''

                kargs.append(f'{const} {va.dtype}* __restrict__ {va.name}_v')

                if self.needs_ldim(va):
                    kargs.append(f'int ld{va.name}')

        # Determine the launch bounds for the kernel
        nthrds = prod(self.block1d if self.ndim == 1 else self.block2d)
        kattrs = f'__global__ __launch_bounds__({nthrds})'

        return '{0} void {1}({2})'.format(kattrs, self.name, ', '.join(kargs))
