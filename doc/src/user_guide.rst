.. highlight:: none

**********
User Guide
**********

For information on how to install PyFR see :ref:`installation`.

.. _running-pyfr:

Running PyFR
============

PyFR |release| uses three distinct file formats:

1. ``.ini`` --- configuration file
2. ``.pyfrm`` --- mesh file
3. ``.pyfrs`` --- solution file

The following commands are available from the ``pyfr`` program:

1. ``pyfr import`` --- convert a `Gmsh
   <http:http://geuz.org/gmsh/>`_ .msh file into a PyFR .pyfrm file.

   Example::

        pyfr import mesh.msh mesh.pyfrm

2. ``pyfr partition`` --- partition or repartition an existing mesh and
   associated solution files.

   Example::

       pyfr partition 2 mesh.pyfrm solution.pyfrs outdir/

   Here, the newly partitioned mesh and solution are placed into the
   directory `outdir`.  Multiple solutions can be provided.
   Time-average files can also be partitioned, too.

   For mixed grids it is usually necessary to provide weights for each
   element type.  Further details can be found in the
   :ref:`performance guide <perf mixed grids>`.

3. ``pyfr run`` --- start a new PyFR simulation. Example::

        pyfr run mesh.pyfrm configuration.ini

4. ``pyfr restart`` --- restart a PyFR simulation from an existing
   solution file. Example::

        pyfr restart mesh.pyfrm solution.pyfrs

   It is also possible to restart with a different configuration file.
   Example::

        pyfr restart mesh.pyfrm solution.pyfrs configuration.ini

5. ``pyfr export`` --- convert a PyFR ``.pyfrs`` file into an
   unstructured VTK ``.vtu`` or ``.pvtu`` file. If a ``-k`` flag is
   provided with an integer argument then ``.pyfrs`` elements are
   converted to high-order VTK cells which are exported, where the
   order of the VTK cells is equal to the value of the integer
   argument. Example::

        pyfr export -k 4 mesh.pyfrm solution.pyfrs solution.vtu

   If a ``-d`` flag is provided with an integer argument then
   ``.pyfrs`` elements are subdivided into linear VTK cells which are
   exported, where the number of sub-divisions is equal to the value of
   the integer argument. Example::

        pyfr export -d 4 mesh.pyfrm solution.pyfrs solution.vtu

   If no flags are provided then ``.pyfrs`` elements are converted to
   high-order VTK cells which are exported, where the order of the
   cells is equal to the order of the solution data in the ``.pyfrs``
   file.

   By default all of the fields in the ``.pyfrs`` file will be
   exported. If only a specific field is desired this can be specified
   with the ``-f`` flag; for example ``-f density -f velocity`` will
   only export the *density* and *velocity* fields.

PyFR can be run in parallel. To do so prefix ``pyfr`` with
``mpiexec -n <cores/devices>``. Note that the mesh must be
pre-partitioned, and the number of cores or devices must be equal to
the number of partitions.

.. _configuration-file:

Configuration File (.ini)
=========================

The .ini configuration file parameterises the simulation. It is written
in the `INI <http://en.wikipedia.org/wiki/INI_file>`_ format.
Parameters are grouped into sections. The roles of each section and
their associated parameters are described below. Note that both ``;`` and
``#`` may be used as comment characters.  Additionally, all parameter
values support environment variable expansion.

Backends
--------

These sections detail how the solver will be configured for a range of
different hardware platforms. If a hardware specific backend section is omitted,
then PyFR will fall back to built-in default settings.

.. toctree::
   :maxdepth: 3

   backends/backend.rst
   backends/backend-cuda.rst
   backends/backend-hip.rst
   backends/backend-opencl.rst
   backends/backend-openmp.rst

Systems
-------

These sections setup and control the physical system being solved, as well as
charateristics of the spatial and temporal schemes to be used.

.. toctree::
   :maxdepth: 3

   systems/constants.rst
   systems/solver.rst
   systems/solver-time-integrator.rst
   systems/solver-dual-time-integrator-multip.rst
   systems/solver-entropy-filter.rst
   systems/solver-artificial-viscosity.rst
   systems/soln-filter.rst
   systems/solver-interfaces.rst

Boundary and Initial Conditions
-------------------------------

These sections allow users to set the boundary and initial
conditions of calculations.

.. toctree::
   :maxdepth: 3

   boundary-initial-conditions/soln-bcs.rst
   boundary-initial-conditions/soln-ics.rst

Nodal Point Sets
----------------

Solution point sets must be specified for each element type that is used and
flux point sets must be specified for each interface type that is used. If
anti-aliasing is enabled then quadrature point sets for each element and
interface type that is used must also be specified. For example, a 3D mesh
comprised only of prisms requires a solution point set for prism elements and
flux point sets for quadrilateral and triangular interfaces.

.. toctree::
   :maxdepth: 3

   nodal-point-sets/solver-interfaces-line.rst
   nodal-point-sets/solver-interfaces-quad.rst
   nodal-point-sets/solver-interfaces-tri.rst
   nodal-point-sets/solver-elements-quad.rst
   nodal-point-sets/solver-elements-tri.rst
   nodal-point-sets/solver-elements-hex.rst
   nodal-point-sets/solver-elements-tet.rst
   nodal-point-sets/solver-elements-pri.rst
   nodal-point-sets/solver-elements-pyr.rst

Plugins
-------

Plugins allow for powerful additional functionality to be swapped
in and out. There are two classes of plugin available; solution
plugins which are prefixed by ``soln-`` and solver plugins which
are prefixed by ``solver-``. It is possible to create multiple
instances of the same solution plugin by appending a suffix, for
example::

    [soln-plugin-writer]
    ...

    [soln-plugin-writer-2]
    ...

    [soln-plugin-writer-three]
    ...

[soln-plugin-writer]
^^^^^^^^^^^^^^^^^^^^
Periodically write the solution to disk in the pyfrs format.
Parameterised with

1. ``dt-out`` --- write to disk every ``dt-out`` time units:

    *float*

   plugins/soln-plugin-ascent.rst
   plugins/soln-plugin-dtstats.rst
   plugins/soln-plugin-fluid-force.rst
   plugins/soln-plugin-fwh.rst
   plugins/soln-plugin-integrate.rst
   plugins/soln-plugin-nancheck.rst
   plugins/soln-plugin-pseudostats.rst
   plugins/soln-plugin-residual.rst
   plugins/soln-plugin-sampler.rst
   plugins/soln-plugin-tavg.rst
   plugins/soln-plugin-writer.rst

    *string*

3. ``basename`` --- pattern of output names:

    *string*

4. ``post-action`` --- command to execute after writing the file:

    *string*

5. ``post-action-mode`` --- how the post-action command should be
   executed:

    ``blocking`` | ``non-blocking``

4. ``region`` --- region to be written, specified as either the
   entire domain using ``*``, a combination of the geometric shapes
   specified in :ref:`regions`, or a sub-region of elements that have
   faces on a specific domain boundary via the name of the domain
   boundary:

    ``*`` | ``shape(args, ...)`` | *string*

Example::

    [soln-plugin-writer]
    dt-out = 0.01
    basedir = .
    basename = files-{t:.2f}
    post-action = echo "Wrote file {soln} at time {t} for mesh {mesh}."
    post-action-mode = blocking
    region = box((-5, -5, -5), (5, 5, 5))

[soln-plugin-fluidforce-*name*]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Periodically integrates the pressure and viscous stress on the boundary
labelled ``name`` and writes out the resulting force and moment (if requested)
vectors to a CSV file. Parameterised with

1. ``nsteps`` --- integrate every ``nsteps``:

    *int*

2. ``file`` --- output file path; should the file already exist it
   will be appended to:

    *string*

3. ``header`` --- if to output a header row or not:

    *boolean*

4. ``morigin`` --- origin used to compute moments (optional):

    ``(x, y, [z])``

5. ``quad-deg-{etype}`` --- degree of quadrature rule for fluid force
   integration, optionally this can be specified for different element types:

    *int*

6. ``quad-pts-{etype}`` --- name of quadrature rule (optional):

    *string*

Example::

    [soln-plugin-fluidforce-wing]
    nsteps = 10
    file = wing-forces.csv
    header = true
    quad-deg = 6
    morigin = (0.0, 0.0, 0.5)

[soln-plugin-nancheck]
^^^^^^^^^^^^^^^^^^^^^^

Periodically checks the solution for NaN values. Parameterised with

1. ``nsteps`` --- check every ``nsteps``:

    *int*

Example::

    [soln-plugin-nancheck]
    nsteps = 10

[soln-plugin-residual]
^^^^^^^^^^^^^^^^^^^^^^

Periodically calculates the residual and writes it out to a CSV file.
Parameterised with

1. ``nsteps`` --- calculate every ``nsteps``:

    *int*

2. ``file`` --- output file path; should the file already exist it
   will be appended to:

    *string*

3. ``header`` --- if to output a header row or not:

    *boolean*

4. ``norm`` --- sets the degree and calculates an :math:`L_p` norm,
    default is ``2``:

    *float* | ``inf``

Example::

    [soln-plugin-residual]
    nsteps = 10
    file = residual.csv
    header = true
    norm = inf

[soln-plugin-dtstats]
^^^^^^^^^^^^^^^^^^^^^^

Write time-step statistics out to a CSV file. Parameterised with

1. ``flushsteps`` --- flush to disk every ``flushsteps``:

    *int*

2. ``file`` --- output file path; should the file already exist it
   will be appended to:

    *string*

3. ``header`` --- if to output a header row or not:

    *boolean*

Example::

    [soln-plugin-dtstats]
    flushsteps = 100
    file = dtstats.csv
    header = true

[soln-plugin-pseudostats]
^^^^^^^^^^^^^^^^^^^^^^^^^

Write pseudo-step convergence history out to a CSV file. Parameterised
with

1. ``flushsteps`` --- flush to disk every ``flushsteps``:

    *int*

2. ``file`` --- output file path; should the file already exist it
   will be appended to:

    *string*

3. ``header`` --- if to output a header row or not:

    *boolean*

Example::

    [soln-plugin-pseudostats]
    flushsteps = 100
    file = pseudostats.csv
    header = true

[soln-plugin-sampler]
^^^^^^^^^^^^^^^^^^^^^

Periodically samples specific points in the volume and writes them out
to a CSV file. Parameterised with

1. ``nsteps`` --- sample every ``nsteps``:

    *int*

2. ``samp-pts`` --- list of points to sample:

    ``[(x, y), (x, y), ...]`` | ``[(x, y, z), (x, y, z), ...]``

3. ``format`` --- output variable format:

    ``primitive`` | ``conservative``

4. ``file`` --- output file path; should the file already exist it
   will be appended to:

    *string*

5. ``header`` --- if to output a header row or not:

    *boolean*

Example::

    [soln-plugin-sampler]
    nsteps = 10
    samp-pts = [(1.0, 0.7, 0.0), (1.0, 0.8, 0.0)]
    format = primitive
    file = point-data.csv
    header = true

[soln-plugin-tavg]
^^^^^^^^^^^^^^^^^^

Time average quantities. Parameterised with

1. ``nsteps`` --- accumulate the average every ``nsteps`` time steps:

    *int*

2. ``dt-out`` --- write to disk every ``dt-out`` time units:

    *float*

3. ``tstart`` --- time at which to start accumulating average data:

    *float*

4. ``mode`` --- output file accumulation mode:

    ``continuous`` | ``windowed``

    In continuous mode each output file contains average data from
    ``tstart`` until the current time. In windowed mode each output
    file only contains average data for the most recent ``dt-out`` time
    units. The default is ``windowed``.

5. ``std-mode`` --- standard deviation reporting mode:

    ``summary`` | ``all``

    If to output full standard deviation fields or just summary
    statistics.  In lieu of a complete field, summary instead reports
    the maximum and average standard deviation for each field. The
    default is ``summary`` with ``all`` doubling the size of the
    resulting files.

6. ``basedir`` --- relative path to directory where outputs will be
   written:

    *string*

7. ``basename`` --- pattern of output names:

    *string*

8. ``precision`` --- output file number precision:

    ``single`` | ``double``

    The default is ``single``. Note that this only impacts the output,
    with statistic accumulation *always* being performed in double
    precision.

9. ``region`` --- region to be written, specified as either the
   entire domain using ``*``, a combination of the geometric shapes
   specified in :ref:`regions`, or a sub-region of elements that have
   faces on a specific domain boundary via the name of the domain
   boundary:

    ``*`` | ``shape(args, ...)`` | *string*

10. ``avg``-*name* --- expression to time average, written as a
    function of the primitive variables and gradients thereof;
    multiple expressions, each with their own *name*, may be specified:

    *string*

11. ``fun-avg``-*name* --- expression to compute at file output time,
    written as a function of any ordinary average terms; multiple
    expressions, each with their own *name*, may be specified:

    *string*

Example::

    [soln-plugin-tavg]
    nsteps = 10
    dt-out = 2.0
    mode = windowed
    basedir = .
    basename = files-{t:06.2f}

    avg-u = u
    avg-v = v
    avg-uu = u*u
    avg-vv = v*v
    avg-uv = u*v

    fun-avg-upup = uu - u*u
    fun-avg-vpvp = vv - v*v
    fun-avg-upvp = uv - u*v

.. _integrate-plugin:

[soln-plugin-integrate]
^^^^^^^^^^^^^^^^^^^^^^^

Integrate quantities over the compuational domain. Parameterised with:

1. ``nsteps`` --- calculate the integral every ``nsteps`` time steps:

    *int*

2. ``file`` --- output file path; should the file already exist it
   will be appended to:

    *string*

3. ``header`` --- if to output a header row or not:

    *boolean*

4. ``quad-deg`` --- degree of quadrature rule (optional):

    *int*

5. ``quad-pts-{etype}`` --- name of quadrature rule (optional):

    *string*

6. ``norm`` --- sets the degree and calculates an :math:`L_p` norm,
    otherwise standard integration is performed:

    *float* | ``inf`` | ``none``

7. ``region`` --- region to integrate, specified as either the
   entire domain using ``*`` or a combination of the geometric shapes
   specified in :ref:`regions`:

    ``*`` | ``shape(args, ...)``

8. ``int``-*name* --- expression to integrate, written as a function of
   the primitive variables and gradients thereof, the physical coordinates
   [x, y, [z]] and/or the physical time [t]; multiple expressions,
   each with their own *name*, may be specified:

    *string*

Example::

    [soln-plugin-integrate]
    nsteps = 50
    file = integral.csv
    header = true
    quad-deg = 9
    vor1 = (grad_w_y - grad_v_z)
    vor2 = (grad_u_z - grad_w_x)
    vor3 = (grad_v_x - grad_u_y)

    int-E = rho*(u*u + v*v + w*w)
    int-enst = rho*(%(vor1)s*%(vor1)s + %(vor2)s*%(vor2)s + %(vor3)s*%(vor3)s)

[solver-plugin-source]
^^^^^^^^^^^^^^^^^^^^^^

Injects solution, space (x, y, [z]), and time (t) dependent
source terms with

1. ``rho`` --- density source term for ``euler`` | ``navier-stokes``:

    *string*

2. ``rhou`` --- x-momentum source term for ``euler`` | ``navier-stokes``
   :

    *string*

3. ``rhov`` --- y-momentum source term for ``euler`` | ``navier-stokes``
   :

    *string*

4. ``rhow`` --- z-momentum source term for ``euler`` | ``navier-stokes``
   :

    *string*

5. ``E`` --- energy source term for ``euler`` | ``navier-stokes``
   :

    *string*

6. ``p`` --- pressure source term for ``ac-euler`` |
   ``ac-navier-stokes``:

    *string*

7. ``u`` --- x-velocity source term for ``ac-euler`` |
   ``ac-navier-stokes``:

    *string*

8. ``v`` --- y-velocity source term for ``ac-euler`` |
   ``ac-navier-stokes``:

    *string*

9. ``w`` --- w-velocity source term for ``ac-euler`` |
   ``ac-navier-stokes``:

    *string*

Example::

    [solver-plugin-source]
    rho = t
    rhou = x*y*sin(y)
    rhov = z*rho
    rhow = 1.0
    E = 1.0/(1.0+x)

[solver-plugin-turbulence]
^^^^^^^^^^^^^^^^^^^^^^^^^^

Injects synthetic eddies into a region of the domain. Parameterised with

1. ``avg-rho`` --- average free-stream density:

    *float*

2. ``avg-u`` --- average free-stream velocity magnitude:

    *float*

3. ``avg-mach`` --- averge free-stream Mach number:

    *float*

4. ``turbulence-intensity`` --- percentage turbulence intensity:

    *float*

5. ``turbulence-length-scale`` --- turbulent length scale:

    *float*

6. ``sigma`` --- standard deviation of Gaussian sythetic eddy profile:

    *float*

7. ``centre`` --- centre of plane on which synthetic eddies are injected:

    (*float*, *float*, *float*)

8. ``y-dim`` --- y-dimension of plane:

    *float*

9. ``z-dim`` --- z-dimension of plane:

    *float*

10. ``rot-axis`` --- axis about which plane is rotated:

    (*float*, *float*, *float*)

11. ``rot-angle`` --- angle in degrees that plane is rotated:

    *float*

Example::

    [solver-plugin-turbulence]
    avg-rho = 1.0
    avg-u = 1.0
    avg-mach = 0.2
    turbulence-intensity = 1.0
    turbulence-length-scale = 0.075
    sigma = 0.7
    centre = (0.15, 2.0, 2.0)
    y-dim = 3.0
    z-dim = 3.0
    rot-axis = (0, 0, 1)
    rot-angle = 0.0

[soln-plugin-fwh]
^^^^^^^^^^^^^^^^^

Use Ffowcs Williams--Hawkings equation to approximate far field noise in a
uniformly moving medium:

1. ``tstart`` --- time at which to start sampling, default is ``0``:

    *float*

2. ``dt`` --- time step between samples:

    *float*

3. ``file`` --- output file path; should the file already exist it
   will be appended to:

    *string*

4. ``header`` --- if to output a header row or not:

    *boolean*

5. ``surface`` --- a region the surface of which is saple for the FWH sovler,
   only use a combination of the geometric shapes specified in :ref:`regions`:

   ``shape(args, ...)``

6. ``quad-deg`` --- degree of surface quadrature rule (optional):

    *int*

7. ``quad-pts-{etype}`` --- name of surface quadrature rule (optional):

    *string*

8. ``observer-pts`` --- the obversation point in the far field at which noise is
   approximated:

   ``[(x, y), (x, y), ...]`` | ``[(x, y, z), (x, y, z), ...]``

9. ``rho, u, v, (w), p, (c)`` --- the constant far field properties of the
   flow. For incompressible calculations the sound speed ``c`` and the dnsity
   ``rho`` must be given:

    *float*

Example::

    [soln-plugin-fwh]
    file = fwh.csv
    region = box((1, -5), (10, 5))
    header = true
    tstart = 10
    dt = 1e-2
    observer-pts = [(1, 10), (1, 30), (1, 100), (1, 300)]

    rho = 1
    u = 1
    v = 0
    p = 10

Regions
-------

Certain plugins are capable of performing operations on a subset of the
elements inside the domain. One means of constructing these element
subsets is through parameterised regions. Note that an element is
considered part of a region if *any* of its nodes are found to be
contained within the region. Supported regions:

Rectangular cuboid ``box(x0, x1)``
  A rectangular cuboid defined by two diametrically opposed vertices.
  Valid in both 2D and 3D.

Conical frustum ``conical_frustum(x0, x1, r0, r1)``
  A conical frustum whose end caps are at *x0* and *x1* with radii
  *r0* and *r1*, respectively. Only valid in 3D.

Cone ``cone(x0, x1, r)``
  A cone of radius *r* whose centre-line is defined by *x0* and *x1*.
  Equivalent to ``conical_frustum(x0, x1, r, 0)``. Only valid in 3D.

Cylinder ``cylinder(x0, x1, r)``
  A circular cylinder of radius *r* whose centre-line is defined by
  *x0* and *x1*. Equivalent to ``conical_frustum(x0, x1, r, r)``.
  Only valid in 3D.

Cartesian ellipsoid ``ellipsoid(x0, a, b, c)``
  An ellipsoid centred at *x0* with Cartesian coordinate axes whose
  extents in the *x*, *y*, and *z* directions are given by *a*, *b*,
  and *c*, respectively. Only valid in 3D.

Sphere ``sphere(x0, r)``
  A sphere centred at *x0* with a radius of *r*. Equivalent to
  ``ellipsoid(x0, r, r, r)``. Only valid in 3D.

All region shapes also support rotation.  In 2D this is accomplished by
passing a trailing `rot=angle` argument where `angle` is a rotation
angle in degrees; for example ``box((-5, 2), (2, 0), rot=30)``.
In 3D the syntax is `rot=(phi, theta, psi)` and corresponds to a
sequence of Euler angles in the so-called *ZYX convention*.  Region
expressions can also be added and subtracted together  arbitrarily.
For example
``box((-10, -10, -10), (10, 10, 10)) - sphere((0, 0, 0), 3)`` will
result in a cube-shaped region with a sphere cut out of the middle.

Additional Information
----------------------

The :ref:`INI<configuration-file>` file format is very versatile. A feature that
can be useful in defining initial conditions is the substitution feature and
this is demonstrated in the :ref:`soln-plugin-integrate` plugin example.

To prevent situations where you have solutions files for unknown
configurations, the contents of the ``.ini`` file are added as an attribute
to ``.pyfrs`` files. These files use the HDF5 format and can be
straightforwardly probed with tools such as h5dump.

In several places within the ``.ini`` file expressions may be used. As well as
the constant ``pi``, expressions containing the following functions are
supported:

1. ``+, -, *, /`` --- basic arithmetic

2. ``sin, cos, tan`` --- basic trigonometric functions (radians)

3. ``asin, acos, atan, atan2`` --- inverse trigonometric functions

4. ``exp, log`` --- exponential and the natural logarithm

5. ``tanh`` --- hyperbolic tangent

6. ``pow`` --- power, note ``**`` is not supported

7. ``sqrt`` --- square root

8. ``abs`` --- absolute value

9. ``min, max`` --- two variable minimum and maximum functions,
   arguments can be arrays
