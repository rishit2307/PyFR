import os
import re

import h5py
import numpy as np

from pyfr.mpiutil import get_comm_rank_root
from pyfr.util import file_path_gen


def write_pyfrms(path, data):
    # Save to disk
    with h5py.File(path, 'w', libver='latest') as f:
        for k in filter(lambda k: isinstance(k, str), data):
            f[k] = data[k]

        for p, q in filter(lambda k: isinstance(k, tuple), data):
            f[p].attrs[q] = data[p, q]


class NativeWriter:
    def __init__(self, intg, basedir, basename, prefix, *, extn='.pyfrs'):
        # Data prefix
        self.prefix = prefix

        # Our physical rank
        self.prank = intg.rallocs.prank

        # Append the relevant extension
        if not basename.endswith(extn):
            basename += extn

        # Output counter (incremented each time write() is called)
        self.fgen = file_path_gen(basedir, basename, intg.isrestart)
        next(self.fgen)

        # MPI info
        comm, rank, root = get_comm_rank_root()

        # Parallel I/O
        if (h5py.get_config().mpi and
            'PYFR_FORCE_SERIAL_HDF5' not in os.environ):
            print("Hils")
            self._write = self._write_parallel
        # Serial I/O
        else:
            self._write = self._write_serial

    def write(self, data, tcurr, metadata=None):
        # Determine the output path
        path = self.fgen.send(tcurr)

        # Delegate to _write to do the actual outputting
        self._write(path, data, metadata)

        # Return the path
        return path

    def _prepare_data_info(self, data):
        info = {}

        for k, v in data.items():
            info[f'{self.prefix}_{k}_p{self.prank}'] = (v.shape, v.dtype.str)

        return info

    def _write_parallel(self, path, data, metadata):
        comm, rank, root = get_comm_rank_root()

        info = self._prepare_data_info(data)

        # If we are the root rank then process any metadata
        if rank == root:
            data = dict(data)

            for k, v in metadata.items():
                if isinstance(v, str):
                    data[k] = np.array(v.encode(), dtype='S')
                    info[k] = ((), data[k].dtype.str)
                else:
                    data[k] = v
                    info[k] = (v.shape, v.dtype.str)
        elif metadata:
            raise ValueError('Metadata must be written by the root rank')

        # Distribute the data info to all of the ranks
        ginfo = comm.allgather(info)

        with h5py.File(path, 'w', driver='mpio', comm=comm) as f:
            # Parallel HDF5 requires that data sets be created collectively
            for minfo in ginfo:
                for name, (shape, dtype) in minfo.items():
                    f.create_dataset(name, shape, dtype=dtype)

            # Write out our local data
            for name, dat in zip(info, data.values()):
                fdata = f[name]

                if dat.shape:
                    nrows = len(dat)
                    rowsz = dat.nbytes // nrows
                    rstep = 2*1024**3 // rowsz

                    if rstep == 0:
                        raise IOError('Array is too large for parallel I/O')

                    for ix in range(0, nrows, rstep):
                        fdata[ix:ix + rstep] = dat[ix:ix + rstep]
                else:
                    fdata.write_direct(dat)

        # Wait for everyone to finish writing
        comm.barrier()

    def _write_serial(self, path, data, metadata):
        comm, rank, root = get_comm_rank_root()

        info = self._prepare_data_info(data)

        if rank != root:
            if metadata:
                raise ValueError('Metadata must be written by the root rank')

            # Send the info about our data to the root rank
            comm.gather(info, root=root)

            # Send the data itself
            for v in data.values():
                comm.Send(np.ascontiguousarray(v), root)
        else:
            with h5py.File(path, 'w') as f:
                # Collect info about what remote ranks want to write
                ginfo = comm.gather({}, root=root)

                # Write the metadata
                
                for k, v in metadata.items():
                 
                    if isinstance(v, str):
                        f[k] = np.array(v.encode(), dtype='S')
                    else:
                        f[k] = v

                # Write our local data
                for k, v in zip(info, data.values()):
                    f[k] = v

                # Receive and write the remote data
                for mrank, minfo in enumerate(ginfo):
                    for k, (shape, dtype) in minfo.items():
                        v = np.empty(shape, dtype=dtype)
                        comm.Recv(v, mrank)

                        f[k] = v

        # Wait for the root rank to finish writing
        comm.barrier()
