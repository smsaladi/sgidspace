#------------------------------------------------------------------------------
#
#  This file is part of sgidspace.
#
#  sgidspace is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  sgidspace is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with sgidspace.  If not, see <http://www.gnu.org/licenses/>.
#
#------------------------------------------------------------------------------
import gzip
import json
import random
import argparse

import numpy as np
import pandas as pd
from keras.utils import Sequence

IUPAC_CODES = list('ACDEFGHIKLMNPQRSTVWY*')

class FileCache():
    """A simple queue-like file cache
    """
    def __init__(self, fns, cache_size, read_file):
        self.fns = fns
        self.max_cache = cache_size
        self.read_file = read_file
        self._fc = []
        return

    def __getitem__(self, fn):
        for i, x in enumerate(self._fc):
            f, fdata = x
            if fn == f:
                # move to front
                if i != 0:
                    self._fc = self._fc.insert(0, self._fc.pop(i))
                return fdata

        if len(self) >= self.max_cache - 1:
            self._fc = self._fc[:(self.max_cache - 1)]

        fdata = self.read_file(fn)
        self._fc.insert(0, (fn, fdata))
        
        return fdata

    def __len__(self):
        return len(self._fc)


def read_json_lines(fn):
    with gzip.open(fn, 'rt') as fh:
        return [json.loads(l) for l in fh.readlines()]

class SGISequence(Sequence):
    def __init__(
            self,
            filenames,
            shard_count=None,
            shard_index=None,
            file_cache_size=10,
    ):
        filenames = [f for f in filenames if not f.endswith('.count')]
        if shard_count is not None:
            if shard_count > len(filenames):
                raise ValueError((
                    'shard_count must be <= the number of files for now. '
                    'requested {} shards but only found {} filenames'
                ).format(shard_count, len(filenames)))
            if shard_index is None:
                raise ValueError('if shard_count is not None, shard_index must not be None')

            filenames.sort()
            filenames = filenames[shard_index::shard_count]
        
        # set up sequence "directory"
        seqs = pd.Series(index=filenames, dtype=int)
        seqs.index.name = 'fn'
        for fn in filenames:
           with open(fn + '.count', 'r') as fh:
               seqs[fn] = int(fh.readline())
        seqs = seqs.apply(np.arange).explode()
        seqs.name = 'lidx'
        self.df = seqs.reset_index()
        
        # set up file cache
        self.file = FileCache(filenames, file_cache_size, read_json_lines)
        self.reset_count = 0
        return
    
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, i):
        fn, lidx = self.df.loc[i, ['fn', 'lidx']]
        return self.file[fn][lidx]
   
    def on_epoch_end(self):
        """
        Resets the starting index of this dataset to zero. Shuffles shard order
        and within shards
        """
        self.reset_count += 1
        # shuffle sequences
        self.df = self.df.sample(frac=1).reset_index(drop=True)

        # shuffle shards (seqs from each shard stay together)
        groups = [self.df for _, self.df in self.df.groupby('fn')]
        random.shuffle(groups)
        self.df = pd.concat(groups).reset_index(drop=True)
        return

    
def main():
    """For testing purposes
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('shard_files', nargs='+')
    args = parser.parse_args()
    seqs = SGISequence(args.shard_files)
    
    # these are always the same (for a given shard_file input)
    for i in range(5):
        print(seqs[i])
    
    seqs.on_epoch_end()
    # these will change everytime
    for i in range(5):
        print(seqs[i])

    return

if __name__ == '__main__':
    main()

