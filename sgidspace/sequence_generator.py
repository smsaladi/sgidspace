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
import sys
import gzip
import json
import random
import argparse
import zlib
import time
import random
import threading

import numpy as np
import pandas as pd
from keras.utils import Sequence

import sqlalchemy as db
from sqlalchemy.orm import Session

from convert_sqlitedb import Entry

IUPAC_CODES = list('ACDEFGHIKLMNPQRSTVWY*')

class DBEngine():
    def __init__(self, sqlitefn):
        self.threadid = threading.get_ident()
        self.sqlitefn = sqlitefn
        self._start_conn()
        
    def _start_conn(self):
        engine = db.create_engine('sqlite:///' + self.sqlitefn,
            connect_args={'check_same_thread': False})
        self.s = Session(bind=engine)

    def __len__(self):
        return self.s.query(Entry).count()

    def __getitem__(self, i):
        if threading.get_ident() != self.threadid:
            # self._start_conn()
            pass
        i = int(i) + 1
        e = self.s.query(Entry).get(i)
        return json.loads(e.data)

class SGISequence(Sequence):
    def __init__(
            self,
            filenames,
            shard_count=None,
            shard_index=None,
    ):

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

        self.files = {f: DBEngine(f) for f in filenames}

        # set up sequence "directory"
        seqs = pd.Series(index=filenames, dtype=int)
        seqs.index.name = 'fn'
        for f in self.files.keys():
            seqs[f] = len(self.files[f])
        seqs = seqs.apply(np.arange).explode()
        seqs.name = 'lidx'
        self.df = seqs.reset_index()

        self.n_epoch = 0
        return
    
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, i):
        fn, lidx = self.df.loc[i, ['fn', 'lidx']]
        return self.files[fn][lidx]

    def on_epoch_end(self):
        """
        Resets the starting index of this dataset to zero. Shuffles sequences
        """
        self.n_epoch += 1
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        return

def main():
    """For testing purposes
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('sqlitedb', nargs='+')
    args = parser.parse_args()

    seqs = SGISequence(args.sqlitedb)
    
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

