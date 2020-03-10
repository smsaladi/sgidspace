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
import argparse

import numpy as np

from sgidspace.sequence_generator import SGISequence, IUPAC_CODES
from sgidspace.filters import get_nested_key
from sgidspace.keywords import extract_keywords
from sgidspace.gene_names import process_gene_name
from load_outputs import load_outputs

import keras.backend as K


def flatten_ec(ecnum):
    if ecnum is None:
        return None
    elif ecnum is '':
        return ''

    out = []
    x = ecnum.split('.')
    for i in range(len(x)):
        if x[i] != '-':
            out.append('.'.join(x[:(i+1)]))

    return out


def transform_record(data, data_dicts):
    """
    modify records from the form they take on disk to the format used by batch processor
    """
    data = data.copy()
    if 'translation_description' in data:
        data['translation_description_keywords'] = extract_keywords(data['translation_description'])

    if 'gene_name' in data:
        data['gene_name'] = [process_gene_name(data['gene_name'])]

    if 'ec_number' in data:
        data['ec_number_ecflat'] = flatten_ec(data['ec_number'])

    if 'gene3d' in data:
        x = []
        for v in data['gene3d']:
            x += flatten_ec(v)
        data['gene3d_ecflat'] = list(np.unique(x))

    if 'diamond' in data:
        data['diamond_cluster_id'] = [data['diamond']['cluster_id']]

    for field in data_dicts:
        if field in data:
            dtransform = data_dicts[field]
            if type(data[field]) is list:
                s = set()
                for x in data[field]:
                    for v in dtransform.get(str(x), []):
                        s.add(v)
                        data[field + '_dict'] = list(s)
            else:
                data[field + '_dict'] = dtransform.get(str(data[field]), [])

    return data


class BatchProcessor(SGISequence):
    """
    Generates batches of data for training

    SGISequence is used as a starting point which is subsequently
    transformed and grouped into batches of size `batch_size`.
    """
    def __init__(
            self,
            seq_shard_files,
            batch_size,
            outputs,
            file_cache_size=10,
            inference=False,
            from_embed=False,
    ):
        super().__init__(filenames=seq_shard_files, file_cache_size=file_cache_size)
        self.batch_size = batch_size

        self.outputs = outputs
        self.inference = inference
        self.input_symbols = {label: i for i, label in enumerate(IUPAC_CODES)}
        self.from_embed = from_embed

        if from_embed:
            self.esize = 256

        self.class_index = {}
        data_dicts = {}
        for o in outputs:
            if 'class_labels' in o:
                self.class_index[o['name']] = {label: i for i, label in enumerate(o['class_labels'])}
            if type(o['datafun']) is dict:
                data_dicts[o['field']] = o['datafun']
        self.data_dicts = data_dicts

        return

    def __len__(self):
        return np.ceil(super().__len__()/self.batch_size).astype(int)

    def __getitem__(self, batch_idx):
        start = batch_idx * self.batch_size
        stop = start + self.batch_size
        records = [super(BatchProcessor, self).__getitem__(i) for i in range(start, stop)]
        return self.format_batch(records)

    def format_batch(self, records):
        # initialize input
        X = {}
        if self.from_embed:
            X['embedding'] = np.zeros(
                [
                    len(records),
                    self.esize,
                ],
                dtype=K.floatx(),
            )
        else:
            X['sequence_input'] = np.zeros(
                [
                    len(records),
                    2000,
                    len(IUPAC_CODES),
                ],
                dtype=K.floatx(),
            )
        Y = {}

        # initialize output buffer
        for o in self.outputs:
            dtype = K.floatx()
            shape = [len(records), o['classcount']]
            Y[o['name']] = np.zeros(shape, dtype=dtype)

        # Copy record information
        for i, record in enumerate(records):
            record = transform_record(record, self.data_dicts)

            if self.from_embed:
                X['embedding'][i,:] = np.array(record['embedding'], dtype=K.floatx())
            else:
                # input_sequence
                input_sequence = record['protein_sequence']
                for sequence_index in range(len(input_sequence)):
                    symbol_index = self.input_symbols.get(input_sequence[sequence_index])
                    if symbol_index is not None:
                        X['sequence_input'][i, sequence_index, symbol_index] = 1

                # add zero padding for the rest
                aai = self.input_symbols.get("*")
                X['sequence_input'][i, sequence_index + 1:, aai] = 1

                for o in self.outputs:
                    output_name = o['name']
                    r = get_nested_key(record, o['name'])

                    if r is not None:
                        if o['type'] == 'numeric':
                            Y[output_name][i] = r
                        elif o['type'] == 'boolean':
                            Y[output_name][i, int(r)] = 1
                        elif o['type'] in ['onehot', 'multihot']:
                            for label in r:
                                index = self.class_index[output_name].get(str(label))
                                if index is not None:
                                    Y[output_name][i, index] = 1
                                elif o['type'] == 'onehot':
                                    Y[output_name][i, o['classcount'] - 1] = 1
                    else:
                        if o['type'] == 'onehot':
                            Y[output_name][i, o['classcount'] - 1] = 1
        
        if self.inference:
            return X, records
        
        return X, Y


def main():
    """For testing purposes
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('shard_files', nargs='+')
    parser.add_argument('--batch_id', default=1)
    parser.add_argument('--batch_size', default=8)
    args = parser.parse_args()
    
    outputs = load_outputs('outputs.txt') 
    batch = BatchProcessor(args.shard_files,
            args.batch_size,
            outputs)

    print(len(batch))

    # this will always be the same for a given shard_file input    
    print(batch[args.batch_id])
    
    batch.on_epoch_end()
    # these will change everytime
    print(batch[args.batch_id])

    return

if __name__ == '__main__':
    main()

