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
import json
import gzip

import logging
logger = logging.getLogger()

import numpy as np
import keras.backend as K

from load_outputs import load_outputs
from sgidspace.sequence_generator import SGISequence, IUPAC_CODES

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('shard_files', nargs='+')
    parser.add_argument('--tigrfam_filter', nargs='+',
            help='Skip records containing at least one of the listed tifgrfams. (space separated)')
    args = parser.parse_args()
    
    outputs = load_outputs('outputs.txt')

    classflags = None
    if args.tigrfam_filter:
        classflags = {"tigrfams": args.tigrfam_filter.split(',')}
    
    gen = (r for r in SGISequence(args.shard_files))
    gen = filter_records(gen, outputs, classflags=classflags)

    for r in gen:
        print(json.dumps(r))
    return


def get_nested_key(d, key, default=None):
    keys = key.split(".")
    for k in keys:
        if k not in d:
            return default
        d = d[k]
    return d


def set_nested_key(d, key, value):
    keys = key.split(".")
    for k in keys[:-1]:
        if k not in d:
            return None
        d = d[k]
    d[keys[-1]] = value


def filter_max_len(gen, length):
    """
    A generator which passes through all valued in the input generator `gen` so
    long as their protein_sequence length is <= the maximum feature length
    specified in the hyper parameters.
    """
    for record in gen:
        if len(record['protein_sequence']) > length:
            continue

        yield record


def filter_invalid_sequence(gen, source_name, valid_alphabet):
    valid_alphabet = set(valid_alphabet)

    for record in gen:
        if source_name not in record:
            continue

        if source_name == 'protein_sequence':
            record[source_name] = record[source_name].replace('*', '')

        extra_characters = set(record[source_name]) - valid_alphabet
        if extra_characters:
            logger.debug('sequence contained extra characters: {}'.format(extra_characters))
            continue

        yield record


def record_hasvalue(record, field, values):
    if field in record:
        if type(record[field]) is list:
            for x in values:
                if x in record[field]:
                    return True
        else:
            return(record[field] == values)
    return False


def filter_classflags(gen, classflags):
    for record in gen:
        ok = True
        for field in classflags:
            if record_hasvalue(record, field, classflags[field]):
                ok = False
                break
        if ok:
            yield record


def filter_uncharacterized(gen, outputs):
    for record in gen:
        for output in outputs:
            if output['type'] == 'multihot' and output['name'] in record and len(record[output['name']]) > 0:
                yield record
                break


def filter_records(gen, outputs, classflags=None):
    gen = filter_invalid_sequence(gen, 'protein_sequence', IUPAC_CODES)
    gen = filter_max_len(gen, 2000)

    if classflags:
        gen = filter_classflags(gen, classflags)

    o = []
    for output in outputs:
        if output['name'] not in ['gene_name', 'translation_description_keywords']:
            o.append(output)
        gen = filter_uncharacterized(gen, o)

    return gen


if __name__ == '__main__':
    main()

