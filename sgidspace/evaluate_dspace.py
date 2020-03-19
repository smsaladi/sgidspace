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
import os.path
import glob

import tensorflow as tf
import tensorflow.keras.backend as K

from sgidspace.batch_processor import BatchProcessor
from sgidspace.sgikeras.metrics import precision, recall, fmeasure
from load_outputs import load_outputs

def evaluate_model(
        model_fn,
        datafiles,
        outputs,
):
    """
    Main function for setting up model and running training
    """
    # Get the dataloaders
    dataloader = BatchProcessor(datafiles,
        batch_size=512, outputs=outputs, floatx=K.floatx()
    )

    model = tf.keras.models.load_model(model_fn,
        custom_objects={
            'precision': precision,
            'recall': recall,
            'fmeasure': fmeasure
    })

    model.evaluate(
        dataloader,
        workers=6,
        max_queue_size=32,
        use_multiprocessing=False,
    )
    return

def main():

    # Construct argument parser
    parser = argparse.ArgumentParser(description='DSPACE Model training.')
    parser.add_argument('model_fn', help='model h5 file')
    parser.add_argument('--shards', nargs='+')
    args = parser.parse_args()

    outputs = load_outputs("outputs.txt")

    # Train model
    evaluate_model(args.model_fn, args.shards, outputs)
    return


if __name__ == '__main__':
    main()
