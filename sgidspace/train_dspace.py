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
import os
import subprocess
import glob

import tensorflow as tf
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
from keras.utils import plot_model, multi_gpu_model
from keras.optimizers import Nadam
import keras.backend as K 
import keras.backend.tensorflow_backend as tfback


from sgidspace.batch_processor import BatchProcessor
from sgidspace.architecture import build_network

# Monkey patching
# from sgidspace.sgikeras.models import patch_all, load_model
# patch_all()
from sgidspace.sgikeras.metrics import precision, recall, fmeasure

import numpy as np

from datetime import datetime

from load_outputs import load_outputs

start_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).
    # https://github.com/keras-team/keras/issues/13684#issuecomment-595054461
    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]
tfback._get_available_gpus = _get_available_gpus



class WeightsSaver(Callback):
    """From https://stackoverflow.com/a/44058144/2320823"""
    def __init__(self, N, fmt, **kwargs):
        self.N = N
        self.batch = 0
        self.fmt = fmt
        self.kwargs = kwargs

    def on_batch_end(self, batch, logs={}):
        if self.batch % self.N == self.N - 1:
            time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            name = self.fmt.format(**self.kwargs, t=time, b=batch)
            self.model.save(name)
        self.batch += 1


def get_callbacks(outdir):
    # Get callbacks

    tensorboard_callback = TensorBoard(
        log_dir=outdir + '/tfboard/run-' + start_time,
        batch_size=1024,
    )

    best_checkpoint_callback = ModelCheckpoint(
        '%s/model.run-%s.best.hdf5' % (outdir, start_time),
        monitor='val_loss',
        save_best_only=True,
        mode='auto',
        period=1,
        verbose=1
    )

    all_checkpoint_callback = ModelCheckpoint(
        '%s/model.run-%s.{epoch:02d}.hdf5' % (outdir, start_time),
        monitor='val_loss',
        save_best_only=False,
        mode='auto',
        period=1,
        verbose=1,
    )

    batch_checkpoint_callback = WeightsSaver(10000,
            fmt='{outdir}/model.run-{start}.t{t}.b{b}.h5',
            outdir=outdir,
            start=start_time,
    )

    callbacks = [
        tensorboard_callback,
        best_checkpoint_callback,
        all_checkpoint_callback,
        batch_checkpoint_callback
    ]

    return callbacks


def git_hash():
    dspace_dir = os.path.abspath(os.path.dirname(__file__) + "../../")
    try:
        git_hash = subprocess.check_output([
            "git", "--git-dir", dspace_dir + "/.git", "--work-tree", dspace_dir, "rev-parse", "HEAD"
        ]).strip()
        print('git hash:', git_hash)
    except subprocess.CalledProcessError:
        git_hash = 'git not available'
    return git_hash


def build_model(outputs):
    """
    Main function that calls layers within the architecture module
    """
    # Get run info
    input_layers, output_layers = build_network(outputs)
    model = Model(inputs=input_layers, outputs=output_layers)

    param_count = model.count_params()
    seed = 123456
    np.random.seed(seed)

    model.metadata = {
        'git_hash': git_hash(),
        'start_time': start_time,
        'param_count': param_count,
        'seed': seed
    }

    print("Number of parameters: " + str(param_count))

    losses = {}
    for o in outputs:
        print(o['name'])
        if o['type'] in ['multihot']:
            losses[o['name']] = 'binary_crossentropy'
        elif o['type'] in ['onehot', 'boolean', 'positional']:
            losses[o['name']] = 'categorical_crossentropy'
        elif o['type'] in ['numeric', 'embedding_autoencoder']:
            losses[o['name']] = 'mean_squared_error'

    metrics = {}
    for o in outputs:
        if o['type'] == 'onehot':
            metrics[o['name']] = ['top_k_categorical_accuracy', 'categorical_accuracy']
        elif o['type'] == 'boolean':
            metrics[o['name']] = 'categorical_accuracy'
        elif o['type'] == 'multihot':
            metrics[o['name']] = [precision, recall, fmeasure]
        elif o['type'] in ['numeric', 'embedding_autoencoder']:
            metrics[o['name']] = 'mean_squared_error'

    optimizer = Nadam(lr=0.001)
    if ',' in os.environ.get('CUDA_VISIBLE_DEVICES', ''):
        n_gpu = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
        model = multi_gpu_model(model, gpus=n_gpu)

    model.compile(
        optimizer=optimizer,
        loss=losses,
        metrics=metrics,
        loss_weights={o['name']: o['scale'] for o in outputs}
    )

    return model


def train_model(
        epochs,
        main_datadir,
        outdir,
        outputs,
        output_subdirectory=None,
):
    """
    Main function for setting up model and running training
    """
    # add subdirectory to output
    subdirectory = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    if output_subdirectory is not None:
        subdirectory = subdirectory + '_' + output_subdirectory
    outdir = os.path.join(outdir, subdirectory)
    os.system('mkdir -p {}'.format(outdir))
    print('output directory:', outdir)

    # Get class weights
    class_weights = []
    for o in outputs:
        if o['type'] in ('onehot', 'multihot'):
            class_weights.append({
                i: min(1, max(0.01, 1 / (w * o['classcount'])))
                for i, w in enumerate(o['class_freq'])
            })
        else:
            class_weights.append(None)

    # Get the dataloaders
    dataloader_train = BatchProcessor(
        glob.glob(os.path.join(main_datadir, 'train', '*.sqlite')),
        batch_size=512, outputs=outputs, floatx=K.floatx()
    )
    dataloader_validation = BatchProcessor(
        glob.glob(os.path.join(main_datadir, 'val', '*.sqlite')),
        batch_size=512, outputs=outputs, floatx=K.floatx()
    )

    model = build_model(outputs)

    # Draw model graph
    # plot_model(model, to_file=outdir + '/model.png', show_shapes=True)

    training_start_time = datetime.now()
    print("Started training at: " + training_start_time.strftime("%Y-%m-%d %H:%M:%S"))

    # Draw model graph
    # plot_model(model, to_file=outdir + '/model.png', show_shapes=True)

    model.fit_generator(
        dataloader_train,
        workers=6,
        max_queue_size=32,
        validation_data=dataloader_validation,
        use_multiprocessing=False,
        callbacks=get_callbacks(outdir),
        epochs=epochs
    )
    training_end_time = datetime.now()
    print("Ended training at: " + training_end_time.strftime("%Y-%m-%d %H:%M:%S"))
    return

def main():

    # Construct argument parser
    parser = argparse.ArgumentParser(description='DSPACE Model training.')
    parser.add_argument(
        '-w', '--datadir', type=str, default="data", help="Main data directory for all outputs"
    )
    parser.add_argument(
        '-o', '--output', default='/output', help='output path to write output files to'
    )
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs')
    args = parser.parse_args()

    # Construct outputs
    outputs = load_outputs('outputs.txt')

    # Train model
    train_model(args.epochs, args.datadir, args.output, outputs)


if __name__ == '__main__':
    main()
