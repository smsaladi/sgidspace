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

from sgidspace.batch_processor import make_batch_processor
from sgidspace.architecture import build_network
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.utils import plot_model, multi_gpu_model
from keras.optimizers import Nadam

# Monkey patching
# from sgidspace.sgikeras.models import patch_all, load_model
# patch_all()
from sgidspace.sgikeras.metrics import precision, recall, fmeasure

import numpy as np

from datetime import datetime

from load_outputs import load_outputs

start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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

    latest_checkpoint_callback = ModelCheckpoint(
        '%s/model.run-%s.latest.hdf5' % (outdir, start_time),
        monitor='val_loss',
        save_best_only=False,
        mode='auto',
        period=1
    )

    callbacks = [
        tensorboard_callback,
        best_checkpoint_callback,
        latest_checkpoint_callback,
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
        classflags=None
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
    dataloader_train = make_batch_processor(
        main_datadir, 'train', 512, outputs, classflags=classflags
    )
    dataloader_validation = make_batch_processor(
        main_datadir, 'val', 512, outputs, classflags=classflags
    )

    model = build_model(outputs)

    # Draw model graph
    plot_model(model, to_file=outdir + '/model.png', show_shapes=True)

    training_start_time = datetime.now()
    print("Started training at: " + training_start_time.strftime("%Y-%m-%d %H:%M:%S"))

    # Draw model graph
    plot_model(model, to_file=outdir + '/model.png', show_shapes=True)

    model.fit_generator(
        dataloader_train,
        steps_per_epoch=1000,
        workers=1,
        validation_data=dataloader_validation,
        validation_steps=100,
        use_multiprocessing=False,
        callbacks=get_callbacks(outdir),
        epochs=epochs
    )
    training_end_time = datetime.now()
    print("Ended training at: " + training_end_time.strftime("%Y-%m-%d %H:%M:%S"))


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
    parser.add_argument('--tigrfam_filter', type=str, help='Skip records containing at least one of the listed tifgrfams. (comma separated)')
    args = parser.parse_args()

    # Construct outputs
    outputs = load_outputs('outputs.txt')

    # Construct classflags
    classflags = None
    if args.tigrfam_filter:
        classflags = {"tigrfams": args.tigrfam_filter.split(',')}

    # Train model
    train_model(args.epochs, args.datadir, args.output, outputs, classflags=classflags)


if __name__ == '__main__':
    main()
