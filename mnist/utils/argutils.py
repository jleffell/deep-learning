"""
Basic argument parsing functionality.
"""

import argparse


# Parse command line arguments
def getArgs():
    """
Command line argument parser.

Parameters
------
None

Returns
------
args: argparse.Namespace object
    An object containing the necessary command line parameters
"""
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--save-model",
                    required=False,
                    default=False,
                    choices=['True', 'False'],
                    help="Save model to disk")

    ap.add_argument("-l", "--load-model",
                    required=False,
                    default=False,
                    choices=['True', 'False'],
                    help="Load pre-trained model")

    ap.add_argument("-w", "--weights",
                    type=str,
                    required=False,
                    default='mnist_weights.hdf5',
                    help="Load weights from file")

    ap.add_argument("-b", "--batch-size",
                    type=int,
                    required=False,
                    default=128,
                    help="Batch size")

    ap.add_argument("-e", "--epochs",
                    type=int,
                    required=False,
                    default=20,
                    help="Number of epochs")

    args = ap.parse_args()

    filetype = args.weights.split(".")[-1]

    print "filetype is: ", filetype, filetype == 'hdf5'

    if not filetype == 'h5' and not filetype == 'hdf5':
        ap.error("Please specify an hdf5 weight file (.h5 or .hdf5)")

    if not args.batch_size > 0:
        ap.error("Please choose batch size greater than zero")

    if not args.epochs > 0:
        ap.error("Please choose a number of epochs greater than zero")

    return args
