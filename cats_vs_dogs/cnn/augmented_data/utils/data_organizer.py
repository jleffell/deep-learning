# import the necessary packages
from imutils import paths
# import numpy as np
import argparse
# import cv2
import os
import errno
from random import shuffle

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source",
                required=True,
                help="Path to input dataset")

ap.add_argument("-d", "--destination",
                required=True,
                help="Destination of top level directory")

ap.add_argument("-t", "--train-num",
                required=False,
                default=1000,
                type=int,
                help="Number of training samples")

ap.add_argument("-v", "--val-num",
                required=False,
                default=400,
                type=int,
                help="Number of validation samples")

ap.add_argument("-f", "--shuffle",
                required=False,
                default=True,
                choices=['True', 'False'],
                help="Shuffle Data")

args = ap.parse_args()

# grab the list of images that we'll be describing
print("[INFO] describing images...")
imagePaths = list(paths.list_images(args.source))

if args.shuffle:
    print "Not shuffling data"
    shuffle(imagePaths)

train_num = args.train_num
val_num = args.val_num
nzfill = max(len(str(train_num)), len(str(val_num)))

ndogs = ncats = 0
ntot = train_num + val_num

# Create Directrory Tree and check that top level directory doesn't exist
cwd = os.getcwd()
top = os.path.join(cwd, args.destination)
try:
    os.makedirs(top)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

train_dir = os.path.join(top, "train")
val_dir = os.path.join(top, "validation")
train_dogs_dir = os.path.join(train_dir, "dogs")
train_cats_dir = os.path.join(train_dir, "cats")
val_dogs_dir = os.path.join(val_dir, "dogs")
val_cats_dir = os.path.join(val_dir, "cats")
os.makedirs(train_dogs_dir)
os.makedirs(train_cats_dir)
os.makedirs(val_dogs_dir)
os.makedirs(val_cats_dir)

for (i, imagePath) in enumerate(imagePaths):

    label = imagePath.split(os.path.sep)[-1].split(".")[0]

    if min(ndogs, ncats) == ntot:
        print "We have filled the requested data"
        break

    src = os.path.join(cwd, imagePath)

    if label == 'dog':
        ndogs += 1
        if ndogs <= train_num:
            dst = os.path.join(train_dogs_dir, "dog"
                               + str(ndogs).zfill(nzfill) + ".jpg")
            os.symlink(src, dst)
        elif ndogs <= ntot:
            dst = os.path.join(val_dogs_dir, "dog"
                               + str(ndogs - train_num).zfill(nzfill) + ".jpg")
            os.symlink(src, dst)

    if label == 'cat':
        ncats += 1

        if ncats <= train_num:
            dst = os.path.join(train_cats_dir, "cat"
                               + str(ncats).zfill(nzfill) + ".jpg")
            os.symlink(src, dst)
        elif ncats <= ntot:
            dst = os.path.join(val_cats_dir, "cat"
                               + str(ncats - train_num).zfill(nzfill) + ".jpg")
            os.symlink(src, dst)
