# import the necessary packages
from imutils import paths
import numpy as np
import argparse
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

ap.add_argument("-e", "--test-num",
                required=False,
                default=400,
                type=int,
                help="Number of test samples")

ap.add_argument("-f", "--shuffle",
                action="store_true",
                help="Shuffle Data")

args = ap.parse_args()

train_num = args.train_num
val_num = args.val_num
test_num = args.test_num
shuffle_data = args.shuffle

# grab the list of images that we'll be describing
print("[INFO] describing images...")

imagePaths = list(paths.list_images(args.source))

if shuffle_data:
    shuffle(imagePaths)

# Determine level of zero padding to image files
nzfill = max(max(len(str(train_num)), len(str(val_num))), len(str(test_num))) + 1

ntot = train_num + val_num + test_num

# Create Directrory Tree and check that top level directory doesn't exist
cwd = os.getcwd()
top = os.path.join(cwd, args.destination)
try:
    os.makedirs(top)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

train_dogs_dir = os.path.join(top, "train", "dogs")
train_cats_dir = os.path.join(top, "train", "cats")
val_dogs_dir = os.path.join(top, "validation", "dogs")
val_cats_dir = os.path.join(top, "validation", "cats")
test_dogs_dir = os.path.join(top, "test", "dogs")
test_cats_dir = os.path.join(top, "test", "cats")

dirs = [train_dogs_dir, val_dogs_dir, test_dogs_dir, train_cats_dir, val_cats_dir, test_cats_dir]
for dir in dirs:
    os.makedirs(dir)

ndogs = ncats = ndata = 0

for (i, imagePath) in enumerate(imagePaths):

    label = imagePath.split(os.path.sep)[-1].split(".")[0]
    num = np.int0(imagePath.split(os.path.sep)[-1].split(".")[1])
    
    target_dir = None
    offset = 0

    if label == 'dog':
        
        if shuffle_data:
            num = ndogs
            ndogs += 1
        
        if num < train_num:
            target_dir = train_dogs_dir
        elif train_num <= num < train_num + val_num:
            target_dir = val_dogs_dir
            offset = train_num
        elif train_num + val_num <= num < ntot:
            target_dir = test_dogs_dir
            offset = train_num + val_num

    if label == 'cat':

        if shuffle_data:
            num = ncats
            ncats += 1
        
        if num < train_num:
            target_dir = train_cats_dir
        elif train_num <= num < train_num + val_num:
            target_dir = val_cats_dir
            offset = train_num
        elif train_num + val_num <= num < ntot:
            target_dir = test_cats_dir
            offset = train_num + val_num

    if target_dir is not None:

        ndata += 1

        dst = os.path.join(target_dir, label
                           + str(num - offset).zfill(nzfill) + ".jpg")
        src = os.path.join(cwd, imagePath)
        os.symlink(src, dst)
        
        # Check to see if we have filled the requested amount of data
        if ndata == 2*ntot:
            print "All requested data have been assigned"
            break
