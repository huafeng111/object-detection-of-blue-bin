# Create training samples from polyroi labeling tools.

import os, cv2
import numpy as np
from roipoly import RoiPoly
from matplotlib import pyplot as plt

folder = 'data/training'
files = os.listdir(folder)
samples_dir = "data/train_samples"

for ii, filename in enumerate(files):
    print(f"Labeling {ii}: {filename}")
    # Read image.
    img = cv2.imread(os.path.join(folder,filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Label positive samples twice.
    positives = []
    for idx in range(1):
        # Get positive samples.
        # Display the image and use roipoly for labeling
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.set_title(f"Try {idx} to label positive for {filename}")
        my_roi = RoiPoly(fig=fig, ax=ax, color='r')
        # Get the image mask
        mask = my_roi.get_mask(img)
        mask = np.array(mask)
        positive_samples = img[mask]
        print(len(positive_samples))
        positives.append(positive_samples)
    positives = np.concatenate(positives, axis=0)
    np.random.shuffle(positives)

    # Label positive samples twice.
    negatives = []
    for idx in range(1):
        # Get negative samples.
        # display the image and use roipoly for labeling
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.set_title(f"Try {idx} to label negative for {filename}")
        my_roi = RoiPoly(fig=fig, ax=ax, color='r')
        # get the image mask
        mask = my_roi.get_mask(img)
        mask = np.array(mask)
        negative_samples = img[mask]
        print(len(negative_samples))
        negatives.append(negative_samples)
    negatives = np.concatenate(negatives, axis=0)
    np.random.shuffle(negatives)

    # Get the proper sample number.
    min_number = np.minimum(positives.shape[0], negatives.shape[0])
    positives = positives[:min_number]
    negatives = negatives[:min_number]

    # Save the positive samples.
    positive_samples_dir = os.path.join(samples_dir, "positive")
    if not os.path.exists(positive_samples_dir):
        os.makedirs(positive_samples_dir)
    np.save(os.path.join(positive_samples_dir, f"{filename}.npy"), positives)

    # Save the negative samples.
    negative_samples_dir = os.path.join(samples_dir, "negative")
    if not os.path.exists(negative_samples_dir):
        os.makedirs(negative_samples_dir)
    np.save(os.path.join(negative_samples_dir, f"{filename}.npy"), negatives)


