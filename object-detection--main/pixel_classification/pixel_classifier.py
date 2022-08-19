'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''

import os
import numpy as np
from generate_rgb_data import read_pixels


class PixelClassifier():
  def __init__(self):
    '''
      Initilize your classifier with any parameters and attributes you need
    '''
    # Prepare train and val datasets.
    self.load_train_data()
    self.load_val_data()
    self.W = None

    folder_path = os.path.dirname(os.path.abspath(__file__))
    self.model_path = os.path.join(folder_path, 'model.ckpt.npy')
    # ######### If timeout, then comment the following parts.
    # # Train and test Logistic Regression model.
    #self.train(self.train_samples, self.train_labels)
    # # Save model.
    # np.save(self.model_path, self.W)

  def classify(self,x):
    '''
      Classify a set of pixels into red, green, or blue

      Inputs:
        X: n x 3 matrix of RGB values
      Outputs:
        y: n x 1 vector of with {1,2,3} values corresponding to {red, green, blue}, respectively
    '''
    ################################################################
    # YOUR CODE AFTER THIS LINE
    self.W = np.load(self.model_path)
    y = self.predict(x)

    # YOUR CODE BEFORE THIS LINE
    ################################################################
    return y

  ################### Train and predict model ###################
  def train(self, x, y):
     self.W , self.loss_graph= self.gradient_process(x, y,100,1.0,0.01)

  def predict(self, x):
    value = - x @ self.W
    v = self.softmax(value)
    return 1+np.argmax(v, axis=1)

  ################### Load data ###################
  def load_train_data(self):
    try:
      # Training sample paths.
      self._train_red_path = "data/training/red"
      self._train_green_path = "data/training/green"
      self._train_blue_path = "data/training/blue"
      # Read training sampels.
      self._train_red_samples = read_pixels(self._train_red_path)
      self._train_green_samples = read_pixels(self._train_green_path)
      self._train_blue_samples = read_pixels(self._train_blue_path)

      # Training labels.
      self._train_red_labels = np.array([1 for _ in self._train_red_samples])
      self._train_green_labels = np.array([2 for _ in self._train_green_samples])
      self._train_blue_labels = np.array([3 for _ in self._train_blue_samples])

      # Ensemble samples and labels.
      self.train_samples = np.vstack([self._train_red_samples, self._train_green_samples, self._train_blue_samples])
      self.train_labels = np.hstack([self._train_red_labels, self._train_green_labels, self._train_blue_labels])

      # Shuffle training data.
      indices = np.random.permutation(len(self.train_samples))
      self.train_samples = self.train_samples[indices]
      self.train_labels = self.train_labels[indices]
    except:
      pass

  def load_val_data(self):
    try:
      # Validation sample paths.
      self._val_red_path = "data/validation/red"
      self._val_green_path = "data/validation/green"
      self._val_blue_path = "data/validation/blue"
      # Read validation sampels.
      self._val_red_samples = read_pixels(self._val_red_path)
      self._val_green_samples = read_pixels(self._val_green_path)
      self._val_blue_samples = read_pixels(self._val_blue_path)
      # Validation labels.
      self._val_red_labels = np.array([1 for _ in self._val_red_samples])
      self._val_green_labels = np.array([2 for _ in self._val_green_samples])
      self._val_blue_labels = np.array([3 for _ in self._val_blue_samples])
      # Ensemble samples and labels.
      self.val_samples = np.vstack([self._val_red_samples, self._val_green_samples, self._val_blue_samples])
      self.val_labels = np.hstack([self._val_red_labels, self._val_green_labels, self._val_blue_labels])
    except:
      pass

  ######## Utility to train multi-class LR ########
  def softmax(self, x):
    """Softmax function to normalize to 1."""
    max_x = np.max(x, axis=1, keepdims=True)
    x2 = x - max_x
    return np.exp(x2) / np.sum(np.exp(x2), axis=1, keepdims=True)
  
  def loss_function(self, x, y, j):
    """Get loss for multi-class."""
    dot = - x @ j
    exp=np.exp(dot)
    sum_exp=np.sum( exp, axis=1)
    value = x @ j @ y.T
    remind = np.sum(np.log(sum_exp))
    return (np.trace(value) + remind)* (1/x.shape[0])

  def gradient(self, x, y, j, L_rate):
    """Gradient for multi-class logistic regression."""
    value = - x @ j
    value_dot = self.softmax(value)
    return 2 * L_rate * j+x.T @ (y - value_dot)*1/x.shape[0]

  def gradient_process(self, x, y, max_step, sin, learning_rate):
    """Gradient descent for multi-class logistic regression."""
    y = np.eye(3)[y-1]
    shape = np.zeros((x.shape[1], y.shape[1]))
    traning_step,loss_value=[],[]
    loss_graph = { 'loss': loss_value, 'step': traning_step}
    times=0
    while times < max_step:
      # Update weights per run.
      shape -= sin * self.gradient(x, y, shape, learning_rate)
      # Record the training history.
      traning_step.append(times)
      loss_value.append(self.loss_function(x, y, shape))
      times += 1
    return  shape,loss_graph
