import numpy as np from
import tqdm import cv2
import os
iimport shutil
im import itertools
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer from
sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot from plotly import tools
from keras.preprocessing.image import ImageDataGenerator from
keras.applications.vgg16 import VGG16, preprocess_input from keras import layers
from keras.models import Model, Sequential from keras.optimizers
import Adam, RMSprop from keras.callbacks import EarlyStopping
init_notebook_mode(connected=True)
RANDOM_SEED = 123
