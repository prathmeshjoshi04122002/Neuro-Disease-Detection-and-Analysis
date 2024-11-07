def load_data(dir_path, img_size=(100,100)):
"""
Load resized images as np.arrays to workspace
""" X = [] y = [] i = 0 labels = dict() for path in
tqdm(sorted(os.listdir(dir_path))): if not path.startswith('.'):
labels[i] = path
for file in os.listdir(dir_path + path):
if not file.startswith('.'):
img = cv2.imread(dir_path + path + '/' + file)
X.append(img)
y.append(i)
i += 1 X = np.array(X)
y = np.array(y)
print(f'{len(X)} images loaded from {dir_path} directory.') return X, y, labels

def plot_confusion_matrix(cm, classes,normalize=False, c map=plt.cm.Blues):
title='Confusion matrix',
"""
This function prints and plots the confusion matrix.
Normalization can be applied by setting `normalize=True`.
"""
plt.figure(figsize = (6,6))
plt.imshow(cm, interpolation='nearest', cmap=cmap) plt.title(title)
plt.colorbar()
tick_marks = np.arange(len(classes)) plt.xticks(tick_marks, classes,
rotation=90) plt.yticks(tick_marks, classes) if normalize:
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
thresh = cm.max() / 2. cm = np.round(cm,2)
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
plt.text(j, i, cm[i, j],
horizontalalignment="center",
color="white" if cm[i, j] > thresh else "black") plt.tight_layout()
TEST_DIR = 'TEST/'
VAL_DIR = 'VAL/'
IMG_SIZE = (224,224)

# use predefined function to load the image data into workspace
X_train, y_train, labels = load_data(TRAIN_DIR, IMG_SIZE)
X_test, y_test, _ = load_data(TEST_DIR, IMG_SIZE) X_val, y_val, _ = load_data(VAL_DIR, IMG_SIZE)
