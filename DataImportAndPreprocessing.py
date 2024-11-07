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

#-------------------------------------------------------------------------------------------------------------------------#

def plot_samples(X, y, labels_dict, n=50):
"""
Creates a gridplot for desired number of images (n) from the specified set
""" for index in range(len(labels_dict)):
imgs = X[np.argwhere(y == index)][:n] j = 10 i = int(n/j)
plt.figure(figsize=(15,6))
c = 1
for img in imgs:
plt.subplot(i,j,c) , plt.imshow(img[0]),
plt.xticks([]) ,plt.yticks([])
c += 1
plt.suptitle('Tumor: {}'.format(labels_dict[index])) plt.show()
plot_samples(X_train, y_train, labels, 30)
RATIO_LIST = [] for set in (X_train, X_test, X_val):
for img in set:
RATIO_LIST.append(img.shape[1]/img.shape[0])
plt.hist(RATIO_LIST)
plt.title('Distribution of Image Ratios') plt.xlabel('Ratio
Value') plt.ylabel('Count') plt.show()

#----------------------------------------------------------------------------------------------------------------------------#

def crop_imgs(set_name, add_pixels_value=0):
"""
Finds the extreme points on the image and crops the rectangular out of them """ set_new = []
for img in set_name:
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
# threshold the image, then perform a series of erosions +
# dilations to remove any small regions of noise
thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
thresh = cv2.erode(thresh, None, iterations=2) thresh = cv2.dilate(thresh, None, iterations=2)
# find contours in thresholded image, then grab the largest one
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CH AIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = max(cnts, key=cv2.contourArea)
# find the extreme points
extLeft = tuple(c[c[:, :, 0].argmin()][0])
extRight = tuple(c[c[:, :, 0].argmax()][0])
extTop = tuple(c[c[:, :, 1].argmin()][0])
extBot = tuple(c[c[:, :, 1].argmax()][0])
ADD_PIXELS = add_pixels_value
new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft
[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
set_new.append(new_img)
return np.array(set_new)


plt.figure(figsize=(15,6)) plt.subplot(141)
plt.imshow(img) plt.xticks([]) plt.yticks([])
plt.title('Step 1. Get the original image') plt.subplot(142)
plt.imshow(img_cnt) plt.xticks([]) plt.yticks([])
plt.title('Step 2. Find the biggest contour') plt.subplot(143)
plt.imshow(img_pnt) plt.xticks([]) plt.yticks([])
plt.title('Step 3. Find the extreme points') plt.subplot(144)
plt.imshow(new_img) plt.xticks([]) plt.yticks([])
plt.title('Step 4. Crop the image') plt.show()
