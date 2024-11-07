demo_datagen = ImageDataGenerator(
rotation_range=15, width_shift_range=0.05,
height_shift_range=0.05, rescale=1./255,
shear_range=0.05,
brightness_range=[0.1, 1.5], horizontal_flip=True,
vertical_flip=True
)
os.mkdir('preview') x =
X_train_crop[0]
x = x.reshape((1,) + x.shape)
i = 0
for batch in demo_datagen.flow(x, batch_size=1, save_to_dir='preview', save_pr efix='aug_img',
save_format='jpg'): i += 1 if i > 20: break
plt.imshow(X_train_crop[0]) plt.xticks([])
plt.yticks([])
plt.title('Original Image')
plt.show()
plt.figure(figsize=(15,6)) i = 1 for img in
os.listdir('preview/'):
img = cv2.cv2.imread('preview/' + img) img = cv2.cvtColor(img,
cv2.COLOR_BGR2RGB) plt.subplot(3,7,i) plt.imshow(img)
plt.xticks([]) plt.yticks([]) i += 1 if i > 3*7: break
plt.suptitle('Augemented Images') plt.show()

#--------------------------------------------------------------------------------------------------------------------#

TRAIN_DIR = 'TRAIN_CROP/'
VAL_DIR = 'VAL_CROP/'
train_datagen = ImageDataGenerator( rotation_range=15,
width_shift_range=0.1,
height_shift_range=0.1,
shear_range=0.1,
brightness_range=[0.5, 1.5], horizontal_flip=True,
vertical_flip=True,
preprocessing_function=preprocess_input
)
test_datagen = ImageDataGenerator( preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory( TRAIN_DIR,
color_mode='rgb', target_size=IMG_SIZE, batch_size=32,
class_mode='binary', seed=RANDOM_SEED
)

validation_generator = test_datagen.flow_from_directory(VAL_DIR,
color_mode='rgb', target_size=IMG_SIZE, batch_size=16,
class_mode='binary', seed=RANDOM_SEED)
