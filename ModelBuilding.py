# load base model
vgg16_weight_path = '../input/keras-pretrained-models/vgg16_weights_tf_dim_order ing_tf_kernels_notop.h5'
base_model = VGG16(
weights=vgg16_weight_path, include_top=False,
input_shape=IMG_SIZE + (3,) )

NUM_CLASSES = 1
model = Sequential() model.add(base_model)
model.add(layers.Flatten()) model.add(layers.Dropout(0.5))
model.add(layers.Dense(NUM_CLASSES, activation='sigmoid'))
model.layers[0].trainable = False
model.compile(
loss='binary_crossentropy', optimizer=RMSprop(lr=1e-4),
metrics=['accuracy']
)
model.summary()
