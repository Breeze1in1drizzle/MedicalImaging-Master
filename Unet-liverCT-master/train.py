from resnet import *
from data import *
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=1, min_lr=1e-5)
callbacks = [ModelCheckpoint, reduce_lr]

myGene = trainGenerator(2,'data/data2','Image','Label',data_gen_args,save_to_dir = None)
model = build_model(64)
# model = unet(lr_init=lr_init, lr_decay=lr_decay)
model_checkpoint = ModelCheckpoint('unet_liverCT_fulliamge4.hdf5', monitor='loss',verbose=1, save_best_only=True)
print ("training............")
# h = model.fit(
#     np.expand_dims(X_train,axis=-1),
#     np.expand_dims(y_train,axis=-1),
#     batch_size=4,
#     epochs=50,
#     validation_split=0.02,
#     callbacks=callbacks)
hi = model.fit_generator(myGene, steps_per_epoch=900, epochs=8, callbacks=[model_checkpoint])

plt.plot(hi.history['acc'][1:])
plt.plot(hi.history['loss'][1:],color="r", linestyle="--", marker="*")
plt.title('model')
plt.ylabel('loss acc')
plt.xlabel('epoch')
plt.legend(['train','Validation'], loc='upper left')
plt.savefig("re.jpg")

