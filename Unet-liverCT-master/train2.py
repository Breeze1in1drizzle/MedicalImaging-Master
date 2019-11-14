from data import *
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
from Utils import  *
from sklearn.model_selection import train_test_split

X, y = get_data()
print("=========Split train sets and test sets=========")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=2018)
del X, y
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
print ("training............")
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=1, min_lr=1e-5)
callbacks = [ModelCheckpoint, reduce_lr]

# myGene = trainGenerator(2,'data/data2','Image','Label',data_gen_args,save_to_dir = None)
model = build_model(64)
model_checkpoint = ModelCheckpoint('unet_liverCT_fulliamge4.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit(X_train, y_train, batch_size=1, epochs=30, validation_split=0.01, callbacks=[ModelCheckpoint])
model.model.save("model/model_final_time.h5")


# h = model.fit(
#     np.expand_dims(X_train,axis=-1),
#     np.expand_dims(y_train,axis=-1),
#     batch_size=4,
#     epochs=50,
#     validation_split=0.02,
#     callbacks=callbacks)
# hi = model.fit_generator(myGene, steps_per_epoch=900, epochs=8, callbacks=[model_checkpoint])

# plt.plot(hi.history['acc'][1:])
# plt.plot(hi.history['loss'][1:],color="r", linestyle="--", marker="*")
# plt.title('model')
# plt.ylabel('loss acc')
# plt.xlabel('epoch')
# plt.legend(['train','Validation'], loc='upper left')
# plt.savefig("re.jpg")

