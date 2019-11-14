from model import *
from data import *
from keras.models import Model, load_model
# import cv2
# from imageEhance import *
# path2 = "test2/"
# def readImage(file):
#     testGene = testGenerator2(file)
#     ori = cv2.imread(file)
#     model = load_model("U-Net_1.hdf5")
#     results = model.predict_generator(testGene, 1, verbose=1)
#     saveResult2(path2,results)
#     img_mask,ori_boder=simgelImage(path2+"1.jpg",ori)
#
#     return img_mask,ori_boder
#
# def img_ori(file):
#     ori = cv2.imread(file)
#     return ori
# def saveResult2(save_path,npyfile,flag_multi_class = False,num_class = 2):
#     for i, item in enumerate(npyfile):
#         # img = labelVisualize(num_class, COLOR_DICT, item) if flag_multi_class else item[:, :, 0]
#         print(os.path.join("%d.jpg" % i))
#         io.imsave(os.path.join(save_path,"1.jpg"),item)
#         return item
# # readImage("test2/image/IM1703.png")
print ("testing...........")
path="data/data2/data_white_test/"
testGene = testGenerator(path)
# model = unet("U-Net_1.hdf5")
model = load_model("model/new_data_white_all_30_final.hdf5")
results = model.predict_generator(testGene,47, verbose=1)
saveResult("data/data2/data_white_result_unet_20/", results)


