from matplotlib import pyplot as plt
import shapely.wkt
import pandas as pd
import skimage.transform
import numpy as np
import cv2
# import gdal

"/data/test/AOI_3_Paris_Test/MUL/MUL_AOI_3_Paris_img522.tif/"
"/data/test/AOI_3_Paris_Test/MUL-PanSharpen/MUL-PanSharpen_AOI_3_Paris_img522.tif/"
"/data/test/AOI_3_Paris_Test/PAN/PAN_AOI_3_Paris_img522.tif"


def image_mask_resized_from_summary(df):    
    img = cv2.imread('./RGB-PanSharpen_AOI_3_Paris_img522.tif',3)
    plt.imshow(img)
    plt.show()
    # ds = gdal.open('./RGB-PanSharpen_AOI_3_Paris_img522.tif')
    # data = ds.ReadAsArray(ds)
    # img = data.swapaxes(0,1).swapaxes(1,2)
    # img2 = img.copy()    
    # im_mask = np.zeros(img2.shape[0:2],dtype=np.uint8)
    # logger.info(df)
    # if len(df[df.ImageId == image_id]) == 0:
    #     raise RuntimeError("ImageId not found on summaryData: {}".format(
    #         image_id))

    # for idx, row in df.iterrows():
    #     shape_obj = shapely.wkt.loads(row.PolygonWKT_Pix)
        # print shape_obj

    #     if shape_obj.exterior is not None:
    #         coords = list(shape_obj.exterior.coords)
    #         x = [round(float(pp[0])) for pp in coords]
    #         y = [round(float(pp[1])) for pp in coords]
    #         yy, xx = skimage.draw.polygon(y, x, (650, 650))
    #         im_mask[yy, xx] = 1

    #         interiors = shape_obj.interiors
    #         for interior in interiors:
    #             coords = list(interior.coords)
    #             x = [round(float(pp[0])) for pp in coords]
    #             y = [round(float(pp[1])) for pp in coords]
    #             yy, xx = skimage.draw.polygon(y, x, (650, 650))
    #             im_mask[yy, xx] = 0
    # # im_mask = skimage.transform.resize(im_mask, (256, INPU))
    # # im_mask = (im_mask > 0.5).astype(np.uint8)  
    # print(im_mask.shape)
    # pyplot.imshow(im_mask)
    # pyplot.show()
    # logger.info(np.count_nonzero(im_mask))
    return 0

df = pd.read_csv("./v9s.csv")
image_mask_resized_from_summary(df)


model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


from keras.models import model_from_json 

def reStoreModel():
    
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    return loaded_model
