from matplotlib import pyplot
import shapely.wkt
import pandas as pd
import skimage.transform
import numpy as np


def image_mask_resized_from_summary(df):
    # image_id = "AOI_3_Paris_img197"
    im_mask = np.zeros((650, 650))    
    # logger.info(df)
    # if len(df[df.ImageId == image_id]) == 0:
    #     raise RuntimeError("ImageId not found on summaryData: {}".format(
    #         image_id))

    for idx, row in df.iterrows():
        shape_obj = shapely.wkt.loads(row.PolygonWKT_Pix)
        if shape_obj.exterior is not None:
            coords = list(shape_obj.exterior.coords)
            x = [round(float(pp[0])) for pp in coords]
            y = [round(float(pp[1])) for pp in coords]
            yy, xx = skimage.draw.polygon(y, x, (650, 650))
            im_mask[yy, xx] = 1

            interiors = shape_obj.interiors
            for interior in interiors:
                coords = list(interior.coords)
                x = [round(float(pp[0])) for pp in coords]
                y = [round(float(pp[1])) for pp in coords]
                yy, xx = skimage.draw.polygon(y, x, (650, 650))
                im_mask[yy, xx] = 0
    # im_mask = skimage.transform.resize(im_mask, (256, INPU))
    # im_mask = (im_mask > 0.5).astype(np.uint8)  
    print(im_mask.shape)
    pyplot.imshow(im_mask)
    pyplot.show()
    # logger.info(np.count_nonzero(im_mask))
    return im_mask

df = pd.read_csv("/BuildingDetectors_Round2/1-XD_XD/code/v9s.csv")
image_mask_resized_from_summary(df)