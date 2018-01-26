import pandas as pd
import tqdm
import rasterio
import numpy as np
from logging import getLogger, Formatter, StreamHandler, INFO
import warnings
import scipy
import skimage.transform
import shapely.wkt
import tables as tb
from matplotlib import pyplot


INPUT_SIZE = 256
# Logger
warnings.simplefilter("ignore", UserWarning)
handler = StreamHandler()
handler.setLevel(INFO)
handler.setFormatter(Formatter('%(asctime)s %(levelname)s %(message)s'))

logger = getLogger('spacenet2')
logger.setLevel(INFO)


if __name__ == '__main__':
    logger.addHandler(handler)

#doc id tu csv
def _load_train_summary_data():    
    df = pd.read_csv("./AOI_3_Paris_Train_Building_Solutions.csv")
    # df.loc[:, 'ImageId'] = df.ImageId.str[4:]
    df_agg = df.groupby('ImageId').agg('first')
    image_id_list = df_agg.index.tolist()
    pd.DataFrame({'ImageId': image_id_list[:10]}).to_csv(
        "197.csv".format(prefix="zz"),
        index=False)
    return df
    # print image_id_list
    
# _load_train_summary_data()

def get_train_image_path_from_imageid(image_id, mul=False):    
    if mul:
        return "./img197/MUL-PanSharpen_AOI_3_Paris_img197.tif"
    else:
        return "./img197/RGB-PanSharpen_AOI_3_Paris_img197.tif"


###################################################
#feature tu anh 3 band
###################################################
def __calc_rgb_multiband_cut_threshold():    
    band_values = {k: [] for k in range(3)}
    band_cut_th = {k: dict(max=0, min=0) for k in range(3)}

    image_id_list = pd.read_csv("197.csv").ImageId.tolist()
    # for image_id in tqdm.tqdm(image_id_list[:500]):
    image_fn="./img197/RGB-PanSharpen_AOI_3_Paris_img197.tif"
    with rasterio.open(image_fn, 'r') as f:
        values = f.read().astype(np.float32)
        # logger.info(values)
        for i_chan in range(3):
            values_ = values[i_chan].ravel().tolist()
            # logger.info(values_)
            values_ = np.array(
                [v for v in values_ if v != 0]
            )  # Remove sensored mask
            band_values[i_chan].append(values_)

    # image_id_list = pd.read_csv(FMT_VALTEST_IMAGELIST_PATH.format(
    #     prefix=prefix)).ImageId.tolist()
    # for image_id in tqdm.tqdm(image_id_list[:500]):
    image_fn="./img197/RGB-PanSharpen_AOI_3_Paris_img197.tif"
    with rasterio.open(image_fn, 'r') as f:        
        # pyplot.imshow(f.read(1), cmap='pink')
        # pyplot.show()
        values = f.read().astype(np.float32)        
        for i_chan in range(3):
            values_ = values[i_chan].ravel().tolist()
            values_ = np.array(
                [v for v in values_ if v != 0]
            )  # Remove sensored mask
            band_values[i_chan].append(values_)

    logger.info("Calc percentile point ...")
    for i_chan in range(3):
        band_values[i_chan] = np.concatenate(
            band_values[i_chan]).ravel()
        band_cut_th[i_chan]['max'] = scipy.percentile(
            band_values[i_chan], 98)
        band_cut_th[i_chan]['min'] = scipy.percentile(
            band_values[i_chan], 2)
    logger.info("band_cut_th")
    # logger.info(band_cut_th)
    return band_cut_th


def calc_rgb_multiband_cut_threshold():
    rows = []
    band_cut_th = __calc_rgb_multiband_cut_threshold()
    prefix = "197"
    row = dict(prefix=prefix)
    row['id'] = "197"
    for chan_i in band_cut_th.keys():
        row['chan{}_max'.format(chan_i)] = band_cut_th[chan_i]['max']
        row['chan{}_min'.format(chan_i)] = band_cut_th[chan_i]['min']
    rows.append(row)
    pd.DataFrame(rows).to_csv(
        "197_RGB_BAND_CUT.csv", index=False)




###################################################
#feature tu anh 8 band
###################################################
def __calc_mul_multiband_cut_threshold():
    prefix = 197
    band_values = {k: [] for k in range(8)}
    band_cut_th = {k: dict(max=0, min=0) for k in range(8)}
    # for image_id in tqdm.tqdm(image_id_list[:500]):
    image_fn = "./img197/MUL-PanSharpen_AOI_3_Paris_img197.tif"
    with rasterio.open(image_fn, 'r') as f:
        values = f.read().astype(np.float32)
        for i_chan in range(8):
            values_ = values[i_chan].ravel().tolist()
            values_ = np.array(
                [v for v in values_ if v != 0]
            )  # Remove sensored mask
            band_values[i_chan].append(values_)

    # for image_id in tqdm.tqdm(image_id_list[:500]):
    image_fn = "./img197/MUL-PanSharpen_AOI_3_Paris_img197.tif"
    with rasterio.open(image_fn, 'r') as f:
        values = f.read().astype(np.float32)
        for i_chan in range(8):
            values_ = values[i_chan].ravel().tolist()
            values_ = np.array(
                [v for v in values_ if v != 0]
            )  # Remove sensored mask
            band_values[i_chan].append(values_)

    logger.info("Calc percentile point ...")
    for i_chan in range(8):
        band_values[i_chan] = np.concatenate(
            band_values[i_chan]).ravel()
        band_cut_th[i_chan]['max'] = scipy.percentile(
            band_values[i_chan], 98)
        band_cut_th[i_chan]['min'] = scipy.percentile(
            band_values[i_chan], 2)
    return band_cut_th


def calc_mul_multiband_cut_threshold():
    rows = []
    band_cut_th = __calc_mul_multiband_cut_threshold()
    prefix = "197"
    row = dict(prefix=prefix)
    row['id'] = "197"
    for chan_i in band_cut_th.keys():
        row['chan{}_max'.format(chan_i)] = band_cut_th[chan_i]['max']
        row['chan{}_min'.format(chan_i)] = band_cut_th[chan_i]['min']
    rows.append(row)
    pd.DataFrame(rows).to_csv("197_MUL_BAND_CUT.csv",index=False)


###################################################
#tao mask
###################################################

def image_mask_resized_from_summary(df):
    image_id = "AOI_3_Paris_img197"
    im_mask = np.zeros((650, 650))    
    # logger.info(df)
    # if len(df[df.ImageId == image_id]) == 0:
    #     raise RuntimeError("ImageId not found on summaryData: {}".format(
    #         image_id))

    for idx, row in df[df.ImageId == image_id].iterrows():
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
    im_mask = skimage.transform.resize(im_mask, (INPUT_SIZE, INPUT_SIZE))
    im_mask = (im_mask > 0.5).astype(np.uint8)  
    print(im_mask.shape)
    pyplot.imshow(im_mask)
    pyplot.show()
    # logger.info(np.count_nonzero(im_mask))
    return im_mask


def prep_image_mask(is_valtrain=True):
    prefix = "197"
    logger.info("prep_image_mask for {}".format(prefix))
    if is_valtrain:
        fn_list = "197_valtrain.h5"
        fn_mask = "197_valtrain_msk.h5"
    else:
        fn_list = "197_valtest.h5"
        fn_mask = "197_valtest_msk.h5"

    df = pd.read_csv("AOI_3_Paris_Train_Building_Solutions.csv")
    logger.info("Prepare image container")
    with tb.open_file(fn_mask, 'w') as f:
        # for image_id in tqdm.tqdm(df.index, total=len(df)):
        im_mask = image_mask_resized_from_summary(df)
        atom = tb.Atom.from_dtype(im_mask.dtype)
        filters = tb.Filters(complib='blosc', complevel=9)
        ds = f.create_carray(f.root, "AOI_3_Paris_img197", atom, im_mask.shape,
                                filters=filters)
        ds[:] = im_mask

##############################################        
#
##############################################        
def view_csv():
    df = pd.read_csv("/BuildingDetectors_Round2/1-XD_XD/code/v9s.csv")
    image_mask_resized_from_summary(df)
    


##############################################        
#run everything
##############################################        
calc_rgb_multiband_cut_threshold()
calc_mul_multiband_cut_threshold()
prep_image_mask()


