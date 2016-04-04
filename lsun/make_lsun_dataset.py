import os
import h5py
import numpy
from PIL import Image
from fuel.datasets.hdf5 import H5PYDataset
import glob
import fnmatch
LSUN_PATH = '/data/lisatmp4/taesup/data/lsun/'
BEDROOM_TRAIN_PATH = LSUN_PATH + 'bedroom_train/'

def make_lsun_dataset(scene_path,
                      fuel_hdf5_path,
                      resize_shape):

    # get image list
    image_list = []
    for root, dirs, files in os.walk(scene_path):
        for filename in fnmatch.filter(files, '*.jpg'):
            image_list.append(os.path.join(root, filename))
    num_images = len(image_list)

    print 'num of images :{}'.format(num_images)

    # open image file
    fuel_file     = h5py.File(name=fuel_hdf5_path,
                              mode='w')

    # set new dataset for fuel file
    image_data = fuel_file.create_dataset(name='image_data',
                                          shape=(num_images, 3) + resize_shape,
                                          dtype='uint8')

    for idx, filepath in enumerate(image_list):
        original_image = Image.open(filepath).convert('RGB')
        resize_row = resize_shape[0] if original_image.size[0]<original_image.size[1] else original_image.size[0]
        resize_col = resize_shape[1] if original_image.size[0]>original_image.size[1] else original_image.size[1]
        original_image.thumbnail((resize_row, resize_col), Image.ANTIALIAS)

        if original_image.size[0] != resize_shape[0]:
            excess = (original_image.size[0] - resize_shape[0]) / 2
            original_image = original_image.crop((excess, 0, resize_shape[0]+excess, resize_shape[0]))
        elif original_image.size[1] != resize_shape[1]:
            excess = (original_image.size[1] - resize_shape[1]) / 2
            original_image = original_image.crop((0, excess, resize_shape[1], resize_shape[1]+excess))

        original_image = numpy.asarray(original_image)
        image_data[idx] = numpy.transpose(original_image, (2, 0, 1))

    image_data.dims[0].label = 'batch'
    image_data.dims[1].label = 'channel'
    image_data.dims[2].label = 'height'
    image_data.dims[3].label = 'width'

    split_dict = { 'train' : {'image_data': (0, num_images)}}
    fuel_file .attrs['split'] = H5PYDataset.create_split_array(split_dict)

    fuel_file.flush()
    fuel_file.close()

    print 'DONE : {} (num of images :{})'.format(fuel_hdf5_path, num_images)

if __name__=="__main__":

    output_folder = LSUN_PATH
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    resize_shape = (64, 64)
    print 'START LSUN BEDROOM TRAINING SET'
    make_lsun_dataset(scene_path=BEDROOM_TRAIN_PATH,
                      fuel_hdf5_path=output_folder+'lsun_bedroom_train_64x64.hdf5',
                      resize_shape=resize_shape)
