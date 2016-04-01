import os
import h5py
import numpy
from PIL import Image
from fuel.datasets.hdf5 import H5PYDataset
import glob
IMAGENET_FOLDER = '/data/lisa/data/ImageNet_ILSVRC2010/pylearn2_h5/'
IMAGENET_TRAIN  = 'imagenet_2010_train.h5'
IMAGENET_TEST   = 'imagenet_2010_test.h5'
IMAGENET_VALID  = 'imagenet_2010_valid.h5'

CELEBA_FACE_FOLDER = '/data/lisatmp4/taesup/data/face/CelebA/img_align_celeba/'

def make_imagenet_dataset(original_hdf5_path,
                          fuel_hdf5_path,
                          dataset_type,
                          resize_shape):

    original_file = h5py.File(name=original_hdf5_path,
                              mode='r')
    fuel_file     = h5py.File(name=fuel_hdf5_path,
                              mode='w')

    # get original shape
    original_shape = original_file['x'].shape
    num_images     = original_shape[0]
    num_channels   = original_shape[1]

    # set new dataset for fuel file
    image_data = fuel_file.create_dataset(name='image_data',
                                          shape=(num_images, num_channels) + resize_shape,
                                          dtype='uint8')

    for idx, original_image in enumerate(original_file['x']):
        original_image = Image.fromarray(numpy.transpose(original_image, (1, 2, 0)))
        original_image.thumbnail(resize_shape, Image.ANTIALIAS)

        original_image = numpy.asarray(original_image)
        image_data[idx] = numpy.transpose(original_image, (2, 0, 1))
    image_data.dims[0].label = 'batch'
    image_data.dims[1].label = 'channel'
    image_data.dims[2].label = 'height'
    image_data.dims[3].label = 'width'

    split_dict = { dataset_type : {'image_data': (0, num_images)}}
    fuel_file .attrs['split'] = H5PYDataset.create_split_array(split_dict)

    fuel_file.flush()
    fuel_file.close()

    original_file.close()

    print 'DONE : {} (num of images :{})'.format(fuel_hdf5_path, num_images)



def make_celeb_dataset(fuel_hdf5_path,
                       resize_shape):
    # get image list
    image_list = glob.glob(CELEBA_FACE_FOLDER + '*.jpg')
    num_images = len(image_list)

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

    output_folder = '/data/lisatmp4/taesup/data/face/CelebA/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    resize_shape = (64, 64)
    print 'START TRAINING SET'
    make_celeb_dataset(fuel_hdf5_path=output_folder+'CelebFace_64x64.hdf5',
                       resize_shape=resize_shape)

    # output_folder = '/data/lisatmp4/taesup/data/imagenet64x64/'
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)
    #
    # resize_shape = (64, 64)
    # print 'START TRAINING SET'
    # make_imagenet_dataset(original_hdf5_path=IMAGENET_FOLDER+IMAGENET_TRAIN,
    #                       fuel_hdf5_path=output_folder+'imagenet64x64_train.hdf5',
    #                       dataset_type='train',
    #                       resize_shape=resize_shape)
    #
    # print 'START VALID SET'
    # make_imagenet_dataset(original_hdf5_path=IMAGENET_FOLDER+IMAGENET_VALID,
    #                       fuel_hdf5_path=output_folder+'imagenet64x64_valid.hdf5',
    #                       dataset_type='valid',
    #                       resize_shape=resize_shape)
