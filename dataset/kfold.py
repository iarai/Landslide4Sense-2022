import os
import shutil
import random
import glob2
import numpy as np


def kfold_split(num_fold=10, test_image_number=380):
    print("Splitting for k-fold with %d fold" % num_fold)
    data_root = os.path.join(os.getcwd(), 'data')

    dir_names = []
    for fold in range(num_fold):
        dir_names.append('data/TrainData' + str(fold))

    for dir_name in dir_names:
        print("Creating fold " + dir_name)
        os.makedirs(dir_name)

        # making subdirectory train and test
        os.makedirs(os.path.join(os.getcwd(), dir_name, 'test'))
        os.makedirs(os.path.join(os.getcwd(), dir_name, 'train'))

        # locating to the test and train directory
        test_dir = os.path.join(os.getcwd(), dir_name, 'test')
        train_dir = os.path.join(os.getcwd(), dir_name, 'train')

        # making image and mask sub-dirs
        os.makedirs(os.path.join(test_dir, 'img'))
        os.makedirs(os.path.join(test_dir, 'mask'))
        os.makedirs(os.path.join(train_dir, 'img'))
        os.makedirs(os.path.join(train_dir, 'mask'))

        # read the image and mask directory
        image_files = os.listdir(os.path.join(os.getcwd(), 'data/img'))
        mask_files = os.listdir(os.path.join(os.getcwd(), 'data/mask'))

        # creating random file names for testing
        test_filenames = random.sample(image_files, test_image_number)

        for filename in test_filenames:
            img_data_root = os.path.join(data_root, 'img')
            msk_data_root = os.path.join(data_root, 'mask')

            img_dest = os.path.join(os.getcwd(), dir_name, 'test', 'img')
            msk_dest = os.path.join(os.getcwd(), dir_name, 'test', 'mask')

            img_file_path = os.path.join(img_data_root, filename)
            msk_file_path = os.path.join(msk_data_root, filename.replace('image', 'mask'))

            shutil.copy(img_file_path, img_dest)
            shutil.copy(msk_file_path, msk_dest)

        # saving files for training
        for other_filename in image_files:
            if other_filename in test_filenames:
                continue
            else:
                img_data_root = os.path.join(data_root, 'img')
                msk_data_root = os.path.join(data_root, 'mask')

                img_dest = os.path.join(os.getcwd(), dir_name, 'train', 'img')
                msk_dest = os.path.join(os.getcwd(), dir_name, 'train', 'mask')

                img_file_path = os.path.join(img_data_root, other_filename)
                msk_file_path = os.path.join(msk_data_root, other_filename.replace('image', 'mask'))

                shutil.copy(img_file_path, img_dest)
                shutil.copy(msk_file_path, msk_dest)


def get_train_list(fold):
    all_files = []
    for ext in ["*.h5"]:
        images = glob2.glob(os.path.join("data/TrainData" + str(fold) + "/train/img/", ext))
        all_files += images

    all_train_files = []
    for idx in np.arange(len(all_files)):
        image = str(all_files[idx]).split("/")
        image = os.path.join("TrainData" + str(fold) + "/train/img/", str(image[4]))
        all_train_files.append(image)

    # Create train.txt
    with open("dataset/train.txt", "w") as f:
        for idx in np.arange(len(all_train_files)):
            f.write(all_train_files[idx] + '\n')


def get_test_list(fold):
    all_files = []
    for ext in ["*.h5"]:
        images = glob2.glob(os.path.join("data/TrainData" + str(fold) + "/test/img/", ext))
        all_files += images

    all_test_files = []
    for idx in np.arange(len(all_files)):
        image = str(all_files[idx]).split("/")
        image = os.path.join("TrainData" + str(fold) + "/test/img/", str(image[4]))
        all_test_files.append(image)

    # Create Test.txt
    with open("dataset/test.txt", "w") as f:
        for idx in np.arange(len(all_test_files)):
            f.write(all_test_files[idx] + '\n')


def get_train_test_list(fold):
    get_train_list(fold)
    get_test_list(fold)
