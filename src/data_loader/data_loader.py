import glob
import os
import copy

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import nibabel as nib
from random import shuffle
import SimpleITK as sitk


from data_loader.patch_sampler import patch_generator
from data_loader.patch_sampler import masked_2D_sampler
from data_loader.preprocessing import (
    minmax_normalization, windowing, smoothing)
from data_loader.create_boxdata import finecut_to_thickcut


class DataGenerator():
    '''
    Data generator for external dataset
    Args:
        patch_size (int): patch size
        stride (int): distance of moving window
        data_type (str): 'NTUH' or 'MSD' or 'Pancreas-CT'
    '''

    def __init__(self, patch_size=50, stride=5, data_type='MSD'):
        self.patch_size = patch_size
        self.stride = stride
        self.data_type = data_type
        if data_type == 'tcia':
            self.data_path = '/data2/open_dataset/pancreas/Pancreas-CT/'
        elif data_type == 'msd':
            self.data_path = '/data2/open_dataset/MSD/Task07_Pancreas/'
        elif data_type == 'ntuh':
            self.data_path = '/data2/pancreas/Nifti_data/'

    def load_image(self, filename):
        self.filename = filename
        if self.data_type == 'tcia':
            file_location = glob.glob(os.path.join(
                self.data_path, filename) + '/*/*/000001.dcm')
            imagedir = os.path.dirname(file_location[0])
            labelname = 'label' + filename[-4:] + '.nii.gz'
            labelpath = os.path.join(self.data_path, 'annotations', labelname)

            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(imagedir)
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
            image_array = sitk.GetArrayFromImage(image).transpose((2, 1, 0))
            self.thickness = image.GetSpacing()[2]
            label = nib.load(labelpath).get_data()

        elif self.data_type == 'msd':
            imagepath = os.path.join(self.data_path, 'imagesTr', filename)
            labelpath = os.path.join(self.data_path, 'labelsTr', filename)

            img = nib.load(imagepath)
            image_array = img.get_data()
            label = nib.load(labelpath).get_data()
            self.thickness = img.affine[2, 2]

        elif self.data_type == 'ntuh':
            imagepath = os.path.join(
                self.data_path, 'image', 'IM_' + filename + '.nii.gz')
            labelpath = os.path.join(
                self.data_path, 'label', 'LB_' + filename + '.nii.gz')
            backup_path = '/data2/pancreas/results/image_AD/'
            filename_s = ''.join([char for char in filename][:2]) + str(int(''.join([char for char in filename][2:])))
            rothpath = os.path.join(backup_path, 'IM_'
                                    + filename_s + '/IM_' + filename_s + '_model.nii.gz')
            img = nib.load(imagepath)
            image_array = img.get_data()
            if os.path.exists(labelpath):
                label = nib.load(labelpath).get_data()
            elif os.path.exists(rothpath):
                result = nib.load(rothpath).get_data()
                label = np.zeros(result.shape)
                label[np.where(result==8)] = 1
            else:
                print(labelpath)
            self.thickness = img.affine[2, 2]

        return image_array, label

    def load_box(self, filename):
        self.filename = filename
        if self.data_type == 'tcia':
            file_dir = '/data2/pancreas/box_data/wanyun/tcia'
            file_path = os.path.join(file_dir, filename)
        elif self.data_type == 'msd':
            file_dir = '/data2/pancreas/box_data/wanyun/msd'
            file_path = os.path.join(file_dir, filename)
        elif self.data_type == 'ntuh':
            file_dir = '/data2/pancreas/box_data/wanyun/ntuh'
            file_path = os.path.join(file_dir, filename)
        image = np.load(os.path.join(file_path, 'ctscan.npy'))
        pancreas = np.load(os.path.join(file_path, 'pancreas.npy'))
        lesion = np.load(os.path.join(file_path, 'lesion.npy'))
        return image, pancreas, lesion

    def get_boxdata(self, image, label, border=(10, 10, 3)):

        # Transfer label
        pancreas = np.zeros(label.shape)
        pancreas[np.where(label != 0)] = 1
        lesion = np.zeros(label.shape)
        if self.data_type == 'msd' or self.data_type == 'ntuh':
            lesion[np.where(label == 2)] = 1

        # Finecut to thickcut
        if self.thickness < 5:
            image = finecut_to_thickcut(image, self.thickness)
            pancreas = finecut_to_thickcut(
                pancreas, self.thickness, label_mode=True)
            lesion = finecut_to_thickcut(
                lesion, self.thickness, label_mode=True)

        # Generate box index
        xmin, ymin, zmin = np.max(
            [np.min(np.where(pancreas != 0), 1) - border, (0, 0, 0)], 0)
        xmax, ymax, zmax = np.min(
            [np.max(np.where(pancreas != 0), 1) + border, label.shape], 0)

        # Generate box data
        box_image = image[xmin:xmax, ymin:ymax, zmin:zmax]
        box_pancreas = pancreas[xmin:xmax, ymin:ymax, zmin:zmax]
        box_lesion = lesion[xmin:xmax, ymin:ymax, zmin:zmax]

        return box_image, box_pancreas, box_lesion

    def preprocessing(self, image, pancreas, lesion):
        from skimage import morphology

        if self.data_type == 'tcia':
            image = image[:, ::-1, :]
            pancreas = pancreas[:, ::-1, :]
            lesion = lesion[:, ::-1, :]
            pancreas = smoothing(pancreas)
            lesion = smoothing(lesion)
        elif self.data_type == 'msd':
            image = image[::-1, ::-1, :]
            pancreas = pancreas[::-1, ::-1, :]
            lesion = lesion[::-1, ::-1, :]
        elif self.data_type == 'ntuh':
            image = image[::-1, ::-1, :]
            pancreas = pancreas[::-1, ::-1, :]
            lesion = lesion[::-1, ::-1, :]

        pancreas = morphology.dilation(pancreas, np.ones([3, 3, 1]))
        lesion = morphology.dilation(lesion, np.ones([3, 3, 1]))

        image = windowing(image)
        image = minmax_normalization(image)

        return image, pancreas, lesion

    def generate_patch(self, image, pancreas, lesion):
        X = []
        Y = []

        self.coords = masked_2D_sampler(lesion, pancreas,
                                        self.patch_size, self.stride, threshold=1 / (self.patch_size ** 2))

        image[np.where(pancreas == 0)] = 0

        for coord in self.coords:
            mask_pancreas = image[coord[1]
                                  :coord[4], coord[2]:coord[5], coord[3]]
            X.append(mask_pancreas)
            Y.append(coord[0])

        self.X = X
        self.Y = Y

        return X, Y

    def patch_len(self):
        return len(self.X)

    def gt_pancreas_num(self):
        return len(self.Y) - np.sum(self.Y)

    def gt_lesion_num(self):
        return np.sum(self.Y)

    def get_prediction(self, model, patch_threshold=0.5):
        from sklearn.metrics import confusion_matrix
        if len(self.X) > 0:
            test_X = np.array(self.X)
            test_X = test_X.reshape(
                (test_X.shape[0], test_X.shape[1], test_X.shape[2], 1))
            test_Y = np.array(self.Y)
        else:
            return filename

        self.probs = model.predict_proba(test_X)
        predict_y = predict_binary(self.probs, patch_threshold)
        self.patch_matrix = confusion_matrix(test_Y, predict_y, labels=[1, 0])

        return self.patch_matrix

    def get_auc(self):
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, thresholds = roc_curve(self.Y, self.probs)
        return auc(fpr, tpr)

    def get_probs(self):
        return self.probs, self.Y

    def get_roc_curve(self):
        from data_description.visualization import plot_roc
        plot_roc(self.probs, self.Y)

    def get_all_value(self):
        tp = self.patch_matrix[0][0]
        fn = self.patch_matrix[0][1]
        fp = self.patch_matrix[1][0]
        tn = self.patch_matrix[1][1]
        return tp, fn, fp, tn


def get_patches(config, case_partition, mode='train'):
    X = []
    y = []
    idx = []
    patch_size = config['dataset']['input_dim'][0]
    stride = config['dataset']['stride']
    load_way = config['dataset']['load']

    for partition in case_partition:
        print("GET_PATCHES:\tLoading {} data ... ".format(partition['type']))
        datagen = DataGenerator(
            patch_size, stride=stride, data_type=partition['type'])
        for case_id in tqdm(partition[mode]):
            if load_way == 'ori':
                img, lbl = datagen.load_image(case_id)
                box_img, box_pan, box_les = datagen.get_boxdata(
                    img, lbl)
                image, pancreas, lesion = datagen.preprocessing(
                    box_img, box_pan, box_les)
            elif load_way == 'box':
                image, pancreas, lesion = datagen.load_box(case_id)
            tmp_X, tmp_y = datagen.generate_patch(
                image, pancreas, lesion)
            X = X + tmp_X
            y = y + tmp_y
            idx.append([partition['type'], case_id, len(tmp_y)])

    return X, y, idx


def save_boxdata(config, case_partition, mode='train'):
    patch_size = config['dataset']['input_dim'][0]
    stride = config['dataset']['stride']
    save_list = [[], [], []]
    index = 0

    for partition in case_partition:
        print("GET_PATCHES:\tLoading {} data ... ".format(partition['type']))
        file_path = '/data2/pancreas/box_data/wanyun/' + partition['type'] + '/'
        save_list_av = []
        datagen = DataGenerator(
            patch_size, stride=stride, data_type=partition['type'])
        for case_id in tqdm(partition[mode]):
            img, lbl = datagen.load_image(case_id)
            if len(lbl)>0 and not os.path.exists(file_path + case_id):
                save_list_av.append(case_id)
                box_img, box_pan, box_les = datagen.get_boxdata(
                    img, lbl)
                image, pancreas, lesion = datagen.preprocessing(
                    box_img, box_pan, box_les)

                os.mkdir(file_path + case_id)
                np.save(os.path.join(file_path + case_id, 'ctscan.npy'), image)
                np.save(os.path.join(file_path + case_id, 'pancreas.npy'), pancreas)
                np.save(os.path.join(file_path + case_id, 'lesion.npy'), lesion)

        save_list[index] = save_list_av
        print(len(save_list_av))
        index += 1
    return save_list
    

            


                
