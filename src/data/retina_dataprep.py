"""
Download the diabetic retinopathy detection dataset manually from Kaggle:
https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data
Please use retinal_dataprep_selection to preprocess the data: resampling and crop into 256 * 256 images,
and put them into different folders according to their image sizes.
Different image sizes indicate different camera devices and hence can be assumed from different centers.
"""

import os
import torch
import shutil
import subprocess
import csv
import random
from torchvision import transforms

import utilities.utils as utils
from data.imgfolder import random_split, ImageFolderTrainVal


def download_dset(path):
    utils.create_dir(path)

    if not os.path.exists(os.path.join(path, 'processed')):

        print("Please manually download the data and preprocess the dataset.")
    else:
        print("Already selected retina dataset in {}".format(os.path.join(path, 'retina_preprocessed')))


def create_training_classes_file(root_path):
    """
    training dir is ImageFolder like structure.
    Gather all classnames in 1 file for later use.
    Ordering may differ from original classes.txt in project!
    :return:
    """
    with open(os.path.join(root_path, 'classes.txt'), 'w') as classes_file:
        for class_dir in utils.get_immediate_subdirectories(os.path.join(root_path, 'train')):
            classes_file.write(class_dir + "\n")


def preprocess_val(root_path):
    """
    Uses val_annotations.txt to construct ImageFolder like structure.
    Images in 'image' folder are moved into class-folder.
    :return:
    """
    val_path = os.path.join(root_path, 'val')
    annotation_path = os.path.join(val_path, 'val_annotations.txt')

    lines = [line.rstrip('\n') for line in open(annotation_path)]
    for line in lines:
        subs = line.split('\t')
        imagename = subs[0]
        dirname = subs[1]
        this_class_dir = os.path.join(val_path, dirname, 'images')
        if not os.path.isdir(this_class_dir):
            os.makedirs(this_class_dir)

        # utils.attempt_move(os.path.join(val_path, 'images', imagename), this_class_dir)
        utils.attempt_move(os.path.join(val_path, 'images', imagename), this_class_dir)



# def divide_into_centers(root_path, center_count=10, num_classes=2, min_num=50, max_num=200):
#     """
#     Divides total subset data into multi-centers (into dirs "task_x").
#     center_count: number of centers
#     num_class: either 2 or 5 for retinal fundus images
#     min_num: the minimum number of images at each center
#     max_num: the maximum number of images used at each center (some centers have thounsands of images)
#     :return:
#     """
#     print("Be patient: dividing into research centers...")
#
#     if num_classes == 5:
#         classes = ('absent', 'mild', 'moderate', 'severe', 'proliferative retinopathy')
#     elif num_classes == 2:
#         classes = ('absent', 'DR')
#     else:
#         raise NotImplementedError('num_class should be either 2 or 5!')
#     class_to_idx = {classes[i]: i for i in range(len(classes))}
#
#     patient2class = getPatient2Class(os.path.join(root_path, 'trainLabels.csv'))
#     subsets = ['train', 'val']
#
#     img_paths = {t: {s: [] for s in subsets + ['classes', 'class_to_idx']} for t in range(1, center_count + 1)}
#     folders = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]
#     total_folders = len(folders)
#     print('total number of folders/centers is ', total_folders)
#     assert center_count <= total_folders, "center_count should be smaller than {}".format(total_folders)
#
#     folder_id = 0
#
#     for center_id in range(1, center_count + 1):
#
#         if len(img_paths[center_id]['classes']) == 0:
#             img_paths[center_id]['classes'].extend(classes)
#         img_paths[center_id]['class_to_idx'] = class_to_idx
#
#         num_files = 0
#         allfiles = []
#         while num_files < 2 * min_num:
#             center_path = os.path.join(root_path, folders[folder_id])
#             num_per_class, total_diseased = getFileNumbers(center_path)
#             if total_diseased < min_num:
#                 folder_id = folder_id + 1
#                 continue
#
#             healthy_files = []
#             diseased_files = []
#             for category in range(0, 5):
#                 subfolder = os.path.join(center_path, str(category))
#                 if not os.path.exists(subfolder):
#                     continue
#                 allfiles_sub = os.listdir(subfolder)
#                 allfiles_sub = [os.path.join(subfolder, f) for f in allfiles_sub if os.path.isfile(os.path.join(subfolder, f))]
#                 if category == 0:
#                     healthy_files.extend(allfiles_sub)
#                 else:
#                     diseased_files.extend(allfiles_sub)
#             num_class_0 = num_per_class[0] if num_per_class[0] < total_diseased else total_diseased
#             num_class_0 = max_num if num_class_0 > max_num else num_class_0
#             total_diseased = max_num if total_diseased > max_num else total_diseased
#             random.seed(0)
#             healthy_files = random.sample(healthy_files, int(num_class_0/2.0))
#             diseased_files = random.sample(diseased_files, total_diseased)
#             allfiles = healthy_files + diseased_files
#             num_files = len(allfiles)
#             allfiles = random.sample(allfiles, num_files)
#             folder_id = folder_id + 1
#         print('Folder ' + folders[folder_id] + ' is selected!')
#         num_train = int(num_files * 0.8)
#         for subset in subsets:
#             if subset == 'train':
#                 initial_image_id = 0
#                 num_imgs = num_train
#             else:
#                 initial_image_id = num_train
#                 num_imgs = num_files - num_train
#             imgs = []
#             for f in allfiles[initial_image_id: initial_image_id + num_imgs]:
#                 patientName = os.path.basename(f)[0:-5]
#                 original_class = patient2class.get(patientName)
#                 isinclude, new_class = isIncluded(original_class=original_class, num_class=num_classes)
#                 if isinclude:
#                    imgs.append((f, new_class))
#
#             img_paths[center_id][subset].extend(imgs)
#
#     return img_paths

def divide_into_centers(root_path, center_count=10, num_classes=2, min_num=50, max_num=200):
    """
    Divides total subset data into multi-centers (into dirs "task_x").
    center_count: number of centers
    num_class: either 2 or 5 for retinal fundus images
    min_num: the minimum number of images at each center
    max_num: the maximum number of images used at each center (some centers have thounsands of images)
    :return:
    """
    print("Be patient: dividing into research centers...")

    classes = ('absent', 'DR')
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    subsets = ['train', 'val']

    img_paths = {t: {s: [] for s in subsets + ['classes', 'class_to_idx']} for t in range(1, center_count + 1)}
    folders = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]
    total_folders = len(folders)
    print('total number of folders/centers is ', total_folders)
    assert center_count <= total_folders, "center_count should be smaller than {}".format(total_folders)

    for center_id in range(1, center_count + 1):

        if len(img_paths[center_id]['classes']) == 0:
            img_paths[center_id]['classes'].extend(classes)
        img_paths[center_id]['class_to_idx'] = class_to_idx

        folder_id = center_id - 1
        center_path = os.path.join(root_path, folders[folder_id])

        for category in range(0, 2):
            subfolder = os.path.join(center_path, str(category))
            if not os.path.exists(subfolder):
                continue
            allfiles_sub = os.listdir(subfolder)
            allfiles = [os.path.join(subfolder, f) for f in allfiles_sub if os.path.isfile(os.path.join(subfolder, f))]

            num_files = len(allfiles)
            num_train = int(num_files * 0.8)

            for subset in subsets:
                if subset == 'train':
                    initial_image_id = 0
                    num_imgs = num_train
                else:
                    initial_image_id = num_train
                    num_imgs = num_files - num_train
                    if num_imgs % 2 == 1:
                        num_imgs = num_imgs - 1
                imgs = []
                for f in allfiles[initial_image_id: initial_image_id + num_imgs]:
                    imgs.append((f, category))

                img_paths[center_id][subset].extend(imgs)

    return img_paths


def getFileNumbers(mainPath):
    num_per_class = [0, 0, 0, 0, 0] #initialization number
    for category in range(0, 5):
        subfolder = os.path.join(mainPath, str(category))
        if not os.path.exists(subfolder):
            continue
        allfiles = os.listdir(subfolder)
        allfiles = [f for f in allfiles if os.path.isfile(os.path.join(subfolder, f))]
        num_per_class[category] = len(allfiles)
    total_diseased = sum(num_per_class[1:])
    return num_per_class, total_diseased

def isIncluded(original_class, num_class):
    if num_class == 2:
        if original_class == 1:
            return False, original_class
        else:
            new_class = 0 if original_class < 1 else 1
        return True, new_class
    else:
        return True, original_class


def getPatient2Class(csv_path):
    csv_file = open(csv_path, 'r')
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    patient2class={}
    for row in csv_reader:
        line_count = line_count + 1 # only left eyes
        if line_count % 2 == 1:
            continue
        patient_name = row[0]
        disease_type = int(row[1])
        patient2class.update({patient_name:disease_type})
    return patient2class

def create_train_test_val_imagefolders(img_paths, root, normalize, include_rnd_transform, no_crop):
    # TRAIN
    pre_transf = None
    if include_rnd_transform:
        if no_crop:
            pre_transf = transforms.RandomHorizontalFlip()
        else:
            pre_transf = transforms.Compose([
                transforms.RandomResizedCrop(56),  # Crop
                transforms.RandomHorizontalFlip(), ])
    else:  # No rnd transform
        if not no_crop:
            pre_transf = transforms.Compose([
                transforms.Resize(64),
                transforms.CenterCrop(56),  # Crop
            ])
    sufx_transf = [transforms.ToTensor(), normalize, ]
    train_transf = transforms.Compose([pre_transf] + sufx_transf) if pre_transf else transforms.Compose(sufx_transf)
    train_dataset = ImageFolderTrainVal(root, None, transform=train_transf, classes=img_paths['classes'],
                                        class_to_idx=img_paths['class_to_idx'], imgs=img_paths['train'])

    # Validation
    pre_transf_val = None
    sufx_transf_val = [transforms.ToTensor(), normalize, ]
    if not no_crop:
        pre_transf_val = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(56), ])
    val_transf = transforms.Compose([pre_transf_val] + sufx_transf_val) if pre_transf_val \
        else transforms.Compose(sufx_transf_val)
    test_dataset = ImageFolderTrainVal(root, None, transform=val_transf, classes=img_paths['classes'],
                                       class_to_idx=img_paths['class_to_idx'], imgs=img_paths['val'])

    # Validation set of TinyImgnet is used for testing dataset,
    # Training data set is split into train and validation.
    dsets = {}
    dsets['train'] = train_dataset
    dsets['test'] = test_dataset

    # Split original TinyImgnet trainset into our train and val sets
    dset_trainval = random_split(dsets['train'],
                                 [round(len(dsets['train']) * (0.8)), round(len(dsets['train']) * (0.2))])
    dsets['train'] = dset_trainval[0]
    dsets['val'] = dset_trainval[1]
    dsets['val'].transform = val_transf  # Same transform val/test
    print("Created Dataset:{}".format(dsets))
    return dsets


def create_train_val_test_imagefolder_dict(dataset_root, img_paths, task_count, outfile, no_crop=True, transform=False):
    """
    Makes specific wrapper dictionary with the 3 ImageFolder objects we will use for training, validation and evaluation.
    """
    # Data loading code
    if no_crop:
        out_dir = os.path.join(dataset_root, "no_crop", "{}tasks".format(task_count))
    else:
        out_dir = os.path.join(dataset_root, "{}tasks".format(task_count))

    for task in range(1, task_count + 1):
        print("\nTASK ", task)

        # Tiny Imgnet total values from pytorch
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        dsets = create_train_test_val_imagefolders(img_paths[task], dataset_root, normalize, transform, no_crop)
        utils.create_dir(os.path.join(out_dir, str(task)))
        torch.save(dsets, os.path.join(out_dir, str(task), outfile))
        print("SIZES: train={}, val={}, test={}".format(len(dsets['train']), len(dsets['val']),
                                                        len(dsets['test'])))
        print("Saved dictionary format of train/val/test dataset Imagefolders.")


def create_train_val_test_imagefolder_dict_joint(dataset_root, img_paths, outfile, no_crop=True):
    """
    For JOINT training: All 10 tasks in 1 data folder.
    Makes specific wrapper dictionary with the 3 ImageFolder objects we will use for training, validation and evaluation.
    """
    # Data loading code
    if no_crop:
        out_dir = os.path.join(dataset_root, "no_crop")
    else:
        out_dir = dataset_root

    # Tiny Imgnet total values from pytorch
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dsets = create_train_test_val_imagefolders(img_paths[1], dataset_root, normalize, True, no_crop=no_crop)

    ################ SAVE ##################
    utils.create_dir(out_dir)
    torch.save(dsets, os.path.join(out_dir, outfile))
    print("JOINT SIZES: train={}, val={}, test={}".format(len(dsets['train']), len(dsets['val']),
                                                          len(dsets['test'])))
    print("JOINT: Saved dictionary format of train/val/test dataset Imagefolders.")


def prepare_dataset(dset, target_path, survey_order=True, joint=True, task_count=5, overwrite=False,
                    num_class=10):
    """
    Main datapreparation code for Tiny Imagenet.
    First download the set and set target_path to unzipped download path.
    See README dataprep.

    :param target_path: Path to Tiny Imagenet dataset
    :param survey_order: Use the original survey ordering of the labels to divide in tasks
    :param joint: Prepare the joint dataset
    """
    print("Preparing dataset")
    if not os.path.isdir(target_path):
        raise Exception("RETINA PATH IS NON EXISTING DIR: ", target_path)

    # Make different subset dataset for each task/center
    if not os.path.isfile(os.path.join(target_path, "DIV.TOKEN")) or overwrite:
        print("PREPARING DATASET: DIVIDING INTO {} TASKS".format(task_count))

        img_paths = divide_into_centers(target_path, center_count=task_count, num_classes=num_class, min_num=300, max_num=2000)

        torch.save({}, os.path.join(target_path, 'DIV.TOKEN'))
    else:
        print("Already divided into tasks")

    if not os.path.isfile(os.path.join(target_path, "IMGFOLDER.TOKEN")) or overwrite:
        print("PREPARING DATASET: IMAGEFOLDER GENERATION")
        create_train_val_test_imagefolder_dict(target_path, img_paths, task_count, dset.raw_dataset_file,
                                               no_crop=True, transform=False)
        create_train_val_test_imagefolder_dict(target_path, img_paths, task_count, dset.transformed_dataset_file,
                                               no_crop=True, transform=True)
        torch.save({}, os.path.join(target_path, 'IMGFOLDER.TOKEN'))
    else:
        print("Task imgfolders already present.")

    if joint:
        if not os.path.isfile(os.path.join(target_path, "IMGFOLDER_JOINT.TOKEN")) or overwrite:
            print("PREPARING JOINT DATASET: IMAGEFOLDER GENERATION")
            # img_paths = divide_into_centers(target_path, center_count=task_count, num_classes=num_class, min_num=300,
            #                                 max_num=2000, isJoint=True)
            img_paths = utils.merge_individual_centers(img_paths, center_count=task_count)
            # Create joint
            create_train_val_test_imagefolder_dict_joint(target_path, img_paths, dset.joint_dataset_file, no_crop=True)
            torch.save({}, os.path.join(target_path, 'IMGFOLDER_JOINT.TOKEN'))
        else:
            print("Joint imgfolders already present.")

    print("PREPARED DATASET")
