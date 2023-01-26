"""
Download the diabetic retinopathy detection dataset manually from Kaggle:
https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data

Please use this file to preprocess the data: resampling and crop into 256 * 256 images,
and put them into different folders according to their image sizes.
Different image sizes indicate different camera devices and hence can be assumed from different centers.

Please put all the images in the "train" folder before running this script.
The processed images will be saved in the "./datasets/retina/processed/" folder with subdirectories named after image sizes.

"""


import csv
import cv2
import os

data = 2
if data == 0: # training folder:
    csv_file = open('C:/Data/RetinalData2/Training_Set/Training_Set/RFMiD_Training_Labels.csv', 'r')
    input_folder = 'C:/Data/RetinalData2/Training_Set/Training_Set/Training/' #modify this path according to your source data location
    save_folder = 'C:/Data/RetinalData2/Training_Set/Training_Set/preprocessedMultiple/'
elif data == 1: #Test folder
    csv_file = open('C:/Data/RetinalData2/Test_Set/Test_Set/RFMiD_Testing_Labels.csv', 'r')
    input_folder = 'C:/Data/RetinalData2/Test_Set/Test_Set/Test/'  # modify this path according to your source data location
    save_folder = 'C:/Data/RetinalData2/Test_Set/Test_Set/preprocessedMultiple/'
elif data == 2: # evaluation/validation folder
    csv_file = open('C:/Data/RetinalData2/Evaluation_Set/Evaluation_Set/RFMiD_Validation_Labels.csv', 'r')
    input_folder = 'C:/Data/RetinalData2/Evaluation_Set/Evaluation_Set/Validation/'  # modify this path according to your source data location
    save_folder = 'C:/Data/RetinalData2/Evaluation_Set/Evaluation_Set/preprocessedMultiple/'
else:
    print("data should be 0, 1 ,2")

suffix = '.png'
csv_reader = csv.reader(csv_file, delimiter=',')
line_count = 0
new_size = 256.0
new_disease_names = ['healthy', 'DR', 'others']
for row in csv_reader:
    line_count = line_count + 1
    if line_count == 1:
        disease_names = row
        print(disease_names)
        continue

    patient_name = row[0]
    is_disease = int(row[1])
    disease_type = 0
    disease_name = 'healthy'
    if is_disease == 1:
        location = row[2:].index('1')
        disease_type = location + 1
        disease_name = disease_names[location + 2]

    img_name = input_folder + patient_name + suffix
    img = cv2.imread(img_name)

    szs = img.shape
    camera_size_name = str(szs[0]) + "_" + str(szs[1])
    directory = save_folder + camera_size_name + '/' + str(disease_type) + "_" + disease_name + '/'
    save_name = directory + str(int(patient_name) + data * 10000) + suffix
    if not os.path.exists(directory):
        os.makedirs(directory)
    rx = new_size/szs[0]
    ry = new_size/szs[1]
    if ry > rx: #Zoom in accordint to the smaller size
        rx = ry
    img_size2 = (int(szs[1] * rx), int(szs[0] * rx)) #note that numpy array and images swap height and width.

    img2 = cv2.resize(img, dsize=img_size2, interpolation=cv2.INTER_AREA)
    if img_size2[0] > new_size:
        half = int((img_size2[0] - new_size)/2)
        img2 = img2[:, half: half + int(new_size)]
    else:
        half = int((img_size2[1] - new_size) / 2)
        img2 = img2[half: half + int(new_size), :]
    cv2.imwrite(save_name, img2)
    if line_count < 20: #print out some exemplary cases for check
        print(patient_name, is_disease, szs, camera_size_name, disease_type)

print('Process done!')