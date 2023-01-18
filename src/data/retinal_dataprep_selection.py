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


csv_file = open('C:/Data/RetinalData/trainLabels.csv', 'r')
csv_reader = csv.reader(csv_file, delimiter=',')
line_count = 0
input_folder = 'C:/Data/RetinalData/train/' #modify this path according to your source data location
suffix = '.jpeg'
save_folder = './datasets/retina/retina_preprocessed/'
new_size = 256.0
for row in csv_reader:
    line_count = line_count + 1
    if line_count % 2 == 1:
        continue
    patient_name = row[0]
    disease_type = int(row[1])
    if not (disease_type == 1):
        continue

    img_name = input_folder + patient_name + suffix
    img = cv2.imread(img_name)

    szs = img.shape
    camera_size_name = str(szs[0]) + "_" + str(szs[1])
    directory = save_folder + camera_size_name + '/'
    save_name = directory + patient_name + suffix
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
        print(patient_name, disease_type, szs, camera_size_name)

print('Process done!')