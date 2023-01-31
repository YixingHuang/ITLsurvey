import os
import cv2
import numpy as np
classes = ['n01641577', 'n02124075', 'n02802426', 'n03970156', 'n04067472', 'n04099969', 'n04540053', 'n07749582', 'n07920052', 'n09246464']
root_path = '../data/datasets/tiny-imagenet/tiny-imagenet-200'
subsets = ['train', 'val']
suffix = '_noisy'
mean = 0
sigma = 25
for subfolder in subsets:
    main_path = os.path.join(root_path, subfolder)
    for class_ in classes:
        class_folder = os.path.join(main_path, class_, 'images')
        noisy_class_folder = os.path.join(main_path, class_+suffix + "_" + str(sigma), 'images')
        os.makedirs(noisy_class_folder, exist_ok=True)
        allfiles = os.listdir(class_folder)
        valid_files = [f for f in allfiles if os.path.isfile(os.path.join(class_folder, f))]
        for f in valid_files:
            img_name = os.path.join(class_folder, f)
            img = cv2.imread(img_name)
            noisy_image = np.zeros(img.shape, np.float32)
            if len(img.shape) == 2:
                gaussian = np.random.normal(mean, sigma, (64, 64))
                noisy_image = img + gaussian
            else:
                for channel in range(3):
                    gaussian = np.random.normal(mean, sigma, (64, 64))
                    noisy_image[:, :, channel] = img[:, :, channel] + gaussian
            cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
            noisy_image = noisy_image.astype(np.uint8)
            cv2.imwrite(os.path.join(noisy_class_folder, f), noisy_image)



