import os
def main():
    subsets = ['train', 'val']
    img_paths = {t: {s: [] for s in subsets + ['classes', 'class_to_idx']} for t in range(1, 5 + 1)}
    print(img_paths)

    classes = ('absent', 'mild', 'moderate', 'severe', 'proliferative retinopathy')
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    print(class_to_idx)

    center_path = 'C:/MachineLearning/CLsurveyMCofficial/src/data/datasets/retina/retina_preprocessed/1957_2196'
    # allfiles = os.listdir(center_path)
    # print(len(allfiles))
    # print(allfiles)
    # allfiles2 = [f for f in allfiles]
    # print(allfiles2)

    print(isIncluded(2, 5)[0])


def isIncluded(original_class, num_class):
    if num_class == 2 and original_class == 1:
        return False, original_class
    elif num_class == 2:
        new_class = 0 if original_class < 1 else 1
        return True, new_class
    else:
        return True, original_class

if __name__ == "__main__":
    main()
