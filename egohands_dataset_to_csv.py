import scipy.io as sio
import numpy as np
import os
import gc
import six.moves.urllib as urllib
import cv2
import time
import xml.etree.cElementTree as ET
import random
import shutil
from shutil import copyfile
import zipfile

import csv

"""
ORIGINAL SOURCE: https://github.com/molyswu/hand_detection/blob/temp/hand_detection/egohands_dataset_clean.py

I have cleaned up slightly, allowed for python3 functionality and some other things too:
   > using os.path.join instead of string1 + "/" + string2
   > ignores bounding boxes in original dataset where xmin = xmax or ymin=ymax
   > allows to choose a subset of the 40 subfolders of images.
"""

image_filetypes = ['jpg', 'jpeg', 'png']

def save_csv(csv_path, csv_content):
    with open(csv_path, 'w', newline='') as csvfile:
        wr = csv.writer(csvfile)
        for i in range(len(csv_content)):
            wr.writerow(csv_content[i])


def get_bbox_visualize(base_path, dir):
    image_path_array = []
    for root, dirs, filenames in os.walk(os.path.join(base_path, dir)):
        for f in filenames:
            if f.split(".")[-1] in image_filetypes:
                img_path = os.path.join(base_path, dir, f)
                image_path_array.append(img_path)

    #sort image_path_array to ensure its in the low to high order expected in polygon.mat
    image_path_array.sort()
    boxes = sio.loadmat(
        os.path.join(base_path, dir, "polygons.mat")
    )
    # there are 100 of these per folder in the egohands dataset
    polygons = boxes["polygons"][0]
    pointindex = 0

    for first in polygons:
        index = 0

        font = cv2.FONT_HERSHEY_SIMPLEX

        img_id = image_path_array[pointindex]
        img = cv2.imread(img_id)

        pointindex += 1

        csvholder = []
        for pointlist in first:
            pst = np.empty((0, 2), int)
            max_x = max_y = min_x = min_y = height = width = 0

            findex = 0
            for point in pointlist:
                if(len(point) == 2):
                    x = int(point[0])
                    y = int(point[1])

                    if(findex == 0):
                        min_x = x
                        min_y = y
                    findex += 1
                    max_x = x if (x > max_x) else max_x
                    min_x = x if (x < min_x) else min_x
                    max_y = y if (y > max_y) else max_y
                    min_y = y if (y < min_y) else min_y
                    appeno = np.array([[x, y]])
                    pst = np.append(pst, appeno, axis=0)

            if (min_x > 0 and min_y > 0 and max_x > 0 and max_y > 0):
                if (min_x != max_x and min_y != max_y):
                    labelrow = [tail,
                                np.size(img, 1), np.size(img, 0), "hand", min_x, min_y, max_x, max_y]
                    csvholder.append(labelrow)

        csv_path = img_id.split(".")[0]
        if os.path.exists(csv_path + ".csv"):
            os.remove(csv_path+".csv")
        save_csv(csv_path + ".csv", csvholder)


def create_directory(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

# combine all individual csv files for each image into a single csv file per folder.


def generate_label_files(image_dir):
    header = ['filename', 'width', 'height',
              'class', 'xmin', 'ymin', 'xmax', 'ymax']
    for root, dirs, filenames in os.walk(image_dir):
        for dir in dirs:
            csvholder = []
            csvholder.append(header)
            loop_index = 0
            files_in_dir = os.listdir(os.path.join(image_dir, dir))
            for f in files_in_dir:
                if(f.split(".")[-1] == "csv"):
                    loop_index += 1
                    csv_file = open(os.path.join(image_dir, dir, f), 'r')
                    reader = csv.reader(csv_file)
                    row_idx = 0
                    for row in reader:
                        try:
                            row[0] = os.path.join(image_dir, dir, row[0])
                        except IndexError:
                            print(row)
                        csvholder.append(row)
                        row_idx += 1
                    csv_file.close()
                    os.remove(os.path.join(image_dir, dir, f))
            save_csv(os.path.join(image_dir, dir, dir + "_labels.csv"), csvholder)
            print("Saved label csv for ", dir, os.path.join(image_dir, dir, dir + "_labels.csv"))


# Split data, copy to train/test folders
def split_data_test_eval_train(source_image_dir, image_dir):
    create_directory(image_dir)
    train_dir = os.path.join(image_dir,'train')
    test_dir = os.path.join(image_dir,'test')
    create_directory(train_dir)
    create_directory(test_dir)

    data_size = 4000
    loop_index = 0
    data_sampsize = int(0.1 * data_size)
    test_samp_array = random.sample(range(data_size), k=data_sampsize)

    for root, dirs, filenames in os.walk(source_image_dir):
        for dir in dirs:
            for f in os.listdir(os.path.join(source_image_dir, dir)):
                if f.split(".")[-1] in image_filetypes:
                    loop_index += 1

                    if loop_index in test_samp_array:
                        os.rename(os.path.join(source_image_dir, dir, f),
                                  os.path.join(test_dir, f))
                        os.rename(os.path.join(source_image_dir, dir, f.split(".")[0] + ".csv"),
                                  os.path.join(test_dir, f.split(".")[0] + ".csv"))
                    else:
                        os.rename(os.path.join(source_image_dir, dir, f),
                                  os.path.join(train_dir, f))
                        os.rename(os.path.join(source_image_dir, dir, f.split(".")[0] + ".csv"),
                                  os.path.join(train_dir, f.split(".")[0] + ".csv"))
            print(">   done scanning directory ", dir)
            os.remove(os.path.join(source_image_dir, dir, "polygons.mat"))
            shutil.rmtree(os.path.join(source_image_dir, dir))

        print("Train/test content generation complete!")


def generate_csv_files(source_image_dir):
    for root, dirs, filenames in os.walk(source_image_dir):
        for dir in dirs:
            get_bbox_visualize(source_image_dir, dir)

    print("CSV generation for each file complete!\nGenerating train/test folders")


# rename image files so we can have them all in a train/test folder.
def rename_files(source_image_dir):
    print("Renaming files")
    loop_index = 0
    for root, dirs, filenames in os.walk(source_image_dir):
        for dir in dirs:
            for f in os.listdir(os.path.join(source_image_dir, dir)):
                if dir not in f:
                    if f.split(".")[-1] in image_filetypes:
                        loop_index += 1
                        os.rename(
                            os.path.join(source_image_dir, dir, f),
                            os.path.join(source_image_dir, dir, dir + "_" + f)
                        )
                else:
                    break

def extract_folder(dataset_path, source_dir, num_directories=4):
    if not os.path.exists("egohands"):
        print("Extracting files")
        zip_ref = zipfile.ZipFile(dataset_path, 'r')
        print("> Extracting Dataset files")
        zip_ref.extractall("egohands")
        print("> Extraction complete")
        zip_ref.close()
        dirs = [os.path.join(source_dir, dir) for dir in os.listdir(source_dir)]
        random.shuffle(dirs)
        for d in dirs[:-num_to_keep]:
            shutil.rmtree(d)
        return True
    else:
        print("Files already extracted.")
        return False

    

def download_egohands_dataset(dataset_url, dataset_path):
    is_downloaded = os.path.exists(dataset_path)
    if not is_downloaded:
        print(
            "> downloading egohands dataset. This may take a while (1.3GB, say 3-5mins). Coffee break?")
        # opener = urllib.request.URLopener()
        # opener.retrieve(dataset_url, dataset_path)
        os.system(f"wget {EGOHANDS_DATASET_URL}")
        print("> download complete")
    else:
        print("Egohands dataset already downloaded.")


if __name__=="__main__":

    EGOHANDS_DATASET_URL = "http://vision.soic.indiana.edu/egohands_files/egohands_data.zip"
    EGO_HANDS_FILE = "egohands_data.zip"
    source_dir = os.path.join('egohands', '_LABELLED_SAMPLES')
    NUM_DIRECTORIES_TO_KEEP = 4
    images_dir = 'images'

    download_egohands_dataset(EGOHANDS_DATASET_URL, EGO_HANDS_FILE)
    extracted = extract_folder(EGO_HANDS_FILE, source_dir, num_directories=NUM_DIRECTORIES_TO_KEEP)
    if extracted:
        rename_files(source_dir)
        
    generate_csv_files(source_dir)
    split_data_test_eval_train(source_dir, images_dir)
    generate_label_files(images_dir)