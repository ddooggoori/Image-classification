import os
import glob
import json
import time
from tqdm import tqdm
from PIL import ImageFile, Image, ImageOps
import pandas as pd
from pathlib import Path
import shutil
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

def Resize(files, save_path):
    """
    Resize images to 256x256 pixels and save them to a new directory.
    """
    
    for f in tqdm(files):
        time.sleep(0.1)
        img = Image.open(f)
        img = ImageOps.exif_transpose(img)  # Handle EXIF orientation
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")  # Convert to RGB if image mode is RGBA or P
        img_resize = img.resize((256, 256))  # Resize image to 256x256 pixels
        base_filename = os.path.basename(f)
        title, ext = os.path.splitext(base_filename)
        final_filepath = os.path.join(save_path, title + '_resized' + ext)
        img_resize.save(final_filepath)  # Save resized image


def Check_files(image_files, json_files):
    """
    Adjust coordinates in JSON files based on original image size.
    """

    image = Image.open(image_files)
    imag_size = image.size

    with open(json_files, 'r') as a:
        json_data = json.load(a)

    # Adjust coordinates based on original image size
    json_data['coordinates']['x1'] = json_data['coordinates']['x1'] * imag_size[1]
    json_data['coordinates']['x2'] = json_data['coordinates']['x2'] * imag_size[1]
    json_data['coordinates']['y1'] = json_data['coordinates']['y1'] * imag_size[0]
    json_data['coordinates']['y2'] = json_data['coordinates']['y2'] * imag_size[0]

    return json_data  # Return updated JSON data


def Coordinate(image_files, json_files):
    """
    Adjust coordinates in JSON files based on resized image size and save to CSV.
    """

    filename = list()
    x1 = list()
    x2 = list()
    y1 = list()
    y2 = list()

    for f, i in tqdm(zip(image_files, json_files)):
        time.sleep(0.1)
        image = Image.open(f)
        imag_size = image.size

        with open(i, 'r') as a:
            json_data = json.load(a)

        # Adjust coordinates based on resized image size (256x256)
        new_x1 = (json_data['coordinates']['x1'] * (256/imag_size[1]))
        new_x2 = (json_data['coordinates']['x2'] * (256/imag_size[1]))
        new_y1 = (json_data['coordinates']['y1'] * (256/imag_size[0]))
        new_y2 = (json_data['coordinates']['y2'] * (256/imag_size[0]))

        base_filename = os.path.basename(f)
        title, ext = os.path.splitext(base_filename)

        filename.append(title)
        x1.append(new_x1)
        x2.append(new_x2)
        y1.append(new_y1)
        y2.append(new_y2)

    # Create pandas DataFrame and save to CSV
    filename = pd.DataFrame(filename, columns=['filename'])
    x1 = pd.DataFrame(x1, columns=['x1'])
    x2 = pd.DataFrame(x2, columns=['x2'])
    y1 = pd.DataFrame(y1, columns=['y1'])
    y2 = pd.DataFrame(y2, columns=['y2'])

    coordinate = pd.concat([filename, x1, y1, x2, y2], axis=1)
    coordinate.to_csv(r"BP1_json_resized.csv", index=False)


def get_files_count(folder_path):
    """
    Get the number of files in a specified folder.
    """
    dirListing = os.listdir(folder_path)
    return len(dirListing)


def Coordinate2(image_files, json_files):
    """
    Adjust coordinates in JSON files based on resized image size and save as new JSON files.
    """

    for f, i in tqdm(zip(image_files, json_files)):
        time.sleep(0.1)
        image = Image.open(f)
        imag_size = image.size

        with open(i, 'r') as a:
            json_data = json.load(a)

        # Adjust coordinates based on resized image size (256x256)
        json_data['coordinates']['x1'] = json_data['coordinates']['x1'] * imag_size[1]
        json_data['coordinates']['x2'] = json_data['coordinates']['x2'] * imag_size[1]
        json_data['coordinates']['y1'] = json_data['coordinates']['y1'] * imag_size[0]
        json_data['coordinates']['y2'] = json_data['coordinates']['y2'] * imag_size[0]

        base_filename = os.path.basename(i)
        title, ext = os.path.splitext(base_filename)

        # Change working directory to save the new JSON files
        os.chdir('BP2_json')
        path = os.getcwd()

        # Save the adjusted JSON data to new files
        with open(title + '_original.json', 'w') as outfile:
            json.dump(json_data, outfile)


def random_extracting(train_size, test_size, val_size):
    """
    Randomly select files from each category (Glucose, Sphygmo, Thermo, Weight) for training, testing, and validation sets.
    """
    for ttv in ['train', 'test', 'val']:        

            if ttv == 'train':                
                def random_extracting(size):
                    for i in ['Glucose', 'Sphygmo', 'Thermo', 'Weight']:
                        dirpath = r"resized/" + str(i)
                        destDirectory = r"testing/" + str(ttv) + '/' + str(i)

                        Path(destDirectory).mkdir(parents=True, exist_ok=True)

                        filenames = random.sample(os.listdir(dirpath), size)
                        for fname in filenames:
                            srcpath = os.path.join(dirpath, fname)
                            shutil.copy(srcpath, destDirectory)

                        data = os.path.abspath(destDirectory)

                        for a, f in enumerate(os.listdir(data)):
                            src = os.path.join(data, f)
                            dst = os.path.join(data, str(i) +(str(a + 1))+'.jpg')
                            os.rename(src, dst)
                    
                random_extracting(train_size)
            
            elif ttv == 'test':
                random_extracting(test_size)

            elif ttv == 'val':
                random_extracting(val_size)
