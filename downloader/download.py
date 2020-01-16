import urllib.request
import os
import argparse
import errno
import random
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from time import time as timer

argparser = argparse.ArgumentParser(description='Download specific objects from Open-Images dataset')
argparser.add_argument('-a', '--annots',
                       help='path to annotations file (.csv)')
argparser.add_argument('-o', '--objects', nargs='+',
                       help='download images of these objects')
argparser.add_argument('-d', '--dir',
                       help='path to output directory')
argparser.add_argument('-l', '--labelmap',
                       help='path to labelmap (.csv)')
argparser.add_argument('-i', '--images',
                       help='path to file containing links to images (.csv)')
argparser.add_argument('-m', '--max',
                       help='maximum number of images to download')
argparser.add_argument('-s', '--notstrict',
                       help='not strict getting objects')
argparser.add_argument('-c', '--notoccluded',
                       help='get not occluded objects')

args = argparser.parse_args()

# parse arguments
ANNOTATIONS = args.annots
OUTPUT_DIR = args.dir
OBJECTS = args.objects
LABELMAP = args.labelmap
IMAGES = args.images
LIMIT = int(args.max)
NOT_STRICT = args.notstrict
NOT_OCCLUDED = args.notoccluded

# make OUTPUT_DIR if not present
if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print("\nCreated {} directory\n".format(OUTPUT_DIR))

# check if input files are valid, raise FileNotFoundError if not found
if not os.path.exists(ANNOTATIONS):
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), ANNOTATIONS)
elif not os.path.exists(LABELMAP):
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), LABELMAP)


def get_ooi_labelmap(labelmap):
    '''
    Given labelmap of all objects in Open Images dataset, get labelmap of objects of interest

    :param labelmap: dataframe containing object labels with respective label codes
    :return: dictionary containing object labels and codes of
                          user-inputted objects
    '''

    # If want not strict class filtering (if 'bicycle' also get 'bicycle wheel')
    if NOT_STRICT is not None:

        object_codes = {}
        for idx, row in labelmap.iterrows():
            if any(obj.lower() in row[1].lower().split(' ') for obj in OBJECTS):
                object_codes[row[1].lower()] = row[0]
    else:
        labelmap['name'] = labelmap['name'].apply(str.lower)
        object_codes = labelmap.set_index('name').loc[OBJECTS].to_dict()['code']

    return object_codes


def generate_download_list(annotations, labelmap, base_url):
    '''
    Parse through input annotations dataframe, find ImageID's of objects of interest,
    and get download urls for the corresponding images

    :param annotations: annotations dataframe
    :param labelmap: dictionary of object labels and codes
    :param base_url: basename of url
    :return: list of urls to download
    '''
    
    label_names = labelmap.values()

    # find ImageID's in original annots dataframe corresponding to ooi's codes
    if NOT_OCCLUDED:
        annotations = annotations[annotations['IsOccluded'] == 0]    
    df_download = annotations[annotations['LabelName'].isin(label_names)]['ImageID'].unique()

    ######################

    url_download_list = []
    existing_images = os.listdir(OUTPUT_DIR)

    for image_id in df_download:
        # get name of the image
        image_name = image_id + ".jpg"

        # check if the image exists in directory
        if image_name not in existing_images:
            # form url
            url = os.path.join(base_url, image_name)
            url_download_list.append(url)   
        
    return url_download_list


def fetch_url(url):
    try:
        urllib.request.urlretrieve(url, os.path.join(OUTPUT_DIR, url.split("/")[-1]))
        return 0
    except Exception as e:
        return 1


def download_objects_of_interest(download_list):
    with Pool(4) as pool:
        failures = sum(tqdm(pool.imap_unordered(fetch_url, download_list), total=len(download_list), desc="Download: "))

    return failures


def main():
    # read images and get base_url
    df_images = pd.read_csv(IMAGES)
    base_url = os.path.dirname(df_images['image_url'][0])  # used to download the images

    # read labelmap
    df_oid_labelmap = pd.read_csv(LABELMAP, header = None, names=['code','name'])  # open images dataset (oid) labelmap
    ooi_labelmap = get_ooi_labelmap(df_oid_labelmap)  # objects of interest (ooi) labelmap

    # read annotations
    df_annotations = pd.read_csv(ANNOTATIONS)

    print("\nGenerating download list for the following objects: ", [k for k, v in ooi_labelmap.items()])

    # get url list to download
    download_list = generate_download_list(annotations=df_annotations,
                                           labelmap=ooi_labelmap,
                                           base_url=base_url)
    
    # get selected number of images
    if LIMIT < len(download_list):
        download_list = random.sample(download_list, LIMIT)

    # download objects of interest
    failures = download_objects_of_interest(download_list)

    print(f"\nFinished downloads. Couldn't load {failures} images")


if __name__ == '__main__':
    main()
