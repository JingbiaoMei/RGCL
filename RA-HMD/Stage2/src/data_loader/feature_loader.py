import base64
import time
import sys
import csv
from tqdm import tqdm
import numpy as np
csv.field_size_limit(sys.maxsize)
import json


def get_attrobj_from_ids(fname, ids=0, topk=None, att=True, length=0):
    """Load object features (text) from tsv file.
    :param fname: The path to the tsv file.
    :param topk: Only load features for top K images (lines) in the tsv file.
        Will load all the features if topk is either -1 or None.
    :param att: If True, load attributes, else only load object names.
    :return: A list of image object features where each feature is a dict.
        See FIELDNAMES above for the keys in the feature dit.
    """
    #print(list(ids))
    # Loading the json file that contains the mappings from integer encoded labels to object names
    dictionary = json.load(open('./data/features/dictionary/VG-SGG-dicts-vgoi6-clipped.json', 'r'))
    idx2label = dictionary['idx_to_label']
    idx2attr = dictionary['idx_to_attribute']
    # For idx 0, the attribute is empty
    idx2attr["0"] = ''
    
    
    FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
    # initialize the empty data dictionary
    data = {}
    start_time = time.time()
    print("Start to load Faster-RCNN detected objects from %s" % fname)
    with open(fname) as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
        #boxes = num_features  # Same boxes for all
        #print(len(reader))
        for i, item in tqdm(enumerate(reader), total=length):
            #print(i, item["img_id"], int(item["img_id"]))
            # Check if id in list of ids to save memory
            # list map list is just a dirty fix 
            '''
            if int(item["img_id"]) not in list(map(int, list(ids))):
                continue
            '''
            item.pop('boxes')
            item.pop('features')
            # Make sure these values of the keys are int
            for key in ['img_h', 'img_w', 'num_boxes']:
                item[key] = int(item[key])


            boxes = item['num_boxes']
            decode_config = [
                ('objects_id', (boxes,), np.int64),
                ('objects_conf', (boxes,), np.float32),
                ('attrs_id', (boxes,), np.int64),
                ('attrs_conf', (boxes,), np.float32),
                #('boxes', (boxes, 4), np.float32),
                #('features', (boxes, -1), np.float32),
            ]
            for key, shape, dtype in decode_config:
                item[key] = np.frombuffer(base64.b64decode(item[key]), dtype=dtype)
                try:
                    #print(key)
                    #print(item[key].shape)
                    #print(item[key])

                    item[key] = item[key].reshape(shape)
                except:
                    # In 1 out of 10K cases, the shape comes out wrong; We make necessary adjustments
                    shape = list(shape)
                    shape[0] += 1
                    shape = tuple(shape)
                    item[key] = item[key].reshape(shape)

                item[key].setflags(write=False)
            
            # map object id to object name
            item["object_names"] = [idx2label[str(objid)] for objid in item["objects_id"]]
            # map attribute id to attribute name
            #print(item["attrs_id"])
            item["attribute_names"] = [idx2attr[str(attrid)] for attrid in item["attrs_id"]]
            
            data[item["img_id"]] = item
            if topk is not None and len(data) == topk:
                break
    elapsed_time = time.time() - start_time
    print("Loaded %d images in file %s in %d seconds." % (len(data), fname, elapsed_time))
    return data