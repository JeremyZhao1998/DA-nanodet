import os
import random
import json
from tqdm import tqdm


categories_dict = {
    "pedestrian": 1,
    "rider": 1,
    "person": 1,
    "sign": 2,
    "vehicle": 3,
    "arrow": 4,
    # "road_block": 5,
    # "light": 6
}


categories = [
    {"id": 1, "name": "pedestrian"},
    {"id": 2, "name": "sign"},
    {"id": 3, "name": "vehicle"},
    {"id": 4, "name": "arrow"},
    # {"id": 5, "name": "road_block"},
    # {"id": 6, "name": "light"}
]


def set_seed(seed):
    random.seed(seed)


def convert_weather(img_path_list, anno_file_path):
    anno_file = {"images": [], "categories": categories, "annotations": []}
    type_cnt = {cid: 0 for cid in categories_dict.values()}
    for img_id, img_path in tqdm(enumerate(img_path_list), total=len(img_path_list)):
        path_split = img_path.split('/')
        basename = os.path.join(path_split[-3], path_split[-2], path_split[-1])
        with open(img_path.replace('jpg', 'json'), 'r', encoding='utf-8') as fp:
            img_anno = json.load(fp)
        anno_file['images'].append({
            "id": img_id,
            "width": img_anno['publicAttrs']['fileWidth'],
            "height": img_anno['publicAttrs']['fileHeight'],
            "file_name": basename
        })
        for box_anno in img_anno['anno']:
            if box_anno['category']['type'] not in categories_dict:
                continue
            anno_file['annotations'].append({
                "id": len(anno_file['annotations']),
                "image_id": img_id,
                "category_id": categories_dict[box_anno['category']['type']],
                "iscrowd": 0,
                "area": box_anno['data']['width'] * box_anno['data']['height'],
                "bbox": [
                    box_anno['data']['x'], box_anno['data']['y'],
                    box_anno['data']['width'], box_anno['data']['height']
                ]
            })
            type_cnt[categories_dict[box_anno['category']['type']]] += 1
    print(type_cnt)
    print("Writing new annotations to " + anno_file_path)
    with open(anno_file_path, 'w', encoding='utf-8') as fp:
        json.dump(anno_file, fp)


def convert_camera(origin_anno_path, key_list, target_anno_path):
    with open(origin_anno_path, 'r', encoding='utf-8') as fp:
        img_anno = json.load(fp)
    anno_file = {"images": [], "categories": categories, "annotations": []}
    type_cnt = {cid: 0 for cid in categories_dict.values()}
    for img_id, key_str in tqdm(enumerate(key_list), total=len(key_list)):
        path_split = key_str.split('/')
        basename = os.path.join(path_split[-2], path_split[-1])
        anno_file['images'].append({
            "id": img_id,
            "width": 1280,
            "height": 720,
            "file_name": basename
        })
        for category in img_anno[key_str]:
            if category not in categories_dict:
                continue
            for box_anno in img_anno[key_str][category]:
                try:
                    assert len(box_anno) == 4
                    assert box_anno[0] <= box_anno[2] and box_anno[1] <= box_anno[3]
                except AssertionError:
                    print("Invalid box annotation: " + str(box_anno))
                    continue
                anno_file['annotations'].append({
                    "id": len(anno_file['annotations']),
                    "image_id": img_id,
                    "category_id": categories_dict[category],
                    "iscrowd": 0,
                    "area": (box_anno[2] - box_anno[0]) * (box_anno[3] - box_anno[1]),
                    "bbox": [
                        box_anno[0], box_anno[1],
                        (box_anno[2] - box_anno[0]), (box_anno[3] - box_anno[1])
                    ]
                })
                type_cnt[categories_dict[category]] += 1
    print(type_cnt)
    print("Writing new annotations to " + target_anno_path)
    with open(target_anno_path, 'w', encoding='utf-8') as fp:
        json.dump(anno_file, fp)


def split_train_val(data_dir, img_path_list):
    print("Processing " + data_dir)
    print("Total images: " + str(len(img_path_list)))
    random.shuffle(img_path_list)
    train_size = int(len(img_path_list) * training_set_ratio)
    train_img_path_list, val_img_path_list = img_path_list[:train_size], img_path_list[train_size:]
    print("Split into training set size: " + str(len(train_img_path_list)),
          " validation set size: " + str(len(val_img_path_list)))
    return train_img_path_list, val_img_path_list


def weather(subset_name):
    data_dir = os.path.join(data_root, subset_name)
    # Read all images and split train/val
    img_path_list = []
    for root, dirs, files in os.walk(data_dir):
        if 'images_train' in root or 'images_val' in root:
            continue
        for file in files:
            if file.endswith('.jpg'):
                img_path = os.path.join(root, file)
                if os.path.exists(img_path.replace('jpg', 'json')):
                    img_path_list.append(img_path)
    train_img_path_list, val_img_path_list = split_train_val(data_dir, img_path_list)
    # Create sub directory
    if not os.path.exists(os.path.join(data_dir, 'annotations')):
        os.mkdir(os.path.join(data_dir, 'annotations'))
    # Convert annotations
    convert_weather(train_img_path_list, os.path.join(data_dir, 'annotations', 'train_coco_style.json'))
    convert_weather(val_img_path_list, os.path.join(data_dir, 'annotations', 'val_coco_style.json'))


def camera(subset_name):
    data_dir = os.path.join(data_root, subset_name)
    # Read all images and split train/val
    anno_file_path = os.path.join(data_dir, subset_name + '_gt.json')
    with open(anno_file_path, 'r', encoding='utf-8') as fp:
        img_anno = json.load(fp)
    anno_keys = list(img_anno.keys())
    train_key_list, val_key_list = split_train_val(data_dir, anno_keys)
    # Create sub directory
    if not os.path.exists(os.path.join(data_dir, 'annotations')):
        os.mkdir(os.path.join(data_dir, 'annotations'))
    # Convert annotations
    convert_camera(anno_file_path, train_key_list, os.path.join(data_dir, 'annotations', 'train_coco_style.json'))
    convert_camera(anno_file_path, val_key_list, os.path.join(data_dir, 'annotations', 'val_coco_style.json'))


if __name__ == '__main__':
    data_root = '/network_space/storage43/zhaozijing/datasets/tencent/'
    set_seed(0)
    training_set_ratio = 0.8
    """for subset in ['day_sunny', 'day_overcast', 'day_light_rain', 'day_heavy_rain', 'night_clear']:
        weather(subset)"""
    for subset in ['camera_a', 'camera_b']:
        camera(subset)
