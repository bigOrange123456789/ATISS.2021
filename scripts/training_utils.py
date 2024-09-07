# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
import json

import string
import os
import random
import subprocess

#################################lzc-add-begin#################################
import pickle
from scene_synthesis.datasets import filter_function
from scene_synthesis.datasets.threed_front import ThreedFront
from scene_synthesis.datasets.threed_future_dataset import ThreedFutureDataset
class Args0:
    def __init__(self):
        self.dataset_filtering='threed_front_bedroom'
        self.path_to_invalid_scene_ids='../config/invalid_threed_front_rooms.txt'
        self.path_to_invalid_bbox_jids='../config/black_list.txt'
        self.annotation_file='../config/bedroom_threed_front_splits.csv'
        self.path_to_3d_front_dataset_directory='../../Dataset/3D-FRONT-TEST'
        self.path_to_model_info='../../Dataset/3D-FUTURE-model/model_info.json'
        self.path_to_3d_future_dataset_directory='../../Dataset/3D-FUTURE-model'
        self.without_lamps=False
        self.dataset_filtering= 'threed_front_bedroom'
        self.output_directory='../../Dataset/out-pickle'
def lzc_pickle():
    args0=Args0()
    config = {
        "filter_fn":                 args0.dataset_filtering,
        "min_n_boxes":               -1,
        "max_n_boxes":               -1,
        "path_to_invalid_scene_ids": args0.path_to_invalid_scene_ids,
        "path_to_invalid_bbox_jids": args0.path_to_invalid_bbox_jids,
        "annotation_file":           args0.annotation_file
    }

    # Initially, we only consider the train split to compute the dataset
    # statistics, e.g the translations, sizes and angles bounds
    scenes_dataset = ThreedFront.from_dataset_directory(
        dataset_directory=args0.path_to_3d_front_dataset_directory,
        path_to_model_info=args0.path_to_model_info,
        path_to_models=args0.path_to_3d_future_dataset_directory,
        filter_fn=filter_function(config, ["train", "val"], args0.without_lamps)
    )
    print("Loading dataset with {} rooms".format(len(scenes_dataset)))

    # Collect the set of objects in the scenes
    objects = {}
    for scene in scenes_dataset:
        for obj in scene.bboxes:
            objects[obj.model_jid] = obj
    objects = [vi for vi in objects.values()]
    # return objects

    objects_dataset = ThreedFutureDataset(objects)
    print("objects_dataset[data]:", objects_dataset["data"])
    print("objects_dataset:",objects_dataset)
    return objects_dataset
    # room_type = args0.dataset_filtering.split("_")[-1]
    # output_path = "{}/threed_future_model_{}.pkl".format(
    #     args0.output_directory,
    #     room_type
    # )
    # with open(output_path, "wb") as f:
    #     pickle.dump(objects_dataset, f)
#################################lzc-add-end  #################################

def load_config(config_file):
    # config_file: ../../Dataset/out-pickle/threed_future_model_bedroom.pkl
    # --lzc [
    import chardet
    with open(config_file, 'rb') as f: # --lzc
        result = chardet.detect(f.read())  # 读取一定量的数据进行编码检测 # --lzc
        print('打印检测到的编码:',result['encoding'],type(result['encoding']))  # 打印检测到的编码
        # with open(config_file, 'r', result['encoding']) as f0:
        #     config = yaml.load(f0, Loader=Loader)
    # return config
        # 编码格式: Windows-1254
    if False:
        return lzc_pickle()
    if False:
        import pickle
        fr = open(config_file, 'rb')
        inf = pickle.load(fr)
        # print("inf:",inf)
        fr.close()
        return inf
    # --lzc ]
    encoding = 'Windows-1254'#'iso-8859-1'  # 假设检测到的编码是ISO-8859-1
    # with open(config_file, 'r', encoding=encoding) as f:
    print("config_file:",config_file)
    # config_file: ../../Dataset/out-pickle/threed_future_model_bedroom.pkl
    with open(config_file, "r") as f:
        # with open(config_file, "r", encoding='Windows-1254') as f:
        # print("f:",f)# --lzc
        print("lzc:t01")
        config = yaml.load(f, Loader=Loader)
        print("lzc:t02")
    return config


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))



def save_experiment_params(args, experiment_tag, directory):
    t = vars(args)
    # tag1
    # args: Namespace(
    #   config_file='../../Dataset/out-pickle/threed_future_model_bedroom.pkl',
    #   continue_from_epoch=0,
    #   experiment_tag=None,
    #   n_processes=0,
    #   output_directory='../../Dataset/out-train',
    #   seed=27,
    #   weight_file=None,
    #   with_wandb_logger=False)
    # t: {
    #   'config_file': '../../Dataset/out-pickle/threed_future_model_bedroom.pkl',
    #   'continue_from_epoch': 0,
    #   'experiment_tag': None,
    #   'n_processes': 0,
    #   'output_directory': '../../Dataset/out-train',
    #   'seed': 27,
    #   'weight_file': None,
    #   'with_wandb_logger': False
    # }
    params = {k: str(v) for k, v in t.items()} # 将所有数值转化为字符串格式
    # params: {
    #   'config_file': '../../Dataset/out-pickle/threed_future_model_bedroom.pkl',
    #   'output_directory': '../../Dataset/out-train',
    #   'weight_file': 'None',
    #   'continue_from_epoch': '0',
    #   'n_processes': '0',
    #   'seed': '27',
    #   'experiment_tag': 'None',
    #   'with_wandb_logger': 'False'}

    git_dir = os.path.dirname(os.path.realpath(__file__))
    # git_dir: /home/lzc/桌面/ATISS.2021/scripts
    git_head_hash = "foo"
    try:
        git_head_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
        ).strip()
        # git_head_hash: b'7c0b4a4ff41f3b45d50e06d1a34f615260fe4360'
    except subprocess.CalledProcessError:
        # Keep the current working directory to move back in a bit
        cwd = os.getcwd()
        os.chdir(git_dir)
        git_head_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
        ).strip()
        os.chdir(cwd)
    params["git-commit"] = str(git_head_hash)
    # git-commit: b'7c0b4a4ff41f3b45d50e06d1a34f615260fe4360'
    params["experiment_tag"] = experiment_tag
    # experiment_tag: 73W51082I
    for k, v in list(params.items()):
        if v == "":
            params[k] = None #这里没有被执行过
    if hasattr(args, "config_file"): # True
        # args.config_file: ../../Dataset/out-pickle/threed_future_model_bedroom.pkl
        print("lzc:test10001")
        config = load_config(args.config_file) # lzc_pickle() #
        # print("finish!")
        # exit(0)
        # print("config:" , config )
        print("lzc:test10002")
        params.update(config)
    with open(os.path.join(directory, "params.json"), "w") as f:
        json.dump(params, f, indent=4)
