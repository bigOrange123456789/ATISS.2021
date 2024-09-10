# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

"""
Script used for parsing the 3D-FRONT data scenes into numpy files in order
    to be able to avoid I/O overhead when training our model.
"""
import argparse
import logging
import json
import os
import sys

import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm

from utils import DirLock, ensure_parent_directory_exists, \
    floor_plan_renderable, floor_plan_from_scene, \
    get_textured_objects_in_scene, scene_from_args, render

from scene_synthesis.datasets import filter_function
from scene_synthesis.datasets.threed_front import ThreedFront
from scene_synthesis.datasets.threed_front_dataset import \
    dataset_encoding_factory

import time #用于异步程序的等待

def main(argv):
    parser = argparse.ArgumentParser(
        description="Prepare the 3D-FRONT scenes to train our model"
    )
    parser.add_argument(
        "output_directory",
        default="/tmp/",
        help="Path to output directory"
    )
    parser.add_argument(
        "path_to_3d_front_dataset_directory",
        help="Path to the 3D-FRONT dataset"
    )
    parser.add_argument(
        "path_to_3d_future_dataset_directory",
        help="Path to the 3D-FUTURE dataset"
    )
    parser.add_argument(
        "path_to_model_info",
        help="Path to the 3D-FUTURE model_info.json file"
    )
    parser.add_argument(
        "path_to_floor_plan_textures",
        help="Path to floor texture images"
    )
    parser.add_argument(
        "--path_to_invalid_scene_ids",
        default="../config/invalid_threed_front_rooms.txt",
        help="Path to invalid scenes"
    )
    parser.add_argument(
        "--path_to_invalid_bbox_jids",
        default="../config/black_list.txt",
        help="Path to objects that ae blacklisted"
    )
    parser.add_argument(
        "--annotation_file",
        default="../config/bedroom_threed_front_splits.csv",
        help="Path to the train/test splits file"
    )
    # parser.add_argument(
    #     "--path_to_invalid_bbox_jids",
    #     default="./ATISS.2021/config/black_list.txt",
    #     help="Path to objects that ae blacklisted"
    # )
    # parser.add_argument(
    #     "--path_to_invalid_scene_ids",
    #     default="./ATISS.2021/config/invalid_threed_front_rooms.txt",
    #     help="Path to invalid scenes"
    # )
    # parser.add_argument(
    #     "--annotation_file",
    #     default="./ATISS.2021/config/bedroom_threed_front_splits.csv",
    #     help="Path to the train/test splits file"
    # )
    parser.add_argument(
        "--room_side",
        type=float,
        default=3.1,
        help="The size of the room along a side (default:3.1)"
    )
    parser.add_argument(
        "--dataset_filtering",
        default="threed_front_bedroom",
        choices=[
            "threed_front_bedroom",
            "threed_front_livingroom",
            "threed_front_diningroom",
            "threed_front_library"
        ],
        help="The type of dataset filtering to be used"
    )
    parser.add_argument(
        "--without_lamps",
        action="store_true",
        help="If set ignore lamps when rendering the room"
    )
    parser.add_argument(
        "--up_vector",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,0,-1",
        help="Up vector of the scene"
    )
    parser.add_argument(
        "--background",
        type=lambda x: list(map(float, x.split(","))),
        default="0,0,0,1",
        help="Set the background of the scene"
    )
    parser.add_argument(
        "--camera_target",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,0,0",
        help="Set the target for the camera"
    )
    parser.add_argument(
        "--camera_position",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,4,0",
        help="Camer position in the scene"
    )
    parser.add_argument(
        "--window_size",
        type=lambda x: tuple(map(int, x.split(","))),
        default="256,256",
        help="Define the size of the scene and the window"
    )

    args = parser.parse_args(argv)
    logging.getLogger("trimesh").setLevel(logging.ERROR)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory): # 检测输出路径是否存在
        os.makedirs(args.output_directory)

    # Create the scene and the behaviour list for simple-3dviz
    scene = scene_from_args(args)

    # 打开文件： ../config/invalid_threed_front_rooms.txt
    with open(args.path_to_invalid_scene_ids, "r") as f:
        invalid_scene_ids = set(l.strip() for l in f)

    # 打开文件： ../config/black_list.txt
    with open(args.path_to_invalid_bbox_jids, "r") as f:
        invalid_bbox_jids = set(l.strip() for l in f)

    config = {
        "filter_fn":                 args.dataset_filtering, # threed_front_bedroom
        "min_n_boxes":               -1,
        "max_n_boxes":               -1,
        "path_to_invalid_scene_ids": args.path_to_invalid_scene_ids, # ../config/invalid_threed_front_rooms.txt
        "path_to_invalid_bbox_jids": args.path_to_invalid_bbox_jids, # ../config/black_list.txt
        "annotation_file":           args.annotation_file # ../config/bedroom_threed_front_splits.csv
    }

    # Initially, we only consider the train split to compute the dataset
    # statistics, e.g the translations, sizes and angles bounds
    # 最初，我们只考虑分割训练来计算数据集 # 统计数据，例如平移、大小和角度边界
    dataset = ThreedFront.from_dataset_directory(
        dataset_directory=args.path_to_3d_front_dataset_directory, # '3D-FRONT'
        path_to_model_info=args.path_to_model_info, # '3D-FUTURE-model/model_info.json'
        path_to_models=args.path_to_3d_future_dataset_directory, # '3D-FUTURE-model'
        filter_fn=filter_function(config, ["train", "val"], args.without_lamps) # <function BaseDataset>
    )
    print("dataset:",dataset) # dataset: Dataset contains 26 scenes with 17 discrete types
    print("Loading dataset with {} rooms".format(len(dataset))) #Loading dataset with 27 rooms

    # Compute the bounds for the translations, sizes and angles in the dataset.
    # 计算数据集中平移、大小和角度的边界。
    # This will then be used to properly align rooms.
    # 然后，这将用于正确对齐房间。
    tr_bounds = dataset.bounds["translations"]
    # 移动: (array([-2.39974681,  0.175839  , -1.82630077]), array([2.147153  , 3.22758752, 2.38883496]))
    si_bounds = dataset.bounds["sizes"]
    # 放缩: (array([0.03998288, 0.0557985 , 0.03462618]), array([2.495   , 1.26    , 1.698315]))
    an_bounds = dataset.bounds["angles"]
    # 旋转: (array([-3.14159265]), array([3.14159265]))

    dataset_stats = {
        "bounds_translations": tr_bounds[0].tolist() + tr_bounds[1].tolist(),
        # [
        #     -2.399746809867355, 0.175839, -1.8263007663849495,
        #     2.1471530000000003, 3.227587519676172, 2.3888349638570734
        # ]
        "bounds_sizes": si_bounds[0].tolist() + si_bounds[1].tolist(),
        # [
        #     0.0399828836528625, 0.055798500000000084, 0.034626179184937544,
        #     2.495, 1.26, 1.698315
        # ]
        "bounds_angles": an_bounds[0].tolist() + an_bounds[1].tolist(),
        # [ -3.141592653589793,    3.141592653589793    ]
        "class_labels": dataset.class_labels,
        # ['armchair', 'bookshelf', 'cabinet', 'ceiling_lamp', 'chair', 'desk',
        #  'double_bed', 'dressing_chair', 'dressing_table', 'nightstand', 'pendant_lamp',
        #  'shelf', 'single_bed', 'stool', 'table', 'tv_stand', 'wardrobe', 'start', 'end'],
        "object_types": dataset.object_types,
        # ['armchair', 'bookshelf', 'cabinet', 'ceiling_lamp', 'chair', 'desk',
        #  'double_bed', 'dressing_chair', 'dressing_table', 'nightstand', 'pendant_lamp',
        #  'shelf', 'single_bed', 'stool', 'table', 'tv_stand', 'wardrobe']
        "class_frequencies": dataset.class_frequencies,
        # { 'nightstand': 0.26515151515151514,'double_bed': 0.1893939393939394,'wardrobe': 0.13636363636363635,
        #   'pendant_lamp': 0.12121212121212122,'chair': 0.05303030303030303, 'bookshelf': 0.05303030303030303,'ceiling_lamp': 0.045454545454545456,
        #   'dressing_table': 0.030303030303030304,'table': 0.022727272727272728, 'tv_stand': 0.015151515151515152,'desk': 0.015151515151515152, 'cabinet': 0.015151515151515152,
        #   'single_bed': 0.007575757575757576,'shelf': 0.007575757575757576, 'armchair': 0.007575757575757576, 'stool': 0.007575757575757576,'dressing_chair': 0.007575757575757576}
        "class_order": dataset.class_order,
        # {'nightstand': 0, 'double_bed': 1, 'wardrobe': 2,
        #  'pendant_lamp': 3, 'chair': 4, 'bookshelf': 5, 'ceiling_lamp': 6,
        #  'dressing_table': 7, 'table': 8, 'tv_stand': 9, 'desk': 10, 'cabinet': 11,
        #  'single_bed': 12, 'shelf': 13, 'armchair': 14, 'stool': 15, 'dressing_chair': 16}
        "count_furniture": dataset.count_furniture
        # [ ('nightstand', 35), ('double_bed', 25), ('wardrobe', 18),
        #   ('pendant_lamp', 16), ('chair', 7), ('bookshelf', 7), ('ceiling_lamp', 6),
        #   ('dressing_table', 4), ('table', 3), ('tv_stand', 2), ('desk', 2), ('cabinet', 2),
        #   ('single_bed', 1), ('shelf', 1), ('armchair', 1), ('stool', 1), ('dressing_chair', 1)]
    }

    path_to_json = os.path.join(args.output_directory, "dataset_stats.txt")
    # path_to_json: ../../Dataset/out/dataset_stats.txt
    with open(path_to_json, "w") as f:
        json.dump(dataset_stats, f)
        print("The file of dataset_stats has saved!")#print("dataset_stats:",dataset_stats)

    # 下面这一段似乎是冗余的代码
    # dataset = ThreedFront.from_dataset_directory(
    #     dataset_directory=args.path_to_3d_front_dataset_directory, # '../../Dataset/3D-FRONT'
    #     path_to_model_info=args.path_to_model_info, # '../../Dataset/3D-FUTURE-model/model_info.json'
    #     path_to_models=args.path_to_3d_future_dataset_directory, # '../../Dataset/3D-FUTURE-model'
    #     filter_fn=filter_function( # <function BaseDataset.filter_compose.<locals>.inner>
    #         config, ["train", "val", "test"], args.without_lamps
    #     )
    # )
    # print("Loading dataset with {} rooms".format(len(dataset))) # Loading dataset with 26 rooms

    encoded_dataset = dataset_encoding_factory(
        "basic", dataset, augmentations=None, box_ordering=None
    ) # <scene_synthesis.datasets.threed_front_dataset.DatasetCollection object>

    for (i, es), ss in tqdm(zip(enumerate(encoded_dataset), dataset)): #这个循环首次执行的时候会给出一句输出
        # ss: Scene: MasterBedroom-2888 of type: masterbedroom contains 7 bboxes
        # ss.uid: 0a25c251-7c80-4808-b609-3d6fbae9efad_MasterBedroom-2888
        # Create a separate folder for each room # 为每个房间创建一个单独的文件夹
        room_directory = os.path.join(args.output_directory, ss.uid)
        # print("room_directory:",room_directory)
        # room_directory: ../../Dataset/out/0a25c251-7c80-4808-b609-3d6fbae9efad_MasterBedroom-2888
        # Check if room_directory exists and if it doesn't create it # 检查room_directory是否存在，以及它是否没有创建
        if os.path.exists(room_directory): # 如果这个房间已经处理完了就处理下一个房间
            if len(os.listdir(room_directory))==3:#如果文件数量为2说明之前这个场景没有处理完成
                continue
        # print('flag')
        # print('room_directory:',room_directory)
        # Make sure we are the only ones creating this file # 确保我们是唯一创建此文件的人
        with DirLock(room_directory + ".lock") as lock:
            # lock.is_acquired: True
            if not lock.is_acquired:
                continue
            if os.path.exists(room_directory):
                if len(os.listdir(room_directory))==3:#如果文件数量为2说明之前这个场景没有处理完成
                    continue
            ensure_parent_directory_exists(room_directory)

            # ss.bboxes: [ 7 * <scene_synthesis.ThreedFutureModel]
            uids = [bi.model_uid for bi in ss.bboxes] # [x*x for x in range(5)]=[0, 1, 4, 9, 16]
            # uids: ['12732/model1', '6875/model', '12789/model', '12789/model', '16432/model', '29045/model', '47513/model']
            jids = [bi.model_jid for bi in ss.bboxes]
            # jids: ['e3d33704-98af-4bb2-afde-fa8079694aff', '58ce0893-d787-4f80-807b-e0f96baca1ab', '5842c4d2-f2da-473e-91dc-48395b63f382', '5842c4d2-f2da-473e-91dc-48395b63f382', 'dabf5085-308d-496b-9a78-cdff7df6d619', '5631b63c-24b2-4074-857b-425794858ca9', 'da2f4cae-4c21-4671-aaaf-2eb1f5424f16']

            # ss.floor_plan: (array([[-0.79839, 0., -5.38639],
            #                        [-4.23072, 0., -1.09086],
            #                        [-4.23072, 0., -5.38639],
            #                        [-4.23072, 0., -1.09086],
            #                        [-0.79839, 0., -5.38639],
            #                        [-0.79839, 0., -1.09086]]), #顶点坐标
            #                 array([[0, 2, 1],[3, 5, 4]]))      #三角面索引
            floor_plan_vertices, floor_plan_faces = ss.floor_plan

            # Render and save the room mask as an image # 渲染房间mask并将其另存为图像
            room_mask = render(
                scene, # <simple_3dviz.scenes.Scene>
                [floor_plan_renderable(ss)], # [<simple_3dviz.renderables.mesh.Mesh>]
                (1.0, 1.0, 1.0),
                "flat",
                os.path.join(room_directory, "room_mask.png") # '../../Dataset/out/0a25c251-7c80-4808-b609-3d6fbae9efad_SecondBedroom-60495/room_mask.png'
            )[:, :, 0:1]
            # room_mask.shape: (256, 256, 1)

            # print("class_labels", es["class_labels"].shape)
            # print(es["class_labels"][1])
            np.savez_compressed(#为啥有4个物体的索引却只有三个物体的参数？
                os.path.join(room_directory, "boxes"), # '../../Dataset/out/0a25c251-7c80-4808-b609-3d6fbae9efad_SecondBedroom-60495/boxes'
                uids=uids, # ['78034/model', '84768/model', '85841/model', '85967/model']
                jids=jids, # ['36a8ee08-14c3-4380-ae0e-1a66525aa7f5', 'faa6f628-cd9f-439a-9dc2-c483c9c3854f', '1a1c11a4-9c90-403c-b7cf-70ee73b58cdd', '441e1921-ed75-4191-ad0f-397b884d1021']
                scene_id=ss.scene_id, # 'SecondBedroom-60495'
                scene_uid=ss.uid, # '0a25c251-7c80-4808-b609-3d6fbae9efad_SecondBedroom-60495'
                scene_type=ss.scene_type, # 'secondbedroom'
                json_path=ss.json_path, # '0a25c251-7c80-4808-b609-3d6fbae9efad'
                room_layout=room_mask,  # room_mask.shape: (256, 256, 1)
                floor_plan_vertices=floor_plan_vertices, # floor_plan_vertices.shape: (12, 3)
                floor_plan_faces=floor_plan_faces,       # floor_plan_faces.shape: (4, 3)
                floor_plan_centroid=ss.floor_plan_centroid,# [1.84475, 0. , 1.837  ]
                class_labels=es["class_labels"],
                # class_labels.shape (7, 19) #不知道这个是啥
                # class_labels[0]=[0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
                translations=es["translations"], #我猜测是有三个物体实例
                # 'translations': array([
                #        [ 0.51856816,  0.5155175 ,  1.2077913 ],
                #        [ 1.4256525 ,  0.24823052,  2.388835  ],
                #        [ 0.5268733 ,  1.09088   , -0.32612094]], dtype = float32)
                sizes=es["sizes"],
                # 'sizes': array([
                #        [0.9852345 , 0.5150675 , 1.0696245 ],
                #        [0.195358  , 0.24823047, 0.18308   ],
                #        [1.10695   , 1.09088   , 0.2952655 ]], dtype=float32)
                angles=es["angles"]
                # 'angles': array([
                #        [-1.5707872],[-1.5707872],[ 0.       ]], dtype=float32)}
            )

            # Render a top-down orthographic projection of the room at a specific pixel resolutin
            # 以特定像素分辨率渲染房间的自上而下正交投影
            path_to_image = "{}/rendered_scene_{}.png".format(
                room_directory, args.window_size[0]
            ) # path_to_image: ../../Dataset/out/0a761819-05d1-4647-889b-a726747201b1_MasterBedroom-24539/rendered_scene_256.png
            if os.path.exists(path_to_image):
                continue

            # 获取要渲染的平面图的simple_3dviz网格 # Get a simple_3dviz Mesh of the floor plan to be rendered
            # time.sleep(2.5)  # 休眠5秒 ，防止报错 #休眠5秒仍然会报错 #休眠2.5秒仍然会报错
            floor_plan, _, _ = floor_plan_from_scene(
                ss, # <scene_synthesis.datasets.threed_front_scene.Room object at 0x7a85ce77d760>
                args.path_to_floor_plan_textures, # '../../Dataset/3D-FRONT-texture'
                without_room_mask=True
            )
            # print("floor_plan:",floor_plan)
            # print("ss:",ss)
            # print("args.without_lamps:",args.without_lamps)
            renderables = get_textured_objects_in_scene(
                ss, ignore_lamps=args.without_lamps
            )
            # print("renderables:",renderables)
            # print("\t\t\t********{}********".format(room_directory),end = '\r')
            render(
                scene,
                renderables + floor_plan,
                color=None,
                mode="shading",
                frame_path=path_to_image
            )


if __name__ == "__main__":
    main(sys.argv[1:])
# python preprocess_data.py out 3D-FRONT 3D-FUTURE-model 3D-FUTURE-model/model_info.json 3D-FRONT-texture --dataset_filtering threed_front_bedroom
# ../../Dataset/
# python preprocess_data.py ../../Dataset/out-preprocess ../../Dataset/3D-FRONT ../../Dataset/3D-FUTURE-model ../../Dataset/3D-FUTURE-model/model_info.json ../../Dataset/3D-FRONT-texture --dataset_filtering threed_front_bedroom

# test
# python preprocess_data.py ../../Dataset/out-preprocess ../../Dataset/3D-FRONT-TEST ../../Dataset/3D-FUTURE-model ../../Dataset/3D-FUTURE-model/model_info.json ../../Dataset/3D-FRONT-texture --dataset_filtering threed_front_bedroom



# desktop
# python ./ATISS.2021/scripts/preprocess_data.py ./Dataset/out-preprocess ./Dataset/3D-FRONT ./Dataset/3D-FUTURE-model ./Dataset/3D-FUTURE-model/model_info.json ./Dataset/3D-FRONT-texture --dataset_filtering threed_front_bedroom


