# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

"""Script used for generating scenes using a previously trained model."""
import argparse
import logging
import os
import sys

import numpy as np
import torch

from training_utils import load_config
from utils import floor_plan_from_scene, export_scene

from scene_synthesis.datasets import filter_function, \
    get_dataset_raw_and_encoded
from scene_synthesis.datasets.threed_future_dataset import ThreedFutureDataset
from scene_synthesis.networks import build_network
from scene_synthesis.utils import get_textured_objects

from simple_3dviz import Scene
# lzc # from simple_3dviz.window import show
from simple_3dviz.behaviours.keyboard import SnapshotOnKey, SortTriangles
from simple_3dviz.behaviours.misc import LightToCamera
from simple_3dviz.behaviours.movements import CameraTrajectory
from simple_3dviz.behaviours.trajectory import Circle
from simple_3dviz.behaviours.io import SaveFrames, SaveGif
from simple_3dviz.utils import render

def saveJson(data,path):
    import json
    b = json.dumps(data)
    f2 = open( path , 'w') # f2 = open('new_json.json', 'w')
    f2.write(b)
    f2.close()
def save_lzc(bbox_params_t, objects_dataset, classes, pathPre):#lzc
    # bbox_params_t：通过网络得到的场景中每个物体的特征
    # objects_dataset：打包后所有物体的参数
    # classes：物体的类别标签
    # For each one of the boxes replace them with an object # 对于每个框，用一个对象替换它们
    my_scene = { # LZC
        'id': 'scene_generate',
        'centroid': [0,0,0],#s.centroid.tolist(),
        'meshes': []
    }
    # bbox_params_t.shape[1]-1: 5
    for j in range(1, bbox_params_t.shape[1]-1): # bbox_params_t.shape[1]-1=11
        # bbox_params_t: shape=(1, 8, 26) type=<class 'numpy.ndarray'>
        query_size = bbox_params_t[0, j, -4:-1] # query_size: [0.21811413 0.48709698 0.3640413 ]
        query_label = classes[bbox_params_t[0, j, :-7].argmax(-1)] # query_label: chair
        furniture = objects_dataset.get_closest_furniture_to_box(
            query_label, query_size
        )
        my_scene['meshes'].append({
            'raw_model_path': furniture.raw_model_path, #
            'texture_image_path': furniture.texture_image_path,
            'scale': furniture.scale, #
            'z_angle': bbox_params_t[0, j, -1],#furniture.z_angle,
            'position': bbox_params_t[0, j, -7:-4].tolist()# furniture.position #仿射变换之前包围盒中心必须在原点
        })
    saveJson(my_scene, pathPre+'/'+my_scene['id']+".json")
    return my_scene

def main(argv):
    parser = argparse.ArgumentParser(
        description="Generate scenes using a previously trained model"
    )

    parser.add_argument(
        "config_file",
        help="Path to the file that contains the experiment configuration"
    )
    parser.add_argument(
        "output_directory",
        default="/tmp/",
        help="Path to the output directory"
    )
    parser.add_argument(
        "path_to_pickled_3d_futute_models",
        help="Path to the 3D-FUTURE model meshes"
    )
    parser.add_argument(
        "path_to_floor_plan_textures",
        help="Path to floor texture images"
    )
    parser.add_argument(
        "--weight_file",
        default=None,
        help="Path to a pretrained model"
    )
    parser.add_argument(
        "--n_sequences",
        default=10,
        type=int,
        help="The number of sequences to be generated" # 要生成的序列数
    )
    parser.add_argument(
        "--background",
        type=lambda x: list(map(float, x.split(","))),
        default="1,1,1,1",
        help="Set the background of the scene"
    )
    parser.add_argument(
        "--up_vector",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,1,0",
        help="Up vector of the scene"
    )
    parser.add_argument(
        "--camera_position",
        type=lambda x: tuple(map(float, x.split(","))),
        default="-0.10923499,1.9325259,-7.19009",
        help="Camer position in the scene"
    )
    parser.add_argument(
        "--camera_target",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,0,0",
        help="Set the target for the camera"
    )
    parser.add_argument(
        "--window_size",
        type=lambda x: tuple(map(int, x.split(","))),
        default="512,512",
        help="Define the size of the scene and the window"
    )
    parser.add_argument(
        "--with_rotating_camera",
        action="store_true",
        help="Use a camera rotating around the object"
    )
    parser.add_argument(
        "--save_frames",
        help="Path to save the visualization frames to"
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=360,
        help="Number of frames to be rendered"
    )
    parser.add_argument(
        "--without_screen",
        action="store_true",
        help="Perform no screen rendering"
    )
    parser.add_argument(
        "--scene_id",
        default=None,
        help="The scene id to be used for conditioning"
    )

    args = parser.parse_args(argv)

    # Disable trimesh's logger # 禁用trimesh的记录器
    logging.getLogger("trimesh").setLevel(logging.ERROR)

    if torch.cuda.is_available(): #可以获取cuda
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Running code on", device) # Running code on cuda:0

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)# ../../Dataset/generate_out

    config = load_config(args.config_file) # config对应yaml配置文件中的值

    raw_dataset, train_dataset = get_dataset_raw_and_encoded(
        config["data"],
        filter_fn=filter_function(
            config["data"],
            split=config["training"].get("splits", ["train", "val"])
            # split=["train", "val"]
        ),
        split=config["training"].get("splits", ["train", "val"])
    )
    # raw_dataset: Dataset contains 26 scenes with 17 discrete types
    # train_dataset: <scene_synthesis.datasets.threed_front_dataset.AutoregressiveWOCM object at 0x7f23fce17df0>

    # Build the dataset of 3D models
    objects_dataset = ThreedFutureDataset.from_pickled_dataset(
        args.path_to_pickled_3d_futute_models
    )
    print("Loaded {} 3D-FUTURE models".format(len(objects_dataset))) # Loaded 104 3D-FUTURE models

    raw_dataset, dataset = get_dataset_raw_and_encoded(
        config["data"],
        filter_fn=filter_function(
            config["data"],
            split=config["validation"].get("splits", ["test"])
        ),
        split=config["validation"].get("splits", ["test"])
    )
    # raw_dataset: Dataset contains 1 scenes with 17 discrete types
    # dataset: <scene_synthesis.datasets.threed_front_dataset.AutoregressiveWOCM object at 0x7737848e80a0>
    print("Loaded {} scenes with {} object types:".format(
        len(dataset), dataset.n_object_types)
    ) # Loaded 1 scenes with 17 object types:

    network, _, _ = build_network(
        dataset.feature_size, dataset.n_classes,
        config, args.weight_file, device=device
    )
    network.eval() # 将网络切换为eval模式 # 不计算梯度
    # 在eval模式下，dropout层会让所有的激活单元都通过，而BN层会停止计算和更新mean和var，直接使用在训练阶段已经学出的mean和var值

    # Create the scene and the behaviour list for simple-3dviz # 为simple-3dviz创建场景和行为列表
    scene = Scene(size=args.window_size)
    scene.up_vector = args.up_vector
    scene.camera_target = args.camera_target
    scene.camera_position = args.camera_position
    scene.light = args.camera_position

    given_scene_id = None
    if args.scene_id: # 这部分代码没有被执行 # args.scene_id: None
        for i, di in enumerate(raw_dataset):
            print('di.scene_id:',di.scene_id,i)
            if str(di.scene_id) == args.scene_id:
                given_scene_id = i

    classes = np.array(dataset.class_labels) # # raw_dataset: Dataset contains 1 scenes with 17 discrete types
    # classes: [
    #  'armchair' 'bookshelf' 'cabinet' 'ceiling_lamp' 'chair'
    #  'desk' 'double_bed' 'dressing_chair' 'dressing_table' 'nightstand'
    #  'pendant_lamp' 'shelf' 'single_bed' 'stool' 'table'
    #  'tv_stand' 'wardrobe'
    #  'start' 'end']
    print('args.n_sequences:',args.n_sequences) #args.n_sequences: 10
    for i in range(args.n_sequences): # args.n_sequences 10 (预先指定的序列数)
        # 序列编号为： 0
        # 0 / 10: Using the 0 floor plan of scene SecondBedroom-6482
        scene_idx = given_scene_id or np.random.choice(len(dataset)) # len(dataset): 1
        # given_scene_id: None
        # scene_idx: 0
        current_scene = raw_dataset[scene_idx]
        # current_scene: <scene_synthesis.datasets.threed_front.CachedRoom object at 0x714303335fa0>
        print("{} / {}: Using the {} floor plan of scene {}".format(
            i, args.n_sequences, scene_idx, current_scene.scene_id)
        )
        # Get a floor plan
        floor_plan, tr_floor, room_mask = floor_plan_from_scene(
            current_scene, args.path_to_floor_plan_textures
        )
        # floor_plan: [<simple_3dviz.**.TexturedMesh>]
        # tr_floor: [<trimesh.Trimesh(vertices.shape=(6, 3), faces.shape=(2, 3))>]
        # room_mask: <class 'torch.Tensor'>, torch.Size([1, 1, 64, 64])

        # device: cuda:0
        bbox_params = network.generate_boxes(
            room_mask=room_mask.to(device),
            device=device
        )
        # bbox_params:  # 输出9个对象？(其中两个对象是开始和结束？)
        #   class_labels torch.Size([1, 9, 19]) <class 'torch.Tensor'>
        #   translations torch.Size([1, 9, 3])  <class 'torch.Tensor'>
        #   sizes        torch.Size([1, 9, 3])  <class 'torch.Tensor'>
        #   angles       torch.Size([1, 9, 1])  <class 'torch.Tensor'>
        boxes = dataset.post_process(bbox_params)
        # boxes: # 只有7个有效输出对象？
        #   class_labels torch.Size([1, 7, 19]) <class 'torch.Tensor'>
        #   translations torch.Size([1, 7, 3])  <class 'torch.Tensor'>
        #   sizes        torch.Size([1, 7, 3])  <class 'torch.Tensor'>
        #   angles       torch.Size([1, 7, 1])  <class 'torch.Tensor'>
        bbox_params_t = torch.cat([
            boxes["class_labels"],
            boxes["translations"],
            boxes["sizes"],
            boxes["angles"]
        ], dim=-1).cpu().numpy() # 拼接最后一阶，中间阶为啥增加了一维？
        # bbox_params_t: shape=(1, 8, 26) type=<class 'numpy.ndarray'>

        if True:
            print('tag1')
            save_lzc(bbox_params_t, objects_dataset, classes, args.output_directory)
            print('程序执行结束！')
            exit(0)
        renderables, trimesh_meshes = get_textured_objects( #这个函数的调用报错了
            bbox_params_t, objects_dataset, classes
            # bbox_params_t：通过网络得到的场景中每个物体的特征
            # objects_dataset：打包后所有物体的参数
            # classes：物体的类别标签
        )
        renderables += floor_plan
        trimesh_meshes += tr_floor

        if args.without_screen:
            # Do the rendering
            path_to_image = "{}/{}_{}_{:03d}".format(
                args.output_directory,
                current_scene.scene_id,
                scene_idx,
                i
            )
            behaviours = [
                LightToCamera(),
                SaveFrames(path_to_image+".png", 1)
            ]
            if args.with_rotating_camera:
                behaviours += [
                    CameraTrajectory(
                        Circle(
                            [0, args.camera_position[1], 0],
                            args.camera_position,
                            args.up_vector
                        ),
                        speed=1/360
                    ),
                    SaveGif(path_to_image+".gif", 1)
                ]

            render(
                renderables,
                behaviours=behaviours,
                size=args.window_size,
                camera_position=args.camera_position,
                camera_target=args.camera_target,
                up_vector=args.up_vector,
                background=args.background,
                n_frames=args.n_frames,
                scene=scene
            )
        else:
            show(
                renderables,
                behaviours=[LightToCamera(), SnapshotOnKey(), SortTriangles()],
                size=args.window_size,
                camera_position=args.camera_position,
                camera_target=args.camera_target,
                up_vector=args.up_vector,
                background=args.background,
                title="Generated Scene"
            )
        if trimesh_meshes is not None:
            # Create a trimesh scene and export it
            path_to_objs = os.path.join(
                args.output_directory,
                "{:03d}_scene".format(i)
            )
            if not os.path.exists(path_to_objs):
                os.mkdir(path_to_objs)
            export_scene(path_to_objs, trimesh_meshes)


if __name__ == "__main__":
    main(sys.argv[1:])
# python generate_scenes.py path_to_config_yaml path_to_output_dir path_to_3d_future_pickled_data path_to_floor_plan_texture_images --weight_file path_to_weight_file

# python generate_scenes.py
# ../config/bedrooms_config_lzc.yaml
# ../../Dataset/generate_out
# ../../Dataset/out-pickle/threed_future_model_bedroom.pkl
# ../../Dataset/3D-FRONT-texture
# --weight_file ../../Dataset/out-train/W1FYCHVEI/model_00000

# python generate_scenes.py ../config/bedrooms_config_lzc.yaml ../../Dataset/out-generate ../../Dataset/out-pickle/threed_future_model_bedroom.pkl ../../Dataset/3D-FRONT-texture --weight_file ../../Dataset/out-train/W1FYCHVEI/model_00000
# python generate_scenes.py ../config/bedrooms_config_lzc3.yaml ../../Dataset/out-generate ../../Dataset/out-pickle/threed_future_model_bedroom.pkl ../../Dataset/3D-FRONT-texture --weight_file ../../Dataset/out-train/model_00050

# export PATH="~/anaconda3/bin:$PATH"
