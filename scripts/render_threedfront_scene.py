# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

"""Script used for visualizing 3D-FRONT room specified by its scene_id."""
import argparse
import logging
import os
import sys

import numpy as np
from PIL import Image
import pyrr
import trimesh

from scene_synthesis.datasets.threed_front import ThreedFront

from simple_3dviz import Scene
from simple_3dviz.behaviours.keyboard import SnapshotOnKey
from simple_3dviz.behaviours.io import SaveFrames
from simple_3dviz.renderables.textured_mesh import TexturedMesh
from simple_3dviz.utils import render
# from simple_3dviz.window import show

#from utils import floor_plan_from_scene, export_scene
############################
import torch
from pyrr import Matrix44

from simple_3dviz import Mesh, Scene
from simple_3dviz.renderables.textured_mesh import Material, TexturedMesh
from simple_3dviz.behaviours.misc import LightToCamera
from simple_3dviz.behaviours.io import SaveFrames
from simple_3dviz.utils import render as render_simple_3dviz

from scene_synthesis.utils import get_textured_objects
############################

def scene_init(mesh, up_vector, camera_position, camera_target, background):
    def inner(scene):
        scene.background = background
        scene.up_vector = up_vector
        scene.camera_position = camera_position
        scene.camera_target = camera_target
        scene.light = camera_position
        if mesh is not None:
            scene.add(mesh)
    return inner

def saveJson(data,path):
    print(data)
    print(path)
    import json
    a = {
        "name": "dabao",
        "id": 123,
        "hobby": {
            "sport": "basketball",
            "book": "python study"
        }
    }
    a = data
    b = json.dumps(a)
    f2 = open( path , 'w') # f2 = open('new_json.json', 'w')
    f2.write(b)
    f2.close()

def main(argv):
    parser = argparse.ArgumentParser(
        description="Visualize a 3D-FRONT room from json file"
    )
    parser.add_argument(
        "scene_id",
        help="The scene id of the scene to be visualized"
    )
    parser.add_argument(
        "output_directory",
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
        "--annotation_file",
        default="../config/bedroom_threed_front_splits.csv",
        help="Path to the train/test splits file"
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
        default="0,0,1",
        help="Up vector of the scene"
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
        default="-2.0,-2.0,-2.0",
        help="Camer position in the scene"
    )
    parser.add_argument(
        "--window_size",
        type=lambda x: tuple(map(int, x.split(","))),
        default="512,512",
        help="Define the size of the scene and the window"
    )
    parser.add_argument(
        "--save_frames",
        help="Path to save the visualization frames to"
    )
    parser.add_argument(
        "--without_screen",
        action="store_true",
        help="Perform no screen rendering"
    )
    parser.add_argument(
        "--with_orthographic_projection",
        action="store_true",
        help="Use orthographic projection"
    )
    parser.add_argument(
        "--with_floor_layout",
        action="store_true",
        help="Visualize also the rooom's floor"
    )
    parser.add_argument(
        "--with_walls",
        action="store_true",
        help="Visualize also the rooom's floor"
    )
    parser.add_argument(
        "--with_door_and_windows",
        action="store_true",
        help="Visualize also the rooom's floor"
    )
    parser.add_argument(
        "--with_texture",
        action="store_true",
        help="Visualize objects with texture"
    )

    args = parser.parse_args(argv)
    logging.getLogger("trimesh").setLevel(logging.ERROR)

    # Check if output directory exists and if it doesn't create it
    # 检查输出目录是否存在，以及是否没有创建它
    # 输出目录: out
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    # Create the scene and the behaviour list for simple-3dviz
    # 为simple-3dviz创建场景和行为列表
    # window_size: (512, 512)
    scene = Scene(size=args.window_size)
    # with_orthographic_projection: False
    if args.with_orthographic_projection: # 不使用正交投影
        scene.camera_matrix = pyrr.Matrix44.orthogonal_projection(
            left=-3.1, right=3.1, bottom=-3.1, top=3.1, near=0.1, far=1000
        )
    scene.light = args.camera_position # camera_position: (-2.0, -2.0, -2.0)
    behaviours = []

    # Loading dataset  6812 /  6813
    d = ThreedFront.from_dataset_directory( # 这里的执行速度非常慢
        args.path_to_3d_front_dataset_directory,  # 3D-FRONT
        args.path_to_model_info,                  # 3D-FUTURE-model/model_info.json
        args.path_to_3d_future_dataset_directory, # 3D-FUTURE-model
        path_to_room_masks_dir=None,
        path_to_bounds=None,
        filter_fn=lambda s: s
    ) # len(d)=17129
    # 场景文件数量6813
    # 贴图文件数量1425
    # 模型文件数量16563
    print("Loading dataset with {} rooms".format(len(d)))
    # s.scene_id: MasterBedroom-4004
    # s.scene_id: Library-1946
    # s.scene_id: LivingDiningRoom-3345
    # s.scene_id: SecondBedroom-3031
    # s.scene_id: Library-1246
    # s.scene_id: LivingDiningRoom-5823
    # s.scene_id: LivingDiningRoom-5414
    # s.scene_id: MasterBedroom-38249
    # s.scene_id: KidsRoom-38953
    # s.scene_id: MasterBedroom-3213
    # s.scene_id: LivingDiningRoom-21823
    # s.scene_id: BedRoom-282737
    # s.scene_id: MasterBedroom-11453
    # s.scene_id: BedRoom-377735
    # s.scene_id: BedRoom-296370
    # s.scene_id: DiningRoom-10543
    # s.scene_id: Bedroom-10581
    # s.scene_id: LivingRoom-10632
    # s.scene_id: Auditorium-10613
    # s.scene_id: LivingDiningRoom-46892
    # s.scene_id: Bedroom-46819
    # s.scene_id: Library-46293
    # s.scene_id: LivingDiningRoom-5987
    # s.scene_id: LivingDiningRoom-5310
    # s.scene_id: Bedroom-5400
    # s.scene_id: Bedroom-1316
    # s.scene_id: LivingRoom-10060
    # s.scene_id: DiningRoom-196107
    # s.scene_id: OtherRoom-184116
    # s.scene_id: SecondBedroom-14246
    # s.scene_id: LivingDiningRoom-16531
    # s.scene_id: CloakRoom-16138
    # s.scene_id: LivingDiningRoom-4513
    # s.scene_id: KidsRoom-2308

    for s in d.scenes:
        print(args.scene_id,s.scene_id)
        if s.scene_id == args.scene_id: # MasterBedroom-28057
            my_scene={
                'id':s.scene_id,
                'centroid':s.centroid.tolist(),
                'meshes':[]
            }
            my_meshes=[]
            # s.bboxes: [<scene_synthesis.datasets.threed_front_scene.ThreedFutureModel>, ...] # len=7
            for b in s.bboxes: # ['king-size bed', 'wardrobe', 'wardrobe', 'tv stand', 'nightstand', 'nightstand', 'pendant lamp']
                print(b.model_jid, b.label)
            print(s.furniture_in_room, s.scene_id, s.json_path)
            # s.json_path:5c8e6bc5-c29e-4ac3-813e-df2145b20c69
            renderables = s.furniture_renderables(
                with_floor_plan_offset=True, with_texture=args.with_texture # with_texture: False
            )
            trimesh_meshes = []
            for furniture in s.bboxes:
                my_scene['meshes'].append({
                    'raw_model_path':furniture.raw_model_path,
                    'texture_image_path':furniture.texture_image_path,
                    'scale':furniture.scale,
                    'z_angle':furniture.z_angle,
                    'position':furniture.position
                })
                print("furniture:",furniture)
                print("furniture.raw_model_path:",furniture.raw_model_path)
                # for i in dir(furniture):
                #     print("i:",i)
                # Load the furniture and scale it as it is given in the dataset
                # 加载家具并按照数据集中给出的比例进行缩放
                # raw_mesh = TexturedMesh.from_file(furniture.raw_model_path)
                try:
                    raw_mesh = TexturedMesh.from_file(furniture.raw_model_path)
                    # furniture: <scene_synthesis.datasets.threed_front_scene.ThreedFutureModel>
                    # furniture.raw_model_path: 3D-FUTURE-model/858870cd-37da-4834-9cc4-fe6ab812bb97/raw_model.obj
                except:
                    try:
                        texture_path = furniture.texture_image_path
                        mesh_info = read_mesh_file(furniture.raw_model_path)
                        vertices = mesh_info.vertices
                        normals = mesh_info.normals
                        uv = mesh_info.uv
                        material = Material.with_texture_image(texture_path)
                        raw_mesh = TexturedMesh(vertices, normals, uv, material)
                    except:
                        print("Failed loading texture info.")
                        raw_mesh = Mesh.from_file(furniture.raw_model_path)
                # print("raw_mesh:",raw_mesh)
                raw_mesh.scale(furniture.scale)

                # Create a trimesh object for the same mesh in order to save
                # everything as a single scene
                # 为同一网格创建一个trimesh对象，以保存单个场景
                # print("furniture:",furniture)
                tr_mesh = trimesh.load(furniture.raw_model_path, force="mesh")
                # print("raw_model:",furniture.raw_model_path)
                tr_mesh.visual.material.image = Image.open(
                    furniture.texture_image_path
                )
                # print("texture:",furniture.texture_image_path)
                # print("scale:",furniture.scale)
                tr_mesh.vertices *= furniture.scale
                # print("furniture.z_angle:",furniture.z_angle)
                theta = furniture.z_angle
                R = np.zeros((3, 3))
                R[0, 0] = np.cos(theta)
                R[0, 2] = -np.sin(theta)
                R[2, 0] = np.sin(theta)
                R[2, 2] = np.cos(theta)
                R[1, 1] = 1.
                # print("furniture.position:",furniture.position)
                tr_mesh.vertices[...] = \
                    tr_mesh.vertices.dot(R) + furniture.position
                tr_mesh.vertices[...] = tr_mesh.vertices - s.centroid # centroid是场景的中心
                # print("s.centroid:",s.centroid)
                trimesh_meshes.append(tr_mesh)

            if args.with_floor_layout:
                # Get a floor plan
                floor_plan, tr_floor, _ = floor_plan_from_scene(
                    s, args.path_to_floor_plan_textures, without_room_mask=True
                )
                renderables += floor_plan
                trimesh_meshes += tr_floor
            saveJson(my_scene,my_scene['id']+".json")

            if args.with_walls:
                for ei in s.extras:
                    if "WallInner" in ei.model_type:
                        renderables = renderables + [
                            ei.mesh_renderable(
                                offset=-s.centroid,
                                colors=(0.8, 0.8, 0.8, 0.6)
                            )
                        ]

            if args.with_door_and_windows:
                for ei in s.extras:
                    if "Window" in ei.model_type or "Door" in ei.model_type:
                        renderables = renderables + [
                            ei.mesh_renderable(
                                offset=-s.centroid,
                                colors=(0.8, 0.8, 0.8, 0.6)
                            )
                        ]

            if True:#args.without_screen:
                path_to_image = "{}/{}_".format(args.output_directory, s.uid)
                # path_to_image: out/00ad8345-45e0-45b3-867d-4a3c88c2517a_LivingRoom-46201_
                behaviours += [SaveFrames(path_to_image+"{:03d}.png", 1)]
                # behaviours: [<~.SaveFrames>]
                render(
                    renderables,
                    # renderables: [
                    # <Mesh>, <Mesh>, <Mesh>, <Mesh>, <Mesh>,
                    # <Mesh>, <Mesh>, <Mesh>, <Mesh>, <Mesh>,
                    # <Mesh>, <Mesh>, <Mesh>, <Mesh>, <Mesh>,
                    # <Mesh>, <Mesh>, <Mesh>]
                    size=args.window_size,      # size: (512, 512)
                    camera_position=args.camera_position, # pos: (-2.0, -2.0, -2.0)
                    camera_target=args.camera_target, # target: (0.0, 0.0, 0.0)
                    up_vector=args.up_vector,   # up: (0.0, 0.0, 1.0)
                    background=args.background, # background: [1.0, 1.0, 1.0, 1.0]
                    behaviours=behaviours,      # behaviours: [<~.SaveFrames>]
                    n_frames=2,
                    scene=scene
                )
                print('程序结束！--LZC')#saveJson({"a":1}, "test.json")
                exit(0)
            else:
                show(
                    renderables,
                    behaviours=behaviours+[SnapshotOnKey()],
                    size=args.window_size,
                    camera_position=args.camera_position,
                    camera_target=args.camera_target,
                    light=args.camera_position,
                    up_vector=args.up_vector,
                    background=args.background,
                )
            # Create a trimesh scene and export it
            path_to_objs = os.path.join(
                args.output_directory,
                "train_{}".format(args.scene_id)
            )
            if not os.path.exists(path_to_objs):
                os.mkdir(path_to_objs)
            export_scene(path_to_objs, trimesh_meshes)
            '''
                path_to_objs: 
                    out/train_LivingRoom-46201
                trimesh_meshes: 
                    [
                        <trimesh.Trimesh(vertices.shape=(34596, 3), faces.shape=(29320, 3), name=`raw_model.obj`)>, 
                        <trimesh.Trimesh(vertices.shape=(8553, 3),  faces.shape=(9724, 3),  name=`raw_model.obj`)>, 
                        <trimesh.Trimesh(vertices.shape=(9246, 3),  faces.shape=(10192, 3), name=`raw_model.obj`)>, 
                        <trimesh.Trimesh(vertices.shape=(9246, 3),  faces.shape=(10192, 3), name=`raw_model.obj`)>, 
                        <trimesh.Trimesh(vertices.shape=(8019, 3),  faces.shape=(4993, 3),  name=`raw_model.obj`)>, 
                        <trimesh.Trimesh(vertices.shape=(5091, 3),  faces.shape=(4952, 3),  name=`raw_model.obj`)>, 
                        <trimesh.Trimesh(vertices.shape=(6334, 3),  faces.shape=(5332, 3),  name=`raw_model.obj`)>, 
                        <trimesh.Trimesh(vertices.shape=(5091, 3),  faces.shape=(4952, 3),  name=`raw_model.obj`)>, 
                        <trimesh.Trimesh(vertices.shape=(5091, 3),  faces.shape=(4952, 3),  name=`raw_model.obj`)>, 
                        <trimesh.Trimesh(vertices.shape=(5091, 3),  faces.shape=(4952, 3),  name=`raw_model.obj`)>, 
                        <trimesh.Trimesh(vertices.shape=(1421, 3),  faces.shape=(1106, 3),  name=`raw_model.obj`)>, 
                        <trimesh.Trimesh(vertices.shape=(14419, 3), faces.shape=(13583, 3), name=`raw_model.obj`)>, 
                        <trimesh.Trimesh(vertices.shape=(12337, 3), faces.shape=(8610, 3),  name=`raw_model.obj`)>, 
                        <trimesh.Trimesh(vertices.shape=(35656, 3), faces.shape=(27897, 3), name=`raw_model.obj`)>, 
                        <trimesh.Trimesh(vertices.shape=(1222, 3),  faces.shape=(1409, 3),  name=`raw_model.obj`)>, 
                        <trimesh.Trimesh(vertices.shape=(3175, 3),  faces.shape=(3987, 3),  name=`raw_model.obj`)>, 
                        <trimesh.Trimesh(vertices.shape=(10939, 3), faces.shape=(6555, 3),  name=`raw_model.obj`)>, 
                        <trimesh.Trimesh(vertices.shape=(35656, 3), faces.shape=(27897, 3), name=`raw_model.obj`)>
                    ]
            '''



if __name__ == "__main__":
    main(sys.argv[1:])
# python render_threedfront_scene.py LivingRoom-46201 ../../Dataset/out-render ../../Dataset/3D-FRONT ../../Dataset/3D-FUTURE-model ../../Dataset/3D-FUTURE-model/model_info.json ../../Dataset/3D-FRONT-texture
