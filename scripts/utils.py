# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

import os

import numpy as np
import torch
from PIL import Image
from pyrr import Matrix44

import trimesh

from simple_3dviz import Mesh, Scene
from simple_3dviz.renderables.textured_mesh import Material, TexturedMesh
from simple_3dviz.utils import save_frame
from simple_3dviz.behaviours.misc import LightToCamera
from simple_3dviz.behaviours.io import SaveFrames
from simple_3dviz.utils import render as render_simple_3dviz

from scene_synthesis.utils import get_textured_objects

import cv2

class DirLock(object):
    def __init__(self, dirpath):
        self._dirpath = dirpath
        self._acquired = False

    @property
    def is_acquired(self):
        return self._acquired

    def acquire(self):
        if self._acquired:
            return
        try:
            os.mkdir(self._dirpath)
            self._acquired = True
        except FileExistsError:
            pass

    def release(self):
        if not self._acquired:
            return
        try:
            os.rmdir(self._dirpath)
            self._acquired = False
        except FileNotFoundError:
            self._acquired = False
        except OSError:
            pass

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()


def ensure_parent_directory_exists(filepath):
    os.makedirs(filepath, exist_ok=True)


def floor_plan_renderable(room, color=(1.0, 1.0, 1.0, 1.0)):
    vertices, faces = room.floor_plan
    # Center the floor
    vertices -= room.floor_plan_centroid
    # Return a simple-3dviz renderable
    return Mesh.from_faces(vertices, faces, color)


def floor_plan_from_scene(
    scene,                       # <scene_synthesis.datasets.threed_front_scene.Room>
    path_to_floor_plan_textures, # '../../Dataset/3D-FRONT-texture'
    without_room_mask=False      # without_room_mask=True
):
    if not without_room_mask:
        room_mask = torch.from_numpy(
            np.transpose(scene.room_mask[None, :, :, 0:1], (0, 3, 1, 2))
        )
    else:
        room_mask = None
    # room_mask = None
    # Also get a renderable for the floor plan # 同时获取平面图的渲染图
    # print("flag.0")
    floor, tr_floor = get_floor_plan(
        scene, # scene Scene: MasterBedroom-9694 of type: masterbedroom contains 6 bboxes
        [# length=1427, item0=../../Dataset/3D-FRONT-texture/2439602f-174d-444d-a6a5-f4a1181f88b6
            os.path.join(path_to_floor_plan_textures, fi)
            for fi in os.listdir(path_to_floor_plan_textures)
        ]
    )
    # print("flag.1")
    return [floor], [tr_floor], room_mask


def get_floor_plan(scene, floor_textures):
    """Return the floor plan of the scene as a trimesh mesh and a simple-3dviz
    TexturedMesh."""
    # scene: Scene: Bedroom-9787 of type: bedroom contains 5 bboxes
    vertices, faces = scene.floor_plan
    # vertices.shape (9, 3)
    # faces.shape(3, 3)
    vertices = vertices - scene.floor_plan_centroid # centroid: [-2.36505  0.  3.6434 ]
    uv = np.copy(vertices[:, [0, 2]]) #沿y轴进行投影
    # uv.shape (9, 2)
    uv -= uv.min(axis=0) # uv.min(axis=0)=[-1.4652 -2.1933]
    uv /= 0.3  # repeat every 30cm
    # floor_textures: 1427 ../../Dataset/3D-FRONT-texture/2439602f-174d-444d-a6a5-f4a1181f88b6
    texture = np.random.choice(floor_textures)
    # texture: ../../Dataset/3D-FRONT-texture/55d18b9c-47c1-46ce-97a3-2b2d1527afad

    #print("material:",Material.with_texture_image(texture))
    # if True:
    #     print("texture:",texture)
    #     img = cv2.imread(texture+"/texture.png", 1)
    #     print("img",type(img),img.shape)
    # print("texture:",texture)
    floor = TexturedMesh.from_faces(
        vertices=vertices,
        uv=uv,
        faces=faces,
        material=Material.with_texture_image(texture+"/texture.png") # Material.with_texture_image(texture)
    )

    tr_floor = trimesh.Trimesh(
        np.copy(vertices), np.copy(faces), process=False
    )
    tr_floor.visual = trimesh.visual.TextureVisuals(
        uv=np.copy(uv),
        material=trimesh.visual.material.SimpleMaterial(
            image=Image.open(texture+"/texture.png") # Image.open(texture)
        )
    )

    return floor, tr_floor


def get_textured_objects_in_scene(scene, ignore_lamps=False):
    renderables = []
    for furniture in scene.bboxes: # 遍历每一个家具
        model_path = furniture.raw_model_path
        # model_path: ../../Dataset/3D-FUTURE-model/36a8ee08-14c3-4380-ae0e-1a66525aa7f5/raw_model.obj
        if not model_path.endswith("obj"): # 这里是False、不会被执行
            import pdb
            pdb.set_trace()

        # Load the furniture and scale it as it is given in the dataset # 加载家具并按照数据集中给出的比例进行缩放
        raw_mesh = TexturedMesh.from_file(model_path)
        # raw_mesh: <simple_3dviz.renderables.textured_mesh.TexturedMesh object at 0x7688f27c3430>
        raw_mesh.scale(furniture.scale)
        # furniture.scale: [1, 1, 1]

        # Compute the centroid of the vertices in order to match the bbox (because the prediction only considers bboxes)
        # 计算顶点的质心以匹配bbox（因为预测只考虑bbox）
        bbox = raw_mesh.bbox
        # bbox: [
        #   [-0.187537,  0.      , -0.187537],
        #   [ 0.187537,  0.529763,  0.187537]   ]
        centroid = (bbox[0] + bbox[1])/2
        # centroid: [0., 0.2648815, 0.]

        # Extract the predicted affine transformation to position the mesh
        # 提取预测的仿射变换以定位网格
        translation = furniture.centroid(offset=-scene.centroid)
        # translation: [ 0.3306,2.61854499,-0.0488 ]
        theta = furniture.z_angle
        # theta: 0
        R = np.zeros((3, 3))
        R[0, 0] = +np.cos(theta)
        R[0, 2] = -np.sin(theta)
        R[2, 0] = +np.sin(theta)
        R[2, 2] = +np.cos(theta)
        R[1, 1] = +1.
        # R: [[ 1.  0. -0.]
        #     [ 0.  1.  0.]
        #     [ 0.  0.  1.]]

        # Apply the transformations in order to correctly position the mesh
        # 应用变换以正确定位网格
        raw_mesh.affine_transform(t=-centroid)#这里的t应该是移动，而R是旋转
        raw_mesh.affine_transform(R=R, t=translation)
        renderables.append(raw_mesh)
        # print()
    return renderables


def render(scene, renderables, color, mode, frame_path=None):
    if color is not None:
        try:
            color[0][0]
        except TypeError:
            color = [color]*len(renderables)
    else:
        color = [None]*len(renderables)

    scene.clear()
    # print("scene flag")
    for r, c in zip(renderables, color):
        if isinstance(r, Mesh) and c is not None:
            r.mode = mode
            r.colors = c
        scene.add(r)
    scene.render()
    if frame_path is not None:
        save_frame(frame_path, scene.frame)

    return np.copy(scene.frame)


def scene_from_args(args):
    # Create the scene and the behaviour list for simple-3dviz
    scene = Scene(size=args.window_size, background=args.background)
    scene.up_vector = args.up_vector
    scene.camera_target = args.camera_target
    scene.camera_position = args.camera_position
    scene.light = args.camera_position
    scene.camera_matrix = Matrix44.orthogonal_projection(
        left=-args.room_side, right=args.room_side,
        bottom=args.room_side, top=-args.room_side,
        near=0.1, far=6
    )
    return scene


def export_scene(output_directory, trimesh_meshes, names=None):
    if names is None:
        names = [
            "object_{:03d}.obj".format(i) for i in range(len(trimesh_meshes))
        ]
    mtl_names = [
        "material_{:03d}".format(i) for i in range(len(trimesh_meshes))
    ]

    for i, m in enumerate(trimesh_meshes):
        obj_out, tex_out = trimesh.exchange.obj.export_obj(
            m,
            return_texture=True
        )

        with open(os.path.join(output_directory, names[i]), "w") as f:
            f.write(obj_out.replace("material0", mtl_names[i]))

        # No material and texture to rename
        if tex_out is None:
            continue

        mtl_key = next(k for k in tex_out.keys() if k.endswith(".mtl"))
        path_to_mtl_file = os.path.join(output_directory, mtl_names[i]+".mtl")
        with open(path_to_mtl_file, "wb") as f:
            f.write(
                tex_out[mtl_key].replace(
                    b"material0", mtl_names[i].encode("ascii")
                )
            )
        tex_key = next(k for k in tex_out.keys() if not k.endswith(".mtl"))
        tex_ext = os.path.splitext(tex_key)[1]
        path_to_tex_file = os.path.join(output_directory, mtl_names[i]+tex_ext)
        with open(path_to_tex_file, "wb") as f:
            f.write(tex_out[tex_key])


def print_predicted_labels(dataset, boxes):
    object_types = np.array(dataset.object_types)
    box_id = boxes["class_labels"][0, 1:-1].argmax(-1)
    labels = object_types[box_id.cpu().numpy()].tolist()
    print("The predicted scene contains {}".format(labels))


def poll_specific_class(dataset):
    label = input(
        "Select an object class from {}\n".format(dataset.object_types)
    )
    if label in dataset.object_types:
        return dataset.object_types.index(label)
    else:
        return None


def make_network_input(current_boxes, indices):
    def _prepare(x):
        return torch.from_numpy(x[None].astype(np.float32))

    return dict(
        class_labels=_prepare(current_boxes["class_labels"][indices]),
        translations=_prepare(current_boxes["translations"][indices]),
        sizes=_prepare(current_boxes["sizes"][indices]),
        angles=_prepare(current_boxes["angles"][indices])
    )


def render_to_folder(
    args,
    folder,
    dataset,
    objects_dataset,
    tr_floor,
    floor_plan,
    scene,
    bbox_params,
    add_start_end=False
):
    boxes = dataset.post_process(bbox_params)
    bbox_params_t = torch.cat(
        [
            boxes["class_labels"],
            boxes["translations"],
            boxes["sizes"],
            boxes["angles"]
        ],
        dim=-1
    ).cpu()

    if add_start_end:
        bbox_params_t = torch.cat([
            torch.zeros(1, 1, bbox_params_t.shape[2]),
            bbox_params_t,
            torch.zeros(1, 1, bbox_params_t.shape[2]),
        ], dim=1)

    renderables, trimesh_meshes = get_textured_objects(
        bbox_params_t.numpy(), objects_dataset, np.array(dataset.class_labels)
    )
    trimesh_meshes += tr_floor

    path_to_objs = os.path.join(args.output_directory, folder)
    if not os.path.exists(path_to_objs):
        os.mkdir(path_to_objs)
    export_scene(path_to_objs, trimesh_meshes)

    path_to_image = os.path.join(
        args.output_directory,
        folder + "_render.png"
    )
    behaviours = [
        LightToCamera(),
        SaveFrames(path_to_image, 1)
    ]
    render_simple_3dviz(
        renderables + floor_plan,
        behaviours=behaviours,
        size=args.window_size,
        camera_position=args.camera_position,
        camera_target=args.camera_target,
        up_vector=args.up_vector,
        background=args.background,
        n_frames=args.n_frames,
        scene=scene
    )


def render_scene_from_bbox_params(
    args,
    bbox_params,
    dataset,
    objects_dataset,
    classes,
    floor_plan,
    tr_floor,
    scene,
    path_to_image,
    path_to_objs
):
    boxes = dataset.post_process(bbox_params)
    print_predicted_labels(dataset, boxes)
    bbox_params_t = torch.cat(
        [
            boxes["class_labels"],
            boxes["translations"],
            boxes["sizes"],
            boxes["angles"]
        ],
        dim=-1
    ).cpu().numpy()

    renderables, trimesh_meshes = get_textured_objects(
        bbox_params_t, objects_dataset, classes
    )
    renderables += floor_plan
    trimesh_meshes += tr_floor

    # Do the rendering
    behaviours = [
        LightToCamera(),
        SaveFrames(path_to_image+".png", 1)
    ]
    render_simple_3dviz(
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
    if trimesh_meshes is not None:
        # Create a trimesh scene and export it
        if not os.path.exists(path_to_objs):
            os.mkdir(path_to_objs)
        export_scene(path_to_objs, trimesh_meshes)
