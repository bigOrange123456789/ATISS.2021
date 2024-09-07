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
    # args = parser.parse_args(argv)
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

    objects_dataset = ThreedFutureDataset(objects)
    room_type = args0.dataset_filtering.split("_")[-1]
    output_path = "{}/threed_future_model_{}.pkl".format(
        args0.output_directory,
        room_type
    )
    with open(output_path, "wb") as f:
        pickle.dump(objects_dataset, f)


if __name__ == "__main__":
    lzc_pickle()
# python pickle_threed_future_dataset2.py ../../Dataset/out-pickle ../../Dataset/3D-FRONT-TEST ../../Dataset/3D-FUTURE-model ../../Dataset/3D-FUTURE-model/model_info.json --dataset_filtering threed_front_bedroom
