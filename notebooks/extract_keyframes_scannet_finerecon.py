import numpy as np
from path import Path
from tqdm import tqdm

import sys
sys.path.insert(0, '/home/ruizhu/Documents/Projects/ml-finerecon/third-party/deep-video-mvs')

from dvmvs.config import Config
from dvmvs.keyframe_buffer import KeyframeBuffer

import json

SCANNET_ROOT = Path('/newfoundland/ScanNet/extracted')
SCANNET_LISTS = Path('/home/ruizhu/Documents/Projects/ScanNet/Tasks/Benchmark/')
json_data = {}

for split in ['train', 'val', 'test']:
# for split in ['train']:
    keyframe_json_path = Path('/newfoundland/ScanNet/extracted/%s_keyframes.json'%split)
    
    scannet_list_path = SCANNET_LISTS / f'scannetv2_{split}.txt'
    assert scannet_list_path.exists(), f'{scannet_list_path} does not exist'
    with open(scannet_list_path) as f:
        scene_names = [line.strip() for line in f.readlines()]
        
    for scene_name in tqdm(scene_names):
        scene_folder = SCANNET_ROOT / scene_name
        assert scene_folder.exists(), f'{scene_folder} does not exist'
        # poses = np.fromfile(scene_folder / "poses.txt", dtype=float, sep="\n ").reshape((-1, 4, 4))
        poses = np.load(scene_folder / "pose.npy")
        assert len(poses.shape)==3 and poses.shape[1:] == (4, 4), f'poses.shape: {poses.shape}'
        # image_filenames = sorted((scene_folder / 'images').files("*.png"))
        # image_filenames = sorted((scene_folder / 'color').glob("*.jpg"))
        # assert len(poses) == len(image_filenames), f'len(poses): {len(poses)}, len(image_filenames): {len(image_filenames)}'
        image_filenames = [Path(scene_folder / 'color' / f'{i:d}.jpg') for i in range(len(poses))]
        for image_filename in image_filenames:
            assert image_filename.exists(), f'{image_filename} does not exist'
        
        keyframe_buffer = KeyframeBuffer(buffer_size=Config.test_keyframe_buffer_size,
                                    keyframe_pose_distance=Config.test_keyframe_pose_distance,
                                    optimal_t_score=Config.test_optimal_t_measure,
                                    optimal_R_score=Config.test_optimal_R_measure,
                                    store_return_indices=True)

        output_lines = []
        keyframe_id_list = []
        
        for i in range(0, len(poses)):
            reference_pose = poses[i]
            # print(i, reference_pose)
            pose_ = np.fromfile(scene_folder / 'pose' / ("%d.txt"%i), dtype=float, sep="\n ").reshape((4, 4))
            if np.any(np.isinf(pose_)):
                continue
            assert np.allclose(reference_pose, pose_), f'pose_[{i}]: {pose_}, reference_pose: {reference_pose}'

            # POLL THE KEYFRAME BUFFER
            response = keyframe_buffer.try_new_keyframe(reference_pose, None, index=i)
            if response == 3:
                output_lines.append("TRACKING LOST")
            elif response == 1:
                measurement_frames = keyframe_buffer.get_best_measurement_frames(Config.test_n_measurement_frames)
                # print(i, len(poses), len(measurement_frames[0]), measurement_frames[0])

                # output_line = image_filenames[i].split("/")[-1]
                output_line = image_filenames[i].name

                for (measurement_pose, measurement_image, measurement_index) in measurement_frames:
                    output_line += (" " + image_filenames[measurement_index].split("/")[-1])
                    # print(i, measurement_index, output_line)

                output_line = output_line.strip(" ")
                output_lines.append(output_line)
                
                keyframe_id_list.append(i)
                
        # output_lines = np.array(output_lines)
        # dataset_name = test_dataset_path.split("/")[-1]
        # np.savetxt('{}/keyframe+{}+{}+nmeas+{}'.format(output_folder, dataset_name, scene, Config.test_n_measurement_frames), output_lines, fmt='%s')
        
        json_data[scene_name] = ['i%d'%_ for _ in keyframe_id_list]
        
    with open(str(keyframe_json_path), 'w') as f:
        json.dump(json_data, f, indent=4)
        
    print('Saved to %s'%keyframe_json_path)


