import torch
import random
import json
import re
import numpy as np
from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset
from copy import deepcopy
from torch.utils.data import Dataset
from unitraj.datasets.base_dataset import BaseDataset
from omegaconf import OmegaConf

#nuscenes_data_start_index의 dict경로
train_dict_path = "/home/chan/Unimm/data/nuscenes/nuscenes_train_start_index_dict.json"
val_dict_path = "/home/chan/Unimm/data/nuscenes/nuscenes_val_start_index_dict.json"


class CombinedDataset(Dataset):
    def __init__(self, config=None, is_validation=False):
        self.is_validation = is_validation
        self.config = deepcopy(config)
        self.sensor_config = OmegaConf.to_container(self.config.SENSOR_DATASET, resolve=True)
        self.traj_dataset = BaseDataset(config.TRAJ_DATASET, is_validation)
        self.sensor_dataset = NuScenesDataset(**self.sensor_config)
        self.data_chunk_size = 1
        self.train_dict_path = train_dict_path
        self.val_dict_path = val_dict_path
        self.fps = self.config.TRAJ_DATASET.fps
        self.skip = self.config.TRAJ_DATASET.skip
        
        
    def __len__(self):
        return self.traj_dataset.__len__()


    def __getitem__(self, idx):
        traj_data = self.traj_dataset.__getitem__(idx)
        traj_scene_token, traj_curr_index = self.traj2sensor(traj_data)
        
        if  self.is_validation == False:
            file_path = self.train_dict_path
        else:
            file_path = self.val_dict_path
        nuscenes_start_index_dict = self.load_json_file(file_path)
        
        scene_start_index = nuscenes_start_index_dict[traj_scene_token]
        sensor_curr_index = scene_start_index + traj_curr_index 
        sensor_data = self.sensor_dataset.__getitem__(sensor_curr_index)
        post_sensor_idx = sensor_data['data_samples'].sample_idx
        
        if sensor_curr_index == post_sensor_idx:
            traj_data = traj_data
            sensor_data = sensor_data
            self.check_timestamp(traj_data, sensor_data)
            
        else:
            post_traj_idx = self._rand_another()
            traj_data, sensor_data = self.__getitem__(post_traj_idx)
            self.check_timestamp(traj_data, sensor_data)

        return traj_data, sensor_data


    def traj2sensor(self, trajdataset):
        traj = trajdataset
        scene_token = traj[0]["scene_token"]
        start_index = traj[0]["start_index"]
        current_time_index = traj[0]["current_time_index"]
        traj_curr_index = start_index + current_time_index * int(self.fps / self.skip)
        return scene_token, traj_curr_index
    
    
    def _rand_another(self) -> int:
        num = list(range(len(self)))
        choosen_list = random.sample(num, 1)
        return choosen_list[0]
    
    
    def check_timestamp(self, traj_data, sensor_data):
        traj_timestamp = traj_data[0]["curr_timestamp_ns"].item()
        lidar_path = sensor_data['data_samples'].lidar_path
        sensor_timestamp = int(re.search(r'\d+', lidar_path.split('__')[-1]).group())
        assert traj_timestamp == sensor_timestamp, "Mismatches of timestamp between traj_data and sensor_data"
        
    
    def load_json_file(self, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
        
            
    def collate_fn(self, data_list):
        # batch_list = []

        # # Collate for traj_data
        # for batch in data_list:
        #     batch_list += batch[0]
            
        traj_batch_list = [traj_batch[0][0] for traj_batch in data_list]

        batch_size = len(traj_batch_list)
        key_to_list = {key: [traj_batch_list[bs_idx][key] for bs_idx in range(batch_size)] 
                       for key in traj_batch_list[0].keys()}

        traj_input_dict = {}
        for key, val_list in key_to_list.items():
            try:
                traj_input_dict[key] = torch.from_numpy(np.stack(val_list, axis=0))
            except Exception:
                traj_input_dict[key] = val_list

        traj_input_dict['center_objects_type'] = traj_input_dict['center_objects_type'].numpy()

        traj_batch_dict = {'batch_size': batch_size, 'input_dict': traj_input_dict, 'batch_sample_count': batch_size}

        # Collate for sensor_data
        sensor_batch_list = [sensor_batch[1] for sensor_batch in data_list]

        data_samples_list = [sensor_data["data_samples"] for sensor_data in sensor_batch_list]
        points_list = [sensor_data['inputs']["points"] for sensor_data in sensor_batch_list]
        img_list = [sensor_data['inputs']["img"] for sensor_data in sensor_batch_list]

        batched_img = torch.stack(img_list, dim=0)

        batch_input_dict = {"points": points_list, "imgs": batched_img}
        sensor_batch_dict = {'data_samples': data_samples_list, 'batch_input_dict': batch_input_dict}
        batch_traj_sensor = {'traj_data': traj_batch_dict, 'sensor_data': sensor_batch_dict}

        return batch_traj_sensor



    
    
    
