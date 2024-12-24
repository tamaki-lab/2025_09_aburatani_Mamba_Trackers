from torch.utils.data import Dataset

import os 
import os.path as osp
import numpy as np 
import json
import torch 

from loguru import logger

"""
segmented trajectory

"""

class TrajDataset(Dataset):

    def __init__(self, cfgs) -> None:
        super().__init__()

        self.cfgs = cfgs
        self.window_size = self.cfgs['window_size']
        self.skip_sample = self.cfgs['skip_sample']

        with open(cfgs['anno_path'], 'r') as f:
            self.anno_file = json.load(f, )  # get the annotation file

        self.datasets = list(self.anno_file.keys())
        self.num_datasets = len(self.datasets)

        self.num_all_objs = 0
        for dataset in self.datasets:
            self.num_all_objs += self.anno_file[dataset]['total_objs']

        # print info
        logger.info(f'Total datasets: {self.datasets}')
        logger.info(f'Total obj numbers: {self.num_all_objs}')


    def __getitem__(self, idx):
        # first, random choose a dataset
        dataset_idx = np.random.randint(0, self.num_datasets)

        dataset_chosen = self.datasets[dataset_idx]

        # second, random choose an object
        total_objs = self.anno_file[dataset_chosen]['total_objs']

        object_chosen = np.random.randint(0, total_objs)

        # third, get the trajectories
        trajectories = self.anno_file[dataset_chosen][str(object_chosen)]['bboxes']
        traj_len = self.anno_file[dataset_chosen][str(object_chosen)]['traj_len']

        assert traj_len >= self.skip_sample * self.window_size, 'check the temporal window size'

        # random sample, choose the start idx
        max_start_idx = traj_len - 1 - self.skip_sample * (self.window_size - 1)
        start_idx = np.random.randint(0, max_start_idx + 1)

        sampled_trajectories = []
        for _ in range(self.window_size):
            sampled_trajectories.append(trajectories[start_idx])
            start_idx += self.skip_sample

        sampled_trajectories = torch.tensor(sampled_trajectories)
        if self.cfgs['manner'] == 'diff':
            # calculate the difference between bboxes
            ret = torch.zeros((self.window_size - 1, 4))   # temp window = 12, then generate 11 diff, first 10 used to forward the model, 
            # and the last 1 used to predict and calculate loss
            for i in range(self.window_size - 1):
                ret[i] = sampled_trajectories[i + 1] - sampled_trajectories[i]  # normalized diff

        return ret[:-1], ret[-1]  # x and label

    def __len__(self):
        return self.num_all_objs
    

class TrajDatasetv2(Dataset):
    """
    sample-wise instead of object-wise in class `TrajDataset`
    """

    def __init__(self, cfgs) -> None:
        super().__init__()

        self.cfgs = cfgs
        self.window_size = self.cfgs['window_size']
        self.skip_sample = self.cfgs['skip_sample']
        self.scale_factor = self.cfgs['scale_factor']
        self.scale_factor_diff = self.cfgs['scale_factor_diff'] if 'scale_factor_diff' in self.cfgs else None

        assert self.skip_sample == 1, 'not implemented yet'

        with open(cfgs['anno_path'], 'r') as f:
            self.anno_file = json.load(f, )  # get the annotation file

        self.datasets = list(self.anno_file.keys())
        self.num_datasets = len(self.datasets)

        self.num_samples = 0  # the total number of samples
        self.idx_obj_map = dict()  # key: start idx, value: obj id

        strat_idx = 0
        for dataset in self.datasets:
            for obj_id in self.anno_file[dataset].keys():
                if obj_id in ['total_objs', 'obj_id_start']: continue

                traj_len = self.anno_file[dataset][obj_id]['traj_len']
                self.idx_obj_map[strat_idx] = obj_id
                strat_idx += traj_len - self.window_size + 1
                
                self.num_samples += traj_len - self.window_size + 1 

        
        self.idx_milestone_list = list(self.idx_obj_map.keys())

    def __getitem__(self, idx):
        
        # binary search for the obj that idx corresponds to
        milestone_idx, obj = self._binary_search(idx)
        # find the dataset that obj in 
        dataset_key = None 
        for dataset in self.datasets:
            if int(obj) >= int(self.anno_file[dataset]['obj_id_start']): dataset_key = dataset
            else: break 

        bboxes = self.anno_file[dataset_key][obj]['bboxes']  # get the bboxes
        traj_start = idx - milestone_idx

        traj = torch.tensor(bboxes[traj_start: traj_start + self.window_size])

        if self.cfgs['manner'] == 'diff':
            # calculate the difference between bboxes
            condition = torch.zeros((self.window_size - 1, 4))   # temp window = 12, then generate 11 diff, first 10 used to forward the model, 
            # and the last 1 used to predict and calculate loss
            for i in range(self.window_size - 1):
                condition[i] = traj[i + 1] - traj[i]  # normalized diff

            return {'traj': traj, 'condition': condition[:-1] * self.scale_factor, 'label': condition[-1] * self.scale_factor}

        if self.cfgs['manner'] == 'bbox_and_diff':
            # calculate the difference between bboxes
            condition = torch.zeros((self.window_size - 1, 4))   # temp window = 12, then generate 11 diff, first 10 used to forward the model, 
            # and the last 1 used to predict and calculate loss
            for i in range(self.window_size - 1):
                condition[i] = traj[i + 1] - traj[i]  # normalized diff

            condition_ = torch.cat([traj[1:] * self.scale_factor, condition * self.scale_factor_diff], dim=1)  # [xc, yc, w, h, dx, dy, dw, dh]

            return {'traj': traj, 'condition': condition_[:-1], 'label': condition_[-1]}

        else:
            raise NotImplementedError

    def _binary_search(self, idx):
        # find the last i that self.idx_milestone_list[i] <= idx

        l, r = 0, len(self.idx_milestone_list) - 1

        while l < r:
            mid = (l + r + 1) >> 1
            if self.idx_milestone_list[mid] <= idx: l = mid
            else: r = mid - 1

        obj_idx = self.idx_milestone_list[l]
        return obj_idx, self.idx_obj_map[obj_idx]



    def __len__(self, ):
        return self.num_samples
