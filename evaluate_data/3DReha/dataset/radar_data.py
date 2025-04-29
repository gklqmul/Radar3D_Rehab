import os
import h5py
import numpy as np
from sklearn.cluster import DBSCAN
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split



class RadarKeypoint(Dataset):
  
    seed = 42              
    partitions = (0.8, 0.1, 0.1)         
    num_keypoints = 32    
    zero_padding = 'zero' 
    zero_padding_styles = ['zero', 'repeat'] 

    def __init__(self, root_dir, mmr_dataset_config=None):
        super().__init__()
        self._parse_config(mmr_dataset_config)
        self.datapath = []          
        self.frame_level_data = []  
        self.info = {
            'num_classes': None,
            'num_samples': len(self.frame_level_data),
            'num_keypoints': self.num_keypoints,
            'stacks': self.stacks,
            'unit': 'dm'
        }

        self._collect_data_paths(root_dir)

        self._process_all_samples()

        self._split_dataset()

        
    def _parse_config(self, config):
        if config is None:
            return
        config = {k: v for k, v in config.items() if v is not None}
        self.seed = config.get('seed', self.seed)
        self.max_points = config.get('max_points', self.max_points)
        self.partitions = (
            config.get('train_split', self.partitions[0]),
            config.get('val_split', self.partitions[1]),
            config.get('test_split', self.partitions[2]))
        self.stacks = config.get('stacks', self.stacks)
        self.zero_padding = config.get('zero_padding', self.zero_padding)
        self.num_keypoints = config.get('num_keypoints', self.num_keypoints)
        self.step = config.get('step', self.step)
        
        if self.zero_padding not in self.zero_padding_styles:
            raise ValueError(f"not support: {self.zero_padding}")

    def _collect_data_paths(self, root_dir):
        for env in ['env1', 'env2']:
            env_path = os.path.join(root_dir, env, 'subjects')
            if not os.path.exists(env_path):
                continue

            for subject in os.listdir(env_path):
                aligned_path = os.path.join(env_path, subject, 'aligned')
                if not os.path.exists(aligned_path):
                    continue

                for action in os.listdir(aligned_path):
                    action_path = os.path.join(aligned_path, action)
                    if not os.path.isdir(action_path):
                        continue

                    radar_files = [f for f in os.listdir(action_path) if f.endswith('.h5')]
                    skeleton_files = [f for f in os.listdir(action_path) if f.endswith('.npy')]
                    
                    if radar_files and skeleton_files:
                        self.datapath.append((
                            os.path.join(action_path, radar_files[0]),
                            os.path.join(action_path, skeleton_files[0])
                        ))
           
    def _process_all_samples(self):
        for radar_path, skeleton_path in self.datapath:
            self._process_single_sample(radar_path, skeleton_path)
        self.info['num_samples'] = len(self.frame_level_data)

    def _process_single_sample(self, radar_path, skeleton_path):
     
        raw_radar_frames = []
        with h5py.File(radar_path, 'r') as f:
            for name in sorted(f["frames"].keys()):
                frame = np.array(f["frames"][name])
                raw_radar_frames.append(frame)
        
        skeleton_data = np.load(skeleton_path).astype(np.float32)
        
        assert len(raw_radar_frames) == len(skeleton_data), \
            f"frame sum don't match: radar{len(raw_radar_frames)} != skeleton{len(skeleton_data)}"

        stack_radius = self.stacks // 2 if self.stacks else 0
        total_frames = len(raw_radar_frames)
        step = self.step if self.step else 1
        for center_idx in range(0, total_frames, step):
            stacked_frames = []
            for offset in range(-stack_radius, stack_radius + 1):
                frame_idx = center_idx + offset
                frame_idx = max(0, min(frame_idx, total_frames - 1))
                stacked_frames.append(raw_radar_frames[frame_idx])
       
            stacked_points = np.vstack(stacked_frames)
            stacked_points = self._clean_and_expend(stacked_points)[:,[5,1,6]] * 10
            if len(stacked_points) > 0:
                stacked_points = np.unique(stacked_points, axis=0)
            
            processed_points = self._process_point_cloud(stacked_points)
            
            skeleton_data = skeleton_data / 100
            self.frame_level_data.append((
                torch.from_numpy(processed_points).float(),  # (max_points, 3)
                torch.from_numpy(skeleton_data[center_idx]).float()  # (32, 3)
            ))

    def _process_point_cloud(self, points):
        if points.size == 0:
            points = np.zeros((0, 3))
        
        if len(points) > self.max_points:
            indices = np.random.choice(len(points), self.max_points, replace=False)
            return points[indices]
        
        if len(points) < self.max_points:
            pad_size = self.max_points - len(points)
            if self.zero_padding == 'zero':
                padding = np.zeros((pad_size, 3))
            elif self.zero_padding == 'repeat':
                padding = np.tile(points[-1:], (pad_size, 1)) if len(points) > 0 else np.zeros((pad_size, 3))
            return np.vstack([points, padding])
        
        return points
    
    
    def _clean_and_expend(self, radar_data):

        if radar_data.shape[0] == 0 or len(radar_data.shape) != 2 or radar_data.shape[1] < 8:
            return np.zeros((1, 8))

        el = radar_data[:, 0]       # el
        z = radar_data[:, 1]        # Z height
        az = radar_data[:, 2]       # az
        doppler = radar_data[:, 3]  # doppler
        rng = radar_data[:, 4]      # rng
        x = radar_data[:, 5]        # X
        y = radar_data[:, 6]        # Y
        snr = radar_data[:, 7]      # snr


        valid_mask = (x >= -1.5) & (x <= 1.5) & (y >= 1.0) & (y <= 4.5)

        if not np.any(valid_mask):
            return np.zeros((1, 8)) 

        el = el[valid_mask]
        z = z[valid_mask]
        az = az[valid_mask]
        doppler = doppler[valid_mask]
        rng = rng[valid_mask]
        x = x[valid_mask]
        y = y[valid_mask]
        snr = snr[valid_mask]

        cleaned_frame = np.column_stack([
            el, z, az, doppler, rng, x, y, snr
        ])

        return cleaned_frame
  
    def _split_dataset(self):

        self._train_data, self._test_data = train_test_split(
            self.frame_level_data,
            test_size=self.partitions[2],
            random_state=self.seed
        )
        self._train_data, self._val_data = train_test_split(
            self._train_data,
            test_size=self.partitions[1]/(self.partitions[0]+self.partitions[1]),
            random_state=self.seed
        )

    def __len__(self):
        return len(self.frame_level_data)

    def __getitem__(self, idx):
        return self.frame_level_data[idx]

    @property
    def train_data(self):
        return self._train_data
    
    @property
    def val_data(self):
        return self._val_data
    
    @property
    def test_data(self):
        return self._test_data

    def validate_dataset(self):

        points, skeleton = self[0]
        
        print(f"radar data range: [{points.min():.2f}, {points.max():.2f}]")
        print(f"skeleton range: [{skeleton.min():.2f}, {skeleton.max():.2f}]")

        for i, (raw_p, raw_s) in enumerate(self.frame_level_data):
            assert raw_p.shape == (self.max_points, 3), \
                f"radar shape error, sample{i}: {raw_p.shape} != {(self.max_points, 3)}"
            assert raw_s.shape == (self.num_keypoints, 3), \
                f"skeleton shape error, sample{i}: {raw_s.shape} != {(self.num_keypoints, 3)}"
        
        print(f"pass, total samples: {len(self)} | train: {len(self.train_data)} | "
              f"val: {len(self.val_data)} | test: {len(self.test_data)}")
        print("="*40)

class RadarIdentity(RadarKeypoint): 
    max_points = 1024
    def __init__(self, root_dir, mmr_dataset_config=None):
    
        super().__init__(root_dir, mmr_dataset_config)
        self.info['max_points'] = self.max_points
        self._calculate_num_classes()
        
    def _process_single_sample(self, radar_path):
        raw_radar_frames = []
        with h5py.File(radar_path, 'r') as f:
            for name in sorted(f["frames"].keys()):
                frame = np.array(f["frames"][name])
                raw_radar_frames.append(frame)

        stack_radius = self.stacks // 2 if self.stacks else 0
        total_frames = len(raw_radar_frames)
        step = self.step if self.step else 1
        
        for center_idx in range(0, total_frames, step):
            stacked_frames = []
            for offset in range(-stack_radius, stack_radius + 1):
                frame_idx = center_idx + offset
                frame_idx = max(0, min(frame_idx, total_frames - 1))
                stacked_frames.append(raw_radar_frames[frame_idx])

            stacked_points = np.vstack(stacked_frames)   # (N*stack_size, 3)
            stacked_points = self._clean_and_expend(stacked_points)[:,[5,1,6]] * 10
            if len(stacked_points) > 0:
                stacked_points = np.unique(stacked_points, axis=0)
            
            processed_points = self._process_point_cloud(stacked_points)
            label = int(radar_path.split('subject')[-1][:2]) - 1
        
            self.frame_level_data.append((
                torch.from_numpy(processed_points).float(),  # (max_points, 3)
                label
            ))

    def _process_all_samples(self):
        self.frame_level_data = []  
        
        for radar_path, skeleton_path in self.datapath:
           self._process_single_sample(radar_path) 
        self.info['num_samples'] = len(self.frame_level_data)
    
    def _calculate_num_classes(self):
        all_labels = []
        for data in [self._train_data, self._val_data, self._test_data]:
            all_labels.extend([y for(x,y) in data])
        
        if all_labels:
            max_label = max(all_labels)
            self.num_classes = max_label + 1
            self.info['num_classes'] = self.num_classes
            print(f"Total number of classes: {self.num_classes}")

    def __getitem__(self, idx):
        return self.frame_level_data[idx]
    
    def validate_dataset(self):
        points, label = self[0]
        
        print(f"Radar points range: [{points.min():.2f}, {points.max():.2f}]")
        print(f"Label: {label}")
        
        for i, (raw_p, raw_l) in enumerate(self.frame_level_data):
            assert raw_p.shape == (self.max_points, 3), \
                f"Radar shape error at sample {i}: {raw_p.shape} != {(self.max_points, 3)}"
            assert isinstance(raw_l, int), \
                f"Label type error at sample {i}: {type(raw_l)} != int"
        
        print(f"Validation passed. Total samples: {len(self)} | "
              f"Train: {len(self.train_data)} | Val: {len(self.val_data)} | "
              f"Test: {len(self.test_data)}")
        print("=" * 40)


class RadarAction(RadarKeypoint):
 
    def __init__(self, root_dir, mmr_dataset_config=None):
        super().__init__(root_dir, mmr_dataset_config)
        self.info['max_points'] = self.max_points
        self.info['step'] = self.step
        self._calculate_num_classes()
        
    def _process_single_sample(self, radar_path, _):
        raw_radar_frames = []
        with h5py.File(radar_path, 'r') as f:
            for name in sorted(f["frames"].keys()):
                frame = np.array(f["frames"][name])
                raw_radar_frames.append(frame)

        stack_radius = self.stacks // 2 if self.stacks else 0
        total_frames = len(raw_radar_frames)
        step = self.step if self.step else 1

        for center_idx in range(0, total_frames, step):
            stacked_frames = []
            for offset in range(-stack_radius, stack_radius + 1):
                frame_idx = center_idx + offset
                frame_idx = max(0, min(frame_idx, total_frames - 1))
                stacked_frames.append(raw_radar_frames[frame_idx])

            stacked_points = np.vstack(stacked_frames)   # (N*stack_size, 8)
            stacked_points = self._clean_and_expend(stacked_points)[:,[5,1,6]] * 10
            if len(stacked_points) > 0:
                stacked_points = np.unique(stacked_points, axis=0)
            
            processed_points = self._process_point_cloud(stacked_points)
            label = int(radar_path.split('action')[-1][:2]) - 1
        
            self.frame_level_data.append((
                torch.from_numpy(processed_points).float(),  # (max_points, 3)
                label
            ))

    def _process_all_samples(self): 
        for radar_path, skeleton_path in self.datapath:
            self._process_single_sample(radar_path, skeleton_path)
        self.info['num_samples'] = len(self.frame_level_data)
    
    
    def _calculate_num_classes(self):
        all_labels = []
        for data in [self._train_data, self._val_data, self._test_data]:
            all_labels.extend([y for(x,y) in data])
        
        if all_labels:
            max_label = max(all_labels)
            self.num_classes = max_label + 1
            self.info['num_classes'] = self.num_classes
            print(f"Total number of classes: {self.num_classes}")

    def __getitem__(self, idx):
        return self.frame_level_data[idx]
    
    def validate_dataset(self):
        points, label = self[0]
        
        print(f"Radar points range: [{points.min():.2f}, {points.max():.2f}]")
        print(f"Label: {label}")
        
        for i, (raw_p, raw_l) in enumerate(self.frame_level_data):
            assert raw_p.shape == (self.max_points, 3), \
                f"Radar shape error at sample {i}: {raw_p.shape} != {(self.max_points, 3)}"
            assert isinstance(raw_l, int), \
                f"Label type error at sample {i}: {type(raw_l)} != int"
        
        print(f"Validation passed. Total samples: {len(self)} | "
              f"Train: {len(self.train_data)} | Val: {len(self.val_data)} | "
              f"Test: {len(self.test_data)}")
        print("=" * 40)