import torch_geometric.transforms as T
import logging
import numpy as np

from .radar_data import RadarAction, RadarKeypoint, RadarIdentity


from torch_geometric.loader import DataLoader

dataset_map = {
    'radar_act': RadarAction,
    'radar_kp': RadarKeypoint,
    'radar_iden': RadarIdentity,
}

class Scale(T.BaseTransform):
    def __init__(self, factor) -> None:
        self.s = factor
        super().__init__()

    def __call__(self, data):
        x, y = data
        return x*self.s, y*self.s


transform_map = {
    'mmr_kp': (None, Scale(100)),
}

def get_dataset(name, batch_size, workers, dataset_config=None):
    dataset_cls = dataset_map[name]
    data_info={}
    
    if name in ['radar_act','radar_kp','radar_iden']:
            dataset = dataset_cls(
                root_dir='./dataset', 
                mmr_dataset_config=dataset_config)
            dataset.validate_dataset()
            train_dataset = dataset.train_data
            test_dataset = dataset.test_data
            val_dataset = dataset.val_data
            data_info = dataset.info
            print("---------",data_info)
    else:
        raise ValueError(f"Dataset {name} not supported.")
    
    if data_info['num_classes'] is not None:
        logging.info('Number of classes: %s' % data_info['num_classes'])
    else:
        logging.info('Number of keypoints: %s' % data_info['num_keypoints'])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    return train_loader, val_loader, test_loader, data_info