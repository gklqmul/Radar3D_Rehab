# import os
# import numpy as np
# from pathlib import Path
# from utils import save_skeleton_segments


# if __name__ == "__main__":
#     parent_dir = "dataset/env1/subjects/subject26/origal/5"
#     file_path = Path(parent_dir) / "action_segments.txt"
#     action_segments = np.loadtxt(file_path, delimiter=",", dtype=int)
#     file_path1 = Path(parent_dir) / "body_skeleton.npy"
#     skeleton_frames = np.load(file_path1, allow_pickle=True)
#     save_skeleton_segments(skeleton_frames, action_segments, parent_dir)

# import os
# import numpy as np

# # 定义路径
# base_path = "dataset/env2/subjects/subject26/origal"  # 存储 action_segments.txt 和 结果的路径
# elif_dir = "C:/PHD/elif"  # npy 文件所在目录

# # 读取分割信息
# def load_action_segments(filepath):
#     """从 action_segments.txt 读取分割范围"""
#     segments = []
#     if not os.path.exists(filepath):
#         print(f"⚠️ 分割文件未找到: {filepath}")
#         return segments  # 返回空列表，避免程序崩溃
#     with open(filepath, "r") as f:
#         for line in f:
#             start, end = map(int, line.strip().split(","))
#             segments.append((start, end))
#     return segments

# # 处理单个 npy 文件
# def process_npy_file(npy_filename, folder_index):
#     """根据 folder_index 目录下的 action_segments.txt 分割 npy 文件，并存储结果"""
#     npy_path = os.path.join(elif_dir, npy_filename)
#     action_segments_path = os.path.join(base_path, str(folder_index), "action_segments.txt")
#     save_folder = os.path.join(base_path, str(folder_index))  # 结果存入该目录

#     if not os.path.exists(npy_path):
#         print(f"⚠️ npy 文件未找到: {npy_path}")
#         return

#     action_segments = load_action_segments(action_segments_path)
#     if not action_segments:
#         print(f"⚠️ 没有找到有效的动作分割信息，跳过: {action_segments_path}")
#         return

#     data = np.load(npy_path, allow_pickle=True)  # 读取 npy 数据
#     os.makedirs(save_folder, exist_ok=True)  # 确保存储路径存在

#     # 分割并保存
#     for idx, (start, end) in enumerate(action_segments, start=1):
#         segment_data = data[start:end+1]  # 提取分割的片段
#         save_path = os.path.join(save_folder, f"timestamp{idx}.npy")
#         np.save(save_path, segment_data)
#         print(f"✅ 已保存: {save_path} (帧 {start}-{end})")

# # 处理多个 npy 文件
# for i, file_id in enumerate(range(15, 21), start=7):  # 9.npy -> 1, 10.npy -> 2, ..., 15.npy -> 7
#     npy_filename = f"{file_id}.npy"
#     process_npy_file(npy_filename, folder_index=i)

# print("🎯 所有文件处理完成！")

import h5py

with h5py.File("output901.h5", "r") as h5f:
    frame_ds = h5f["frames"]["frame_004"]  # Load frame 10
    frame_data = frame_ds[:]  # Get the point data
    timestamp = frame_ds.attrs["timestamp"]  # Get the timestamp
    attribute_names = [name.decode() for name in frame_ds.attrs["attribute_names"]]  # Decode attribute names
    
    print(f"Frame 10: {frame_data.shape} (num_points, 12)")
    print(f"Frame 10: {frame_data} data")
    print(f"Timestamp: {timestamp}")
    print("Attributes:", attribute_names)


    
    
    