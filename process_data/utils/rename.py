import os

folder_path = "D:/kinect/data19/"  # 文件夹路径

# 获取文件夹中的所有文件
files = os.listdir(folder_path)

# 按文件扩展名分组 .npy 和 .mkv 文件
npy_files = [file for file in files if file.endswith(".npy")]
mkv_files = [file for file in files if file.endswith(".mkv")]

# 对文件进行排序（按文件名排序，确保 .npy 和 .mkv 文件一一对应）
npy_files.sort()
mkv_files.sort()

# 以从 9 到 20 的序号进行重命名
for i, (npy_file, mkv_file) in enumerate(zip(npy_files, mkv_files), start=9):
    # 获取文件的基础名称（不含扩展名）
    base_name_npy = os.path.splitext(npy_file)[0]
    base_name_mkv = os.path.splitext(mkv_file)[0]
    
    # 确保基础名称相同后，重新命名
    if base_name_npy == base_name_mkv:
        # 新的文件名（带扩展名）
        new_name = f"{i}"
        new_npy_name = f"{new_name}.npy"
        new_mkv_name = f"{new_name}.mkv"
        
        # 获取旧文件和新文件的完整路径
        old_npy_path = os.path.join(folder_path, npy_file)
        new_npy_path = os.path.join(folder_path, new_npy_name)
        old_mkv_path = os.path.join(folder_path, mkv_file)
        new_mkv_path = os.path.join(folder_path, new_mkv_name)
        
        # 重命名文件
        os.rename(old_npy_path, new_npy_path)
        os.rename(old_mkv_path, new_mkv_path)
        
        print(f"Renamed: {npy_file} -> {new_npy_name}")
        print(f"Renamed: {mkv_file} -> {new_mkv_name}")
