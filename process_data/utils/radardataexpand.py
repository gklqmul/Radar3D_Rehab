import numpy as np
import scipy
from sklearn.cluster import DBSCAN

def clean_radar_data(radar_data, Ni=5, epsilon=1, minpts=30,
                     dRNG=0.0633, dEL=0.04, dAZ=0.01):
    half_Ni = (Ni - 1) // 2
    XYZ_list, IDX_list = [], []
    
    # Step 1: 合并帧 + 添加扰动 + 极坐标转直角
    for i in range(half_Ni, len(radar_data) - half_Ni):
        rng = radar_data[i - half_Ni]['rng']
        el = radar_data[i - half_Ni]['el']
        az = radar_data[i - half_Ni]['az']
        snr = radar_data[i - half_Ni]['snr']

        for k in range(i - half_Ni + 1, i + half_Ni + 1):
            rng = np.concatenate([rng, radar_data[k]['rng']])
            el = np.concatenate([el, radar_data[k]['el']])
            az = np.concatenate([az, radar_data[k]['az']])
            snr = np.concatenate([snr, radar_data[k]['snr']])

        rng += (dRNG * np.random.rand(*rng.shape)) - dRNG / 2
        el  += (dEL  * np.random.rand(*el.shape))  - dEL / 2
        az  += (dAZ  * np.random.rand(*az.shape))  - dAZ / 2

        x = rng * np.sin(az) * np.cos(el)
        y = rng * np.cos(az) * np.cos(el)
        z = rng * np.sin(el)

        XYZ = np.stack([x, y, z], axis=1)
        XYZ_list.append(XYZ)

        # DBSCAN 聚类
        db = DBSCAN(eps=epsilon, min_samples=minpts)
        labels = db.fit_predict(XYZ)
        IDX_list.append(labels)

    # Step 2: 计算初始中心（默认选聚类索引1）
    SELECT_INDEX = 1
    M_all = []
    for j in range(len(XYZ_list)):
        points = XYZ_list[j][IDX_list[j] == SELECT_INDEX]
        if len(points) > 0:
            M_all.append(points)
    if M_all:
        centre = np.mean(np.vstack(M_all), axis=0)
    else:
        centre = np.zeros(3)

    # Step 3: 重新判断每帧正确聚类
    selected_indices = []
    for j, XYZ in enumerate(XYZ_list):
        shifted = XYZ - centre
        com_dists = []
        for k in range(np.max(IDX_list[j]) + 1):
            pts = shifted[IDX_list[j] == k]
            if len(pts) > 0:
                dist = np.linalg.norm(np.mean(pts, axis=0))
            else:
                dist = np.inf
            com_dists.append(dist)
        k_selected = int(np.argmin(com_dists))
        selected_indices.append(k_selected)

    # Step 4: 使用选定聚类，构建输出
    radar_data_cleaned = []
    for j, i in enumerate(range(half_Ni, len(radar_data) - half_Ni)):
        XYZ = XYZ_list[j]
        labels = IDX_list[j]
        selected = selected_indices[j]

        mask = labels == selected
        if np.sum(mask) == 0:
            continue  # skip if no foreground

        x, y, z = XYZ[mask].T
        rng = np.linalg.norm(XYZ[mask], axis=1)
        el = np.arcsin(z / rng)
        az = np.arctan2(x, y)
        snr = radar_data[i]['snr'][mask] if len(radar_data[i]['snr']) == len(mask) else np.zeros_like(rng)

        radar_data_cleaned.append({
            'rng': rng,
            'az': az,
            'el': el,
            'X': x,
            'Y': y,
            'Z': z,
            'snr': snr
        })

    return radar_data_cleaned

# Example usage
if __name__ == "__main__":


    cleaned_data = clean_radar_data(radar_data)
    print("Cleaned Radar Data:")
    for frame in cleaned_data:
        print(frame)