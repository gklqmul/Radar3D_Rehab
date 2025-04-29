import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import scipy.io as sio
import matplotlib

matplotlib.rcParams.update({
        'font.family': 'serif',
    'font.serif': ['Times New Roman'],
})
# Load skeleton data (format: frames x 32 x 3)
skeleton_data = np.load("dataset/env1/subjects/subject26/aligned/action01/aligned_skeleton_segment01.npy")  # Path to your .npy file
radar_data = sio.loadmat("dataset/env1/subjects/subject26/origal/1/aligned/aligned_radar_segment01.mat")
pc = radar_data["radar_data"]  # 888 frames, each with 12 attributes
num_frames = skeleton_data.shape[0]  # Total number of frames

# Skeleton bone connections (torso, limbs)
BONES = [
    (0, 1), (1, 2), (2, 3), (3, 26),  # Torso
    (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (8, 10),  # Left arm
    (3, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (15, 17),  # Right arm
    (0, 18), (18, 19), (19, 20), (20, 21),  # Left leg
    (0, 22), (22, 23), (23, 24), (24, 25)  # Right leg
]

frame_index = 26  # Select a frame for visualization
joints = skeleton_data[frame_index]  # Shape: (32, 3)
joints = joints / 1000
joints[:, 1] = -joints[:, 1]  # 骨骼数据

# Plot 3D skeleton
fig = plt.figure()  # 增加宽度比例
ax = fig.add_subplot(111, projection='3d')

# 关键修改：交换Y和Z轴数据
joints_swapped = joints.copy()
joints_swapped[:, 1], joints_swapped[:, 2] = joints[:, 2], joints[:, 1]

# Draw skeleton connections
# 绘制骨骼连线（增大关节点尺寸）
# for bone in BONES:
#     joint1, joint2 = bone
#     xs = [joints_swapped[joint1, 0], joints_swapped[joint2, 0]]
#     ys = [joints_swapped[joint1, 1], joints_swapped[joint2, 1]]
#     zs = [joints_swapped[joint1, 2], joints_swapped[joint2, 2]]
    
#     # 修改点线参数
#     ax.plot(xs, ys, zs,  
#             'o-',  # 移除'bo-'中的b，用统一颜色
#             color='#2E86AB',
#             markersize=10,  # 关节点从5增大到8
#             alpha=0.9,    # 提高不透明度
#             linewidth=4,  # 连线从2加粗到3
#             solid_capstyle='round',
#             markeredgecolor='white',  # 添加白色边缘
#             markeredgewidth=0.5)     # 边缘线宽

# 绘制雷达点（增大尺寸并增强对比）
radar_frame_data = pc[0, frame_index]
x = radar_frame_data[7].flatten()
z = -radar_frame_data[3].flatten()
y = radar_frame_data[8].flatten()

ax.scatter(x, y, z, 
           color='#F18F01', 
           marker='o', 
           s=50,  # 从20增大到40
           alpha=0.8,  # 提高不透明度
           edgecolors='white',
           linewidth=0.8,  # 边缘线宽从0.3增加到0.8
           label='Radar Points',
           zorder=10)  # 确保雷达点在顶层

# 坐标轴设置（保持不变）
ax.set_xlabel("X (m)", labelpad=15, fontweight='bold')
ax.set_ylabel("Z (m)", labelpad=15, fontweight='bold')
ax.set_zlabel("Y (m)", labelpad=15, fontweight='bold')

# 视角和比例（保持不变）
ax.view_init(elev=20, azim=-45)
ax.set_box_aspect([1, 1, 1])
ax.set_xlim(-1.5, 1.5)
ax.set_zlim(-1, 1)
ax.set_ylim(1, 4)

# 添加网格增强可读性
ax.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout(pad=0)  # 去除多余留白

# 3. 交互调整
plt.ion()
plt.show()
input("调整好角度后按回车...")

# 4. 保存最终视角
plt.savefig(
    'output.png',
    dpi=300,                   # 印刷级分辨率
    bbox_inches='tight',        # 去除白边
    pad_inches=0,               # 零边距
    facecolor='white',          # 确保背景白
    transparent=True,          # 关闭透明
)

print(f"已保存！视角参数：elev={ax.elev:.1f}°, azim={ax.azim:.1f}°")

# Animation setup
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Set axis limits
# x_min, x_max = np.min(skeleton_data[:, :, 0]), np.max(skeleton_data[:, :, 0])
# y_min, y_max = np.min(skeleton_data[:, :, 1]), np.max(skeleton_data[:, :, 1])
# z_min, z_max = np.min(skeleton_data[:, :, 2]), np.max(skeleton_data[:, :, 2])
# ax.set_xlim(x_min, x_max)
# ax.set_ylim(y_min, y_max)
# ax.set_zlim(z_max, z_min)  # Invert Z-axis for better camera view

# # Set viewing angle
# ax.view_init(elev=-90, azim=-90)

# # Update function
# def update(frame_idx):
#     ax.cla()  # Clear previous frame
#     ax.set_xlim(x_min, x_max)
#     ax.set_ylim(y_min, y_max)
#     ax.set_zlim(z_max, z_min)
#     ax.view_init(elev=-90, azim=-90)

#     joints = skeleton_data[frame_idx]  # Get skeleton data for the current frame

#     # Draw skeleton connections
#     for bone in BONES:
#         joint1, joint2 = bone
#         xs = [joints[joint1, 0], joints[joint2, 0]]
#         ys = [joints[joint1, 1], joints[joint2, 1]]
#         zs = [joints[joint1, 2], joints[joint2, 2]]
#         ax.plot(xs, ys, zs, 'bo-', markersize=5, alpha=0.8)

#     ax.set_title(f"Frame {frame_idx}/{num_frames}")

# # Create animation
# ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=50)

# # Save as GIF or MP4
# ani.save("skeleton_animation.gif", writer="pillow", fps=20)  # GIF
# # ani.save("skeleton_animation.mp4", writer="ffmpeg", fps=20)  # MP4

# plt.show()
