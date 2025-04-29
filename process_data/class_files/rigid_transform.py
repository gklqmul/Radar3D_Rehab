import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class RigidTransform:
    def __init__(self, file_path):
        """
        Initialize with the path to the Excel file containing point data.
        :param file_path: Path to the Excel file
        """
        self.file_path = file_path
        self.P, self.Q = self.load_data()

    def load_data(self):
        """
        Load and process point data from the Excel file.
        :return: Two numpy arrays P and Q, each of shape (N, 3)
        """
        df = pd.read_excel(self.file_path)
        P = df.iloc[:, 0:3].to_numpy()  # kinect points
        Q = df.iloc[:, 3:6].to_numpy()  # radar points
        print(P.shape, Q.shape)  # 确认形状一致

        # Adjust for coordinate system differences
        P[:, 2] *= -1  # Invert z-axis

        return P, Q

    def compute_rigid_transform(self):
        """
        Compute the rigid transformation (rotation matrix R and translation vector t) such that Q ≈ R * P + t.
        :return: Rotation matrix R (3x3) and translation vector t (3,)
        """
        P, Q = self.P, self.Q

        # Compute centroids
        p_centroid = np.mean(P, axis=0)
        q_centroid = np.mean(Q, axis=0)

        # Center the points
        P_centered = P - p_centroid
        Q_centered = Q - q_centroid

        # Compute covariance matrix
        H = P_centered.T @ Q_centered

        # Perform SVD decomposition
        U, S, Vt = np.linalg.svd(H)

        # Compute rotation matrix R
        R = Vt.T @ U.T

        # Ensure R is a proper rotation matrix (det(R) should be 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Compute translation vector t
        t = q_centroid - R @ p_centroid

        # Save R and t to file
        with open('rigid_transform.pkl', 'wb') as f:
            pickle.dump((R, t), f)

        return R, t

    def transform_points(self, R, t):
        """
        Transform the 3D point set P using rotation matrix R and translation vector t.
        :param R: (3, 3) rotation matrix
        :param t: (3,) translation vector
        :return: Transformed point set
        """
        return self.P @ R.T + t

    def compute_error(self, P_transformed):
        """
        Compute the transformation error (Euclidean distance between transformed P and Q).
        :param P_transformed: Transformed point set
        :return: Array of errors and the mean error
        """
        errors = np.linalg.norm(P_transformed - self.Q, axis=1)
        return errors, np.mean(errors)

    def plot_alignment(self, P_transformed):
        """
        Plot the original and transformed point sets in 3D space to visualize alignment.
        :param P_transformed: Transformed point set
        """
    # Plot original points before alignment
        fig1 = plt.figure(figsize=(8, 6))
        ax1 = fig1.add_subplot(111, projection='3d')
        ax1.scatter(self.P[:, 0], self.P[:, 1], self.P[:, 2], c='#1f77b4', label='P (Kinect)')
        ax1.scatter(self.Q[:, 0], self.Q[:, 1], self.Q[:, 2], c='#d62728', label='Q (Radar)')
        ax1.set_title('Before Transformation', pad=50, fontsize=20)
        ax1.set_xlabel('X', fontsize=16)
        ax1.set_ylabel('Y', fontsize=16)
        ax1.set_zlabel('Z', fontsize=16)
        ax1.legend(fontsize=20)
        plt.tight_layout()
        plt.savefig('before.png', dpi=300)  # Save high-resolution image
        plt.show()

        # Plot points after alignment
        fig2 = plt.figure(figsize=(8, 6))
        ax2 = fig2.add_subplot(111, projection='3d')
        ax2.scatter(P_transformed[:, 0], P_transformed[:, 1], P_transformed[:, 2], c='#1f77b4', label='P (Kinect)')
        ax2.scatter(self.Q[:, 0], self.Q[:, 1], self.Q[:, 2], c='#d62728', label='Q (Radar)')
        ax2.set_title('After Transformation', pad=50, fontsize=20)
        ax2.set_xlabel('X', fontsize=16)
        ax2.set_ylabel('Y', fontsize=16)
        ax2.set_zlabel('Z', fontsize=16)
        ax2.legend(fontsize=20)
        plt.tight_layout()
        plt.savefig('after.png', dpi=300)  # Save high-resolution image
        plt.show()

    def run(self):
        """
        Execute the rigid transformation computation and evaluate the transformation error.
        """
        R, t = self.compute_rigid_transform()
        print("Computed rotation matrix R:")
        print(R)
        print("\nComputed translation vector t:")
        print(t)

        P_transformed = self.transform_points(R, t)
        errors, mean_error = self.compute_error(P_transformed)

        print("\nTransformation errors (Euclidean distance) for each point:")
        print(errors)
        print("\nAverage error:", mean_error)

        # Visualize alignment
        self.plot_alignment(P_transformed)


# Example usage
if __name__ == "__main__":
    file_path = "./process/process files/central point coordinate.xlsx"
    transform = RigidTransform(file_path)
    transform.run()

