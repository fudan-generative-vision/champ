import os
import numpy as np
import torch
import torch.nn.functional as F

JOINT_NAMES = [
    'left_hip', 
    'right_hip', 
    'spine1', 
    'left_knee', 
    'right_knee', 
    'spine2', 
    'left_ankle', 
    'right_ankle', 
    'spine3', 
    'left_foot', 
    'right_foot', 
    'neck', 
    'left_collar', 
    'right_collar', 
    'head', 
    'left_shoulder', 
    'right_shoulder', 
    'left_elbow', 
    'right_elbow', 
    'left_wrist', 
    'right_wrist'
]

# Manually chosen probability density thresholds for each joint
# Probablities computed using SIGMA=2 gaussian blur on AMASS pose 3D histogram for range (-pi,pi) with 100x100x100 bins
JOINT_NAME_PROB_THRESHOLDS = {
    'left_hip': 5e-5,
    'right_hip': 5e-5,
    'spine1': 2e-3,
    'left_knee': 5e-6,
    'right_knee': 5e-6,
    'spine2': 0.01,
    'left_ankle': 5e-6,
    'right_ankle': 5e-6,
    'spine3': 0.025,
    'left_foot': 0,
    'right_foot': 0,
    'neck': 2e-4,
    'left_collar': 4.5e-4 ,
    'right_collar': 4.5e-4,
    'head': 5e-4,
    'left_shoulder': 2e-4,
    'right_shoulder': 2e-4,
    'left_elbow': 4e-5,
    'right_elbow': 4e-5,
    'left_wrist': 1e-3,
    'right_wrist': 1e-3,
}

JOINT_IDX_PROB_THRESHOLDS = torch.tensor([JOINT_NAME_PROB_THRESHOLDS[joint_name] for joint_name in JOINT_NAMES])

###################################################################
POSE_RANGE_MIN = -np.pi
POSE_RANGE_MAX = np.pi

# Create 21x100x100x100 histogram of all 21 AMASS body pose joints using `create_pose_hist(amass_poses, nbins=100)`
AMASS_HIST100_PATH = 'hmr2_training_data/amass_poses_hist100_SMPL+H_G.npy'
if not os.path.exists(AMASS_HIST100_PATH):
    AMASS_HIST100_PATH = '/shared/shubham/code/hmr2023/amass_poses_hist100_SMPL+H_G.npy'
if not os.path.exists(AMASS_HIST100_PATH):
    AMASS_HIST100_PATH = '/fsx/shubham/code/stable-humans/notebooks/amass_poses_hist100_SMPL+H_G.npy'


def create_pose_hist(poses: np.ndarray, nbins: int = 100) -> np.ndarray:
    N,K,C = poses.shape
    assert C==3, poses.shape
    poses_21x3 = normalize_axis_angle(torch.fromnumpy(poses).view(N*K,3)).numpy().reshape(N, K, 3)
    assert (poses_21x3 > -np.pi).all() and (poses_21x3 < np.pi).all()

    Hs, Es = [], []
    for i in range(K):
        H, edges = np.histogramdd(poses_21x3[:, i, :], bins=nbins, range=[(-np.pi, np.pi)]*3)
        Hs.append(H)
        Es.append(edges)
    Hs = np.stack(Hs, axis=0)
    return Hs

def load_amass_hist_smooth(sigma=2) -> torch.Tensor:
    amass_poses_hist100 = np.load(AMASS_HIST100_PATH)
    amass_poses_hist100 = torch.from_numpy(amass_poses_hist100)
    assert amass_poses_hist100.shape == (21,100,100,100)

    nbins = amass_poses_hist100.shape[1]
    amass_poses_hist100 = amass_poses_hist100/amass_poses_hist100.sum() / (2*np.pi/nbins)**3

    # Gaussian filter on amass_poses_hist100
    from scipy.ndimage import gaussian_filter
    amass_poses_hist100_smooth = gaussian_filter(amass_poses_hist100.numpy(), sigma=sigma, mode='constant')
    amass_poses_hist100_smooth = torch.from_numpy(amass_poses_hist100_smooth)
    return amass_poses_hist100_smooth

# Normalize axis angle representation s.t. angle is in [-pi, pi]
def normalize_axis_angle(poses: torch.Tensor) -> torch.Tensor:
    # poses: N, 3
    # print(f'normalize_axis_angle ...')
    assert poses.shape[1] == 3, poses.shape
    angle = poses.norm(dim=1)
    axis = F.normalize(poses, p=2, dim=1, eps=1e-8)
    
    angle_fixed = angle.clone()
    axis_fixed = axis.clone()

    eps = 1e-6
    ii = 0
    while True:
        # print(f'normalize_axis_angle iter {ii}')
        ii += 1
        angle_too_big = (angle_fixed > np.pi + eps)
        if not angle_too_big.any():
            break
        
        angle_fixed[angle_too_big] -= 2 * np.pi
        angle_too_small = (angle_fixed < -eps)
        axis_fixed[angle_too_small] *= -1
        angle_fixed[angle_too_small] *= -1
    
    return axis_fixed * angle_fixed[:,None]

def poses_to_joint_probs(poses: torch.Tensor, amass_poses_100_smooth: torch.Tensor) -> torch.Tensor:
    # poses: Nx69
    # amass_poses_100_smooth: 21xBINSxBINSxBINS
    # returns: poses_prob: Nx21
    N=poses.shape[0]
    assert poses.shape == (N,69)
    poses = poses[:,:63].reshape(N*21,3)

    nbins = amass_poses_100_smooth.shape[1]
    assert amass_poses_100_smooth.shape == (21,nbins,nbins,nbins)

    poses_bin = (poses - POSE_RANGE_MIN) / (POSE_RANGE_MAX - POSE_RANGE_MIN) * (nbins - 1e-6)
    poses_bin = poses_bin.long().clip(0, nbins-1)
    joint_id = torch.arange(21, device=poses.device).view(1,21).expand(N,21).reshape(N*21)
    poses_prob = amass_poses_100_smooth[joint_id, poses_bin[:,0], poses_bin[:,1], poses_bin[:,2]]

    poses_bad = ((poses < POSE_RANGE_MIN) | (poses >= POSE_RANGE_MAX)).any(dim=1)
    poses_prob[poses_bad] = 0

    return poses_prob.view(N,21)

def poses_check_probable(
        poses: torch.Tensor, 
        amass_poses_100_smooth: torch.Tensor, 
        prob_thresholds: torch.Tensor = JOINT_IDX_PROB_THRESHOLDS
    ) -> torch.Tensor:
    N,C=poses.shape
    poses_norm = normalize_axis_angle(poses.reshape(N*(C//3),3)).reshape(N,C)
    poses_prob = poses_to_joint_probs(poses_norm, amass_poses_100_smooth)
    return (poses_prob > prob_thresholds).all(dim=1)
