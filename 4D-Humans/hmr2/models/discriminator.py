import torch
import torch.nn as nn

class Discriminator(nn.Module):

    def __init__(self):
        """
        Pose + Shape discriminator proposed in HMR
        """
        super(Discriminator, self).__init__()

        self.num_joints = 23
        # poses_alone
        self.D_conv1 = nn.Conv2d(9, 32, kernel_size=1)
        nn.init.xavier_uniform_(self.D_conv1.weight)
        nn.init.zeros_(self.D_conv1.bias)
        self.relu = nn.ReLU(inplace=True)
        self.D_conv2 = nn.Conv2d(32, 32, kernel_size=1)
        nn.init.xavier_uniform_(self.D_conv2.weight)
        nn.init.zeros_(self.D_conv2.bias)
        pose_out = []
        for i in range(self.num_joints):
            pose_out_temp = nn.Linear(32, 1)
            nn.init.xavier_uniform_(pose_out_temp.weight)
            nn.init.zeros_(pose_out_temp.bias)
            pose_out.append(pose_out_temp)
        self.pose_out = nn.ModuleList(pose_out)

        # betas
        self.betas_fc1 = nn.Linear(10, 10)
        nn.init.xavier_uniform_(self.betas_fc1.weight)
        nn.init.zeros_(self.betas_fc1.bias)
        self.betas_fc2 = nn.Linear(10, 5)
        nn.init.xavier_uniform_(self.betas_fc2.weight)
        nn.init.zeros_(self.betas_fc2.bias)
        self.betas_out = nn.Linear(5, 1)
        nn.init.xavier_uniform_(self.betas_out.weight)
        nn.init.zeros_(self.betas_out.bias)

        # poses_joint
        self.D_alljoints_fc1 = nn.Linear(32*self.num_joints, 1024)
        nn.init.xavier_uniform_(self.D_alljoints_fc1.weight)
        nn.init.zeros_(self.D_alljoints_fc1.bias)
        self.D_alljoints_fc2 = nn.Linear(1024, 1024)
        nn.init.xavier_uniform_(self.D_alljoints_fc2.weight)
        nn.init.zeros_(self.D_alljoints_fc2.bias)
        self.D_alljoints_out = nn.Linear(1024, 1)
        nn.init.xavier_uniform_(self.D_alljoints_out.weight)
        nn.init.zeros_(self.D_alljoints_out.bias)


    def forward(self, poses: torch.Tensor, betas: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the discriminator.
        Args:
            poses (torch.Tensor): Tensor of shape (B, 23, 3, 3) containing a batch of SMPL body poses (excluding the global orientation).
            betas (torch.Tensor): Tensor of shape (B, 10) containign a batch of SMPL beta coefficients.
        Returns:
            torch.Tensor: Discriminator output with shape (B, 25)
        """
        #import ipdb; ipdb.set_trace()
        #bn = poses.shape[0]
        # poses B x 207
        #poses = poses.reshape(bn, -1)
        # poses B x num_joints x 1 x 9
        poses = poses.reshape(-1, self.num_joints, 1, 9)
        bn = poses.shape[0]
        # poses B x 9 x num_joints x 1
        poses = poses.permute(0, 3, 1, 2).contiguous()

        # poses_alone
        poses = self.D_conv1(poses)
        poses = self.relu(poses)
        poses = self.D_conv2(poses)
        poses = self.relu(poses)

        poses_out = []
        for i in range(self.num_joints):
            poses_out_ = self.pose_out[i](poses[:, :, i, 0])
            poses_out.append(poses_out_)
        poses_out = torch.cat(poses_out, dim=1)

        # betas
        betas = self.betas_fc1(betas)
        betas = self.relu(betas)
        betas = self.betas_fc2(betas)
        betas = self.relu(betas)
        betas_out = self.betas_out(betas)

        # poses_joint
        poses = poses.reshape(bn,-1)
        poses_all = self.D_alljoints_fc1(poses)
        poses_all = self.relu(poses_all)
        poses_all = self.D_alljoints_fc2(poses_all)
        poses_all = self.relu(poses_all)
        poses_all_out = self.D_alljoints_out(poses_all)

        disc_out = torch.cat((poses_out, betas_out, poses_all_out), 1)
        return disc_out
