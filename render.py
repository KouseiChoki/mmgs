#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from file_utils import write
from utils.graphics_utils import getWorld2View2,getProjectionMatrix
from copy import deepcopy
import numpy as np 

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec



def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def slerp(q1, q2, t):
    """Perform spherical linear interpolation (slerp) between two quaternions."""
    # Convert to numpy arrays
    q1 = np.array(q1)
    q2 = np.array(q2)
    
    # Compute the cosine of the angle between the two vectors.
    dot_product = np.dot(q1, q2)
    
    # If the dot product is negative, slerp won't take the shorter path.
    # Note that q and -q represent the same rotation, but may produce different slerp.
    if dot_product < 0.0:
        q2 = -q2
        dot_product = -dot_product
    
    # Clamp dot_product to be in the range of [0.0, 1.0]
    dot_product = np.clip(dot_product, 0.0, 1.0)
    
    # Calculate the angle between the quaternions
    theta_0 = np.arccos(dot_product)  # theta_0 = angle between input vectors
    sin_theta_0 = np.sin(theta_0)     # compute this value only once
    
    # If the angle is small, use linear interpolation
    if sin_theta_0 < 1e-6:
        return (1.0 - t) * q1 + t * q2
    
    # Compute the actual interpolation factor
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    
    s0 = np.cos(theta) - dot_product * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    
    return (s0 * q1) + (s1 * q2)


def interpolate_translation(t1, t2, t=0.5):
    """Perform linear interpolation between two translations."""
    return (1 - t) * np.array(t1) + t * np.array(t2)

def ja_ajust(prev_extrinsic,extrinsic,judder_angle):
    ja =  judder_angle/720
    # prev_extrinsic = prev_image_info.extrinsic
    # extrinsic = image_info.extrinsic
    new_q = slerp(prev_extrinsic[:4],extrinsic[:4],t=ja)
    new_t = interpolate_translation(prev_extrinsic[4:],extrinsic[4:],t=ja)
    #change uid and name
    new_extrinsic=np.concatenate([new_q,new_t])
    return new_extrinsic

# -r 1 --name normal  -s /home/rg0775/QingHong/MM/3dgs/mydata/1119_to_1127_step_2_cur_2 -m /home/rg0775/QingHong/MM/3dgs/output/0821 --judder_angle 360
def render_set(model_path, name, iteration, views, gaussians, pipeline, background,baseline_distance=0,judder_angle=0,output_name='ours'):
    render_path = os.path.join(model_path, name, "{}_{}".format(output_name,iteration), "renders")
    gts_path = os.path.join(model_path, name, "{}_{}".format(output_name,iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    if baseline_distance!=0:
        bd_path = os.path.join(model_path, name, "{}_{}".format(output_name,iteration), f"baseline_distance_{baseline_distance}")
        makedirs(bd_path, exist_ok=True)
    
    if judder_angle!=0:
        ja_path = os.path.join(model_path, name, "{}_{}".format(output_name,iteration), f"judder_angle_{judder_angle}")
        makedirs(ja_path, exist_ok=True)
        ja_prev = None

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :].cpu().detach().numpy().transpose(1,2,0)
        rendered = rendering.cpu().detach().numpy().transpose(1,2,0)
        # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        write(rendered, os.path.join(render_path, '{0:05d}'.format(idx) + f".{args.format}"))
        # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        write(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + f".{args.format}"))
        if judder_angle!=0:
            if ja_prev is None:
                ja_prev = np.concatenate((rotmat2qvec(np.transpose(view.R.copy())),view.T.copy()))
            else:
                ja_view = deepcopy(view)
                extr = ja_ajust(ja_prev,np.concatenate((rotmat2qvec(np.transpose(view.R.copy())),view.T.copy())),judder_angle)
                ja_view.R = np.transpose(qvec2rotmat(extr[:4]))
                ja_view.T = np.array(extr[4:])
                ja_view.world_view_transform = torch.tensor(getWorld2View2(ja_view.R, ja_view.T, ja_view.trans, ja_view.scale)).transpose(0, 1).cuda()
                ja_view.projection_matrix = getProjectionMatrix(znear=ja_view.znear, zfar=ja_view.zfar, fovX=ja_view.FoVx, fovY=ja_view.FoVy).transpose(0,1).cuda()
                ja_view.full_proj_transform = (ja_view.world_view_transform.unsqueeze(0).bmm(ja_view.projection_matrix.unsqueeze(0))).squeeze(0)
                ja_view.camera_center = ja_view.world_view_transform.inverse()[3, :3]
                ja_rendering = render(ja_view, gaussians, pipeline, background)["render"]
                ja_rendered = ja_rendering.cpu().detach().numpy().transpose(1,2,0)
                write(ja_rendered, os.path.join(ja_path, '{0:05d}'.format(idx-1)+'_to_{0:05d}'.format(idx) + f".{args.format}"))
                ja_prev = np.concatenate((rotmat2qvec(np.transpose(view.R.copy())),view.T.copy()))

        if baseline_distance!=0:
            view.T[0] -= baseline_distance
            view.world_view_transform = torch.tensor(getWorld2View2(view.R, view.T, view.trans, view.scale)).transpose(0, 1).cuda()
            view.projection_matrix = getProjectionMatrix(znear=view.znear, zfar=view.zfar, fovX=view.FoVx, fovY=view.FoVy).transpose(0,1).cuda()
            view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
            view.camera_center = view.world_view_transform.inverse()[3, :3]
            bd_rendering = render(view, gaussians, pipeline, background)["render"]
            bd_rendered = bd_rendering.cpu().detach().numpy().transpose(1,2,0)
            write(bd_rendered, os.path.join(bd_path, '{0:05d}'.format(idx) + f".{args.format}"))
        

        #     view.world_view_transform[0,-1] += baseline_distance
        #     view.camera_center[0] += baseline_distance
        #     rendering_right = render(view, gaussians, pipeline, background)["render"]
        #     torchvision.utils.save_image(rendering_right, os.path.join(right_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,baseline_distance=0,judder_angle=0,output_name='ours'):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "render_result", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background,baseline_distance,judder_angle,output_name)

        # if not skip_test:
        #      render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background,output_name)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--name", default='ours', type=str)
    parser.add_argument("--format", default='png', type=str)
    parser.add_argument("--baseline_distance", default=0, type=float)
    parser.add_argument("--judder_angle", default=0, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test,args.baseline_distance,args.judder_angle,args.name)