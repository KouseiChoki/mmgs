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

def render_set(model_path, name, iteration, views, gaussians, pipeline, background,baseline_distance=0,output_name='ours'):
    render_path = os.path.join(model_path, name, "{}_{}".format(output_name,iteration), "renders")
    gts_path = os.path.join(model_path, name, "{}_{}".format(output_name,iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    # if baseline_distance:
    #     right_path = os.path.join(model_path, name, "{}_{}".format(output_name,iteration), "right_renders")
    #     makedirs(right_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :].cpu().detach().numpy().transpose(1,2,0)
        rendered = rendering.cpu().detach().numpy().transpose(1,2,0)
        # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        write(rendered, os.path.join(render_path, '{0:05d}'.format(idx) + f".{args.format}"))
        # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        write(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + f".{args.format}"))
        # if baseline_distance:
        #     view.world_view_transform[0,-1] += baseline_distance
        #     view.camera_center[0] += baseline_distance
        #     rendering_right = render(view, gaussians, pipeline, background)["render"]
        #     torchvision.utils.save_image(rendering_right, os.path.join(right_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,baseline_distance=0,output_name='ours'):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "render_result", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background,baseline_distance,output_name)

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
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test,args.baseline_distance,args.name)