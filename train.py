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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss,l1_loss_mask, ssim,ssim_mask
from gaussian_renderer import render, network_gui
import sys
# from scene import Scene, GaussianModel
from scene import KouseiScene as Scene
from scene import KouseiGaussianModel as GaussianModel

from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from myutil import mask_adjust
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def print_tensor_memory():
    snapshot = torch.cuda.memory_snapshot()
    for segment in snapshot:
        if segment["state"] == "active_allocated":
            print(f"Tensor: {segment['addr']} | Size: {segment['requested_size'] / 1024**2:.2f} MB")
    
# torch.cuda.memory_snapshot()
# -r 1 -s /home/rg0775/QingHong/MM/3dgs/mydata/1119_to_1123  --output 0815 --cur 3 --iterations 3000
# -r 1 -s /home/rg0775/QingHong/MM/3dgs/mydata/1119_to_1121_cur2  --output 0815  --iterations 3000    
# -r 1 -s /home/rg0775/QingHong/MM/3dgs/mydata/1119_to_1121_cur2  --output 0815  --iterations 3000  --cur 2  
# 


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,output):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset,output)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    if False:
        scene.save(1)
        sys.exit(0)

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    cur = 0
    for iteration in range(first_iter, opt.iterations + 1):       
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            cur = 0
        # viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        viewpoint_cam = viewpoint_stack.pop(0)
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_step = [None]
        if hasattr(viewpoint_cam,'mask') and viewpoint_cam.mask is not None:
            if args.cur>0:
                if args.cur != cur: #背景
                    render_step = [True]
                else: #背景+前景
                    render_step = [True,None]
            else:
                render_step = [True]
        if only_fg:
            render_step = [False]
        for render_bg in render_step:
            # with torch.no_grad():
            #     gaussians._scaling[gaussians.bg_num:].clamp_(max=0)
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg , render_bg=render_bg)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            # if iteration >= 100 and render_bg:
            #     import cv2
            #     import numpy as np 
            #     tmp = np.transpose(image.detach().cpu().numpy(),(1,2,0)) *255
            #     cv2.imwrite('/home/rg0775/QingHong/MM/3dgs/1.png',tmp.astype('uint8')[...,::-1])
            #     tmp = np.transpose(viewpoint_cam.original_image.detach().cpu().numpy(),(1,2,0)) *255
            #     cv2.imwrite('/home/rg0775/QingHong/MM/3dgs/2.png',tmp.astype('uint8')[...,::-1])
            #     tmp = viewpoint_cam.mask.detach().cpu().numpy().astype('uint8') *255
            #     tmp = np.repeat(tmp[..., np.newaxis], 3, axis=-1)
            #     cv2.imwrite('/home/rg0775/QingHong/MM/3dgs/3.png',tmp.astype('uint8')[...,::-1])

            # Loss
            gt_image = viewpoint_cam.original_image.float().cuda()
            # Mask loss
            if render_bg is not None:
                if not render_bg: #fg loss
                    fg_mask = viewpoint_cam.mask
                    if False:
                        tmp_mask = fg_mask.detach().cpu().numpy().astype('float32')
                        tmp_mask = mask_adjust(tmp_mask,-10)
                        fg_mask = torch.tensor(tmp_mask).bool().to(viewpoint_cam.mask.device)
                    Ll1 = l1_loss_mask(image, gt_image,~viewpoint_cam.mask) 
                    # mask_gt_image = gt_image.copy()
                    # print(mask_gt_image.shape)
                    # mask_gt_image[:,~viewpoint_cam.mask] 
                    # Ll1 = l1_loss(image, mask_gt_image) 
                else: #bg loss
                    Ll1 = l1_loss_mask(image, gt_image,viewpoint_cam.mask) 
                loss = Ll1 
            else:
                Ll1 = l1_loss(image, gt_image) 
                loss = Ll1
                # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

            cur += 1
            loss.backward()
            
            iter_end.record()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                # Log and save
                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
                if (iteration in saving_iterations):
                    # if render_bg is not None and not render_bg:
                    #     pass
                    # else:
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)
                    sys.exit(0)
                # print(gaussians._scaling[gaussians.bg_num:].max(),gaussians._scaling[gaussians.bg_num:].mean())
                # Densification
                if iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    if render_bg is None:
                        max_radii2D = gaussians.max_radii2D
                    else:
                        if render_bg:
                            max_radii2D = gaussians.max_radii2D[:gaussians.bg_num]
                        else:
                            max_radii2D = gaussians.max_radii2D[gaussians.bg_num:]
                    max_radii2D[visibility_filter] = torch.max(max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter,bg = render_bg)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

                # Optimizer step
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

                # if (iteration in checkpoint_iterations):
                #     print("\n[ITER {}] Saving Checkpoint".format(iteration))
                #     torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            
def prepare_output_and_logger(args,output):    
    print(args)
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", output)
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    # -r 1 -s /home/rg0775/QingHong/MM/3dgs/mydata/1121fh  --output 0810fh  
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="10.35.116.93")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--cur', type=int, default = -1)
    parser.add_argument('--enhance', type=int, default = 1)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--only_fg", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--output", type=str, default = '01')

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.iterations not in args.save_iterations:
        args.save_iterations.insert(0,args.iterations)
    # print("Optimizing " + args.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet)
    if args.cur<0:
        import re
        match = re.search(r'cur_(\d+)', args.source_path)
        if match is not None:
            args.cur = int(match.group(1))
        print(f'CF frame is :{args.cur}')
    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from,args.output)

    # All done
    print("\nTraining complete.")
