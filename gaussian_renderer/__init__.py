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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

# def render(viewpoint_camera, pc, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None,render_bg=None):
#     """
#     Render the scene. 
    
#     Background tensor (bg_color) must be on GPU!
#     """
#     if render_bg is None:
#         xyz = pc.get_xyz
#         features = pc.get_features
#         opacity = pc.get_opacity
#         scales = pc.get_scaling
#         rotations = pc.get_rotation
#     else:
#         if render_bg:
#             xyz = pc.get_xyz_bg
#             features = pc.get_features_bg
#             opacity = pc.get_opacity_bg
#             scales = pc.get_scaling_bg
#             rotations = pc.get_rotation_bg
#         else:
#             xyz = pc.get_xyz_fg
#             features = pc.get_features_fg
#             opacity = pc.get_opacity_fg
#             scales = pc.get_scaling_fg
#             rotations = pc.get_rotation_fg
#             # print(scales.min(),scales.mean())
#     # scales = pc.get_scaling_bg if render_bg else pc.get_scaling
#     # rotations = pc.get_rotation_bg if render_bg else pc.get_rotation
#     # opacity = pc.get_opacity_bg if render_bg else pc.get_opacity
#     # xyz = pc.get_xyz_bg if render_bg else pc.get_xyz
#     # features = pc.get_features_bg if render_bg else pc.get_features
 
#     # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
#     screenspace_points = torch.zeros_like(xyz, dtype=xyz.dtype, requires_grad=True, device="cuda") + 0
#     try:
#         screenspace_points.retain_grad()
#     except:
#         pass

#     # Set up rasterization configuration
#     tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
#     tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

#     raster_settings = GaussianRasterizationSettings(
#         image_height=int(viewpoint_camera.image_height),
#         image_width=int(viewpoint_camera.image_width),
#         tanfovx=tanfovx,
#         tanfovy=tanfovy,
#         bg=bg_color,
#         scale_modifier=scaling_modifier,
#         viewmatrix=viewpoint_camera.world_view_transform,
#         projmatrix=viewpoint_camera.full_proj_transform,
#         sh_degree=pc.active_sh_degree,
#         campos=viewpoint_camera.camera_center,
#         prefiltered=False,
#         debug=pipe.debug
#     )

#     rasterizer = GaussianRasterizer(raster_settings=raster_settings)

#     means3D = xyz
#     means2D = screenspace_points

#     # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
#     # scaling / rotation by the rasterizer.
#     # scales = None
#     # rotations = None
#     cov3D_precomp = None
#     # if pipe.compute_cov3D_python:
#     #     cov3D_precomp = pc.get_covariance(scaling_modifier)
#     # else:
#     #     scales = pc.get_scaling_bg if render_bg else pc.get_scaling
#     #     rotations = pc.get_rotation_bg if render_bg else pc.get_rotation

#     # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
#     # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
#     shs = None
#     colors_precomp = None
#     if override_color is None:
#         if pipe.convert_SHs_python:
#             shs_view = features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
#             dir_pp = (xyz - viewpoint_camera.camera_center.repeat(features.shape[0], 1))
#             dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
#             sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
#             colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
#         else:
#             shs = features
#     else:
#         colors_precomp = override_color

#     # Rasterize visible Gaussians to image, obtain their radii (on screen). 
#     rendered_image, radii = rasterizer(
#         means3D = means3D, #2.1894
#         means2D = means2D, #0
#         shs = shs, #-0.0891
#         colors_precomp = colors_precomp,
#         opacities = opacity, #0.1000
#         scales = scales, #0.0054
#         rotations = rotations, #0.2500
#         cov3D_precomp = cov3D_precomp)

#     # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
#     # They will be excluded from value updates used in the splitting criteria.
#     return {"render": rendered_image,
#             "viewspace_points": screenspace_points,
#             "visibility_filter" : radii > 0,
#             "radii": radii}

def render(viewpoint_camera, pc , pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None,render_bg=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    if render_bg is None:
        get_xyz = pc.get_xyz
        get_features = pc.get_features
        get_opacity = pc.get_opacity
        get_scaling = pc.get_scaling
        get_rotation = pc.get_rotation
    else:
        if render_bg:
            get_xyz = pc.get_xyz_bg
            get_features = pc.get_features_bg
            get_opacity = pc.get_opacity_bg
            get_scaling = pc.get_scaling_bg
            get_rotation = pc.get_rotation_bg
        else:
            get_xyz = pc.get_xyz_fg
            get_features = pc.get_features_fg
            get_opacity = pc.get_opacity_fg
            get_scaling = pc.get_scaling_fg
            get_rotation = pc.get_rotation_fg
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(get_xyz, dtype=get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = get_xyz
    means2D = screenspace_points
    opacity = get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = get_scaling
        rotations = get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (get_xyz - viewpoint_camera.camera_center.repeat(get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
