import os
from file_utils import read,jhelp_file,mkdir
from scene.fbx_utils import fbx_reader
import numpy as np
import math
from typing import NamedTuple
from tqdm import tqdm
from plyfile import PlyData, PlyElement
import scipy

MAX_DEPTH = 1e6

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    mask: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    intr:np.array
    extr:np.array

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    fg_point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


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


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    # normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    normals = np.vstack([
    vertices['nx'] if 'nx' in vertices else vertices['normal_x'],
    vertices['ny'] if 'ny' in vertices else vertices['normal_y'],
    vertices['nz'] if 'nz' in vertices else vertices['normal_z']
]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def cal_qvec(data):
    from scipy.spatial.transform import Rotation as R
    rx,ry,rz,tx,ty,tz = data
    rotation_matrix = R.from_euler('xyz', [rx,ry,rz],degrees=True).as_matrix()
    c2w = np.eye(4,4)
    c2w[:3,:3] = rotation_matrix
    translation_vector = [tx,ty,tz]
    c2w[:3,-1] = translation_vector
    # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
    c2w[:3, 1:3] *= -1
    return c2w


def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])
        
    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def check_fbx_valid(root,keyword):
    fbx_file = None
    for dir_name, _, file_list in os.walk(root):
        if fbx_file is None:
            for tmp in file_list:
                if keyword in tmp:
                    fbx_file = os.path.join(dir_name,tmp)
                    break
    return fbx_file

def generate_data_from_fbx(root,keyword='_pw_fmt.fbx'):
    fbx_file = check_fbx_valid(root,keyword)
    if fbx_file is None:
        raise ValueError('fbx file not found')
    image_dir,mask_dir,ply_file = '','',''
    for dir_name, subdir_list, file_list in os.walk(root):
        for tmp in subdir_list:
            if tmp.lower() in ['image','images']:
                image_dir = os.path.join(dir_name,tmp)
            if tmp.lower() in ['mask','masks']:
                mask_dir = os.path.join(dir_name,tmp)
        for tmp in file_list:
            if '.ply' in tmp:
                ply_file = os.path.join(dir_name,tmp)
    return fbx_file,image_dir,mask_dir,ply_file

def read_fbx(fbx_file,images_folder):
    instrinsics = {}
    instrinsics['h'],instrinsics['w'] = read(jhelp_file(images_folder)[0],type='image').shape[:2]
    extrinsics,[fw,fh] = fbx_reader(fbx_file)
    focal_length_x = instrinsics['w']  * fw
    focal_length_y = instrinsics['h']  * fh
    instrinsics['fx'],instrinsics['fy'] = focal_length_x,focal_length_y
    return instrinsics,extrinsics

def readFbxCameras(fbx,images_folder,mask_folder):
    cam_infos = []
    images = jhelp_file(images_folder)
    mask_enable = os.path.isdir(mask_folder)
    masks = jhelp_file(mask_folder) if mask_enable else None
    if masks is not None and len(masks) ==0:
        mask_enable = False
    cam_intrinsics,cam_extrinsics = read_fbx(fbx,images_folder)
    for uid in tqdm(range(min(len(cam_extrinsics),len(images))),desc='reading fbx format'):
        extr = cam_extrinsics[uid]
        intr = cam_intrinsics
        height = intr['h']
        width = intr['w']
        c2w = cal_qvec(extr)
        w2c = np.linalg.inv(c2w)
        qx, qy, qz ,qw = scipy.spatial.transform.Rotation.from_matrix(w2c[:3, :3]).as_quat()
        R = np.transpose(qvec2rotmat([qw,qx,qy,qz]))
        T = w2c[:3, 3]
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        # c2w_colmap = c2w.copy()
        # # c2w_colmap[:3, 1:3] *= -1
        # R = c2w_colmap[:3,:3]
        # T = c2w_colmap[:3, 3]
        FovY = focal2fov(intr['fx'], height)
        FovX = focal2fov(intr['fy'], width)
        
        image_path = images[uid]
        z = os.path.basename(image_path)
        image_name = z[:-z[::-1].find('.')-1]
        image = read(image_path,type='ldr')
        mask = read(masks[uid],type='mask') if mask_enable else None

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,mask=mask,
                              image_path=image_path, image_name=image_name, width=width,height=height,intr=intr,extr=c2w)
        cam_infos.append(cam_info)
    # sys.stdout.write('\n')
    return cam_infos


def generate_point_cloud_from_depth(depth_image, intrinsics, extrinsics,mask=None):
    h, w = depth_image.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
    # 相机内参
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    # 计算每个像素的三维坐标
    z = depth_image.astype(np.float32) # 假设深度以毫米为单位，转换为米
    x = (i - cx) * z / fx
    y = (j - cy) * z / fy
    # 将点组合成[N, 3]的点云
    points_camera = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    # 去除非法点
    if mask is not None:
        points_camera = points_camera[mask]
    # points_camera = points_camera[points_camera[:, 2] != 0]
    # 将点云从相机坐标系转换到世界坐标系
    points_world = (extrinsics[:3, :3] @ points_camera.T).T + extrinsics[:3, 3]
    return points_world

def get_ply(xyz,rgbs):
    #create pointcloud
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    # print('writing plyfile........')
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    assert xyz.shape[0] == rgbs.shape[0],'error input, please check your depth data(contains 0) or add --inverse_depth'
    attributes = np.concatenate((xyz, normals, rgbs), axis=1)
    elements[:] = list(map(tuple, attributes))
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    return ply_data

def generate_ply(images_folder,cam_infos,ply_path,down_scale=1,step=1,max_frame=1000):
    depths_folder = images_folder.replace('images','depths').replace('image','depth')
    images = jhelp_file(images_folder)[:max_frame]
    depths = jhelp_file(depths_folder)
    points,rgbs = [],[]
    for i in tqdm(range(0,len(images),step),desc='generating pointcloud..'):
        image = images[i]
        depth = depths[i]
        #intrin
        intr = cam_infos[i].intr
        w,h = intr['w'],intr['h']
        o_cx = w/2.0 
        o_cy = h/2.0
        o_cx = o_cx //down_scale
        o_cy = o_cy //down_scale
        focal_length_x = intr['fx']/down_scale
        focal_length_y = intr['fy']/down_scale
        target_w = w//down_scale
        target_h = h//down_scale
        intrinsics = np.array([[focal_length_x,0,o_cx],[0,focal_length_y,o_cy],[0,0,1]])
        #extrin
        extr = cam_infos[i].extr
        #rgb
        rgb = read(image,type='image')
        #depth
        depth = read(depth)[...,0]
        if down_scale != 1:
            import cv2
            depth = cv2.resize(depth,(target_w,target_h),interpolation=cv2.INTER_NEAREST)
            rgb = cv2.resize(rgb,(target_w,target_h))
        #pointcloud
        point = generate_point_cloud_from_depth(depth,intrinsics,extr)
        if point is not None:
            point = point.reshape(-1,3)[depth.reshape(-1)<MAX_DEPTH]
            points.append(point.reshape(-1,3))
        if rgb is not None:
            rgb = rgb.reshape(-1,3)[depth.reshape(-1)<MAX_DEPTH]
            rgbs.append(rgb.reshape(-1,3))
    print('preparing......')
    xyz = np.concatenate(points)
    rgbs = np.concatenate(rgbs)
    ply_data = get_ply(xyz,rgbs)
    mkdir(os.path.dirname(ply_path))
    ply_data.write(ply_path)
    return True

def readFbxSceneInfo(path, eval, llffhold=8,step=1):
    try:
        fbx,images_folder,mask_folder,ply_path = generate_data_from_fbx(path)
    except:
        raise ValueError('not found FBX file')
    cam_infos = readFbxCameras(fbx,images_folder,mask_folder)
    # cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        ja_images_path = os.path.join(path, "sparse/0", "ja_images.txt") 
        ja_cameras_path = os.path.join(path, "sparse/0", "ja_cameras.txt") 
        if os.path.isfile(ja_images_path) and os.path.isfile(ja_cameras_path):
            test_cam_extrinsics = read_extrinsics_text(ja_images_path)
            test_cam_intrinsics = read_intrinsics_text(ja_cameras_path)
            test_cam_infos_unsorted = readColmapCameras(cam_extrinsics=test_cam_extrinsics, cam_intrinsics=test_cam_intrinsics, images_folder=os.path.join(path, reading_dir))
            test_cam_infos = sorted(test_cam_infos_unsorted.copy(), key = lambda x : x.image_name)
        else:
            test_cam_infos = train_cam_infos[:2]

    nerf_normalization = getNerfppNorm(train_cam_infos)
    # ply_path = os.path.join(path, "sparse/0/points3D.ply")
    # bin_path = os.path.join(path, "sparse/0/points3D.bin")
    # txt_path = os.path.join(path, "sparse/0/points3D.txt")
    fg_ply_path = ply_path.replace('points3D','fg_points3D')
    # -s /home/rg0775/QingHong/MM/3dgs/mydata/UE_XYPitchYaw_nopointcloud --output 0106 -r 8
    if len(ply_path)>1:
        pcd = fetchPly(ply_path)
    else:
        ply_path = os.path.join(path,'Pointcloud','generated','points3D.ply')
        flag = generate_ply(images_folder,cam_infos,ply_path)
        if flag:
            pcd = fetchPly(ply_path)
        else:
            raise EOFError('not enough memory to generate ply file')
    try:
        fgpcd = fetchPly(fg_ply_path)
    except:
        fgpcd = None

    scene_info = SceneInfo(point_cloud=pcd,fg_point_cloud=fgpcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info
