import os
from file_utils import read,jhelp_file
from scene.fbx_utils import fbx_reader
import numpy as np
import math
from typing import NamedTuple
from tqdm import tqdm
from plyfile import PlyData, PlyElement

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

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def cal_qvec(data):
    from scipy.spatial.transform import Rotation as R
    rx,ry,rz,tx,ty,tz = data
    rotation_matrix = R.from_euler('ZYX', [rx,ry,rz],degrees=True).as_matrix()
    c2w = np.eye(4,4)
    c2w[:3,:3] = rotation_matrix
    translation_vector = [tx,ty,tz]
    c2w[:3,-1] = translation_vector
     # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
    c2w[:3, 1:3] *= -1
    # get the world-to-camera transform and set R, T
    w2c = np.linalg.inv(c2w)
    R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
    T = w2c[:3, 3]
    return R,T

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
        return
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
        R,T = cal_qvec(extr)

        FovY = focal2fov(intr['fx'], height)
        FovX = focal2fov(intr['fy'], width)

        image_path = images[uid]
        image_name = os.path.basename(image_path).split(".")[0]
        # image = Image.open(image_path)
        # mask = Image.open(masks[idx]) if mask_enable else None
        image = read(image_path,type='ldr')
        mask = read(masks[uid],type='mask') if mask_enable else None

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,mask=mask,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    # sys.stdout.write('\n')
    return cam_infos

def readFbxSceneInfo(path, eval, llffhold=8,step=1):
    try:
        fbx,images_folder,mask_folder,ply_path = generate_data_from_fbx(path)
    except:
        raise ValueError('not found FBX file')
    cam_infos_unsorted = readFbxCameras(fbx,images_folder,mask_folder)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

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
    pcd = fetchPly(ply_path)
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

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos


# path = '/home/rg0775/QingHong/MM/3dgs/mydata/UE_XYPitchYaw'
# # fbx,images_folder,mask_folder,ply_file = generate_data_from_fbx(path)
# readFbxSceneInfo(path,False)