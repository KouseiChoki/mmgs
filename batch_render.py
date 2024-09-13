
from render import *
from file_utils import jhelp_folder
import sys

def read_txt(path):
    with open(path,'r') as f:
        result = f.readlines()
    return result



def render_one(model_path, views, gaussians, pipeline, background,baseline_distance=0,output_name='ours',mid_num=-1):
    ja_prev = None
    # mid_num = len(views)//2
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :].cpu().detach().numpy().transpose(1,2,0)
        rendered = rendering.cpu().detach().numpy().transpose(1,2,0)
        if idx == mid_num:
            write(rendered, output_name.format('rendered'))
            write(gt, output_name.format('gt'))
        if baseline_distance!=0:
            view.T[0] -= baseline_distance
            view.world_view_transform = torch.tensor(getWorld2View2(view.R, view.T, view.trans, view.scale)).transpose(0, 1).cuda()
            view.projection_matrix = getProjectionMatrix(znear=view.znear, zfar=view.zfar, fovX=view.FoVx, fovY=view.FoVy).transpose(0,1).cuda()
            view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
            view.camera_center = view.world_view_transform.inverse()[3, :3]
            bd_rendering = render(view, gaussians, pipeline, background)["render"]
            bd_rendered = bd_rendering.cpu().detach().numpy().transpose(1,2,0)
            if idx == mid_num:
                write(bd_rendered, output_name.format(f'baseline_distance_{baseline_distance}'))
        

def render_ja(model_path, views, gaussians, pipeline, background,judder_angle=0,output_name='ours'):
    if len(views) < 2:
        return
    ja_prev = np.concatenate((rotmat2qvec(np.transpose(views[0].R)), views[0].T))
    ja_view = views[1]
    extr = ja_ajust(ja_prev,np.concatenate((rotmat2qvec(np.transpose(ja_view.R.copy())),ja_view.T.copy())),judder_angle)
    ja_view.R = np.transpose(qvec2rotmat(extr[:4]))
    ja_view.T = np.array(extr[4:])
    ja_view.world_view_transform = torch.tensor(getWorld2View2(ja_view.R, ja_view.T, ja_view.trans, ja_view.scale)).transpose(0, 1).cuda()
    ja_view.projection_matrix = getProjectionMatrix(znear=ja_view.znear, zfar=ja_view.zfar, fovX=ja_view.FoVx, fovY=ja_view.FoVy).transpose(0,1).cuda()
    ja_view.full_proj_transform = (ja_view.world_view_transform.unsqueeze(0).bmm(ja_view.projection_matrix.unsqueeze(0))).squeeze(0)
    ja_view.camera_center = ja_view.world_view_transform.inverse()[3, :3]
    ja_rendering = render(ja_view, gaussians, pipeline, background)["render"]
    ja_rendered = ja_rendering.cpu().detach().numpy().transpose(1,2,0)
    write(ja_rendered, output_name.format(f'ja_{judder_angle}'))

def render_sets_mid(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,baseline_distance=0,judder_angle=0,output_name='ours',cur=-1):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        render_one(dataset.model_path, scene.getTrainCameras(), gaussians, pipeline, background,baseline_distance,output_name,cur)
        if judder_angle!=0:
            render_ja(dataset.model_path, scene.getTestCameras(), gaussians, pipeline, background,judder_angle,output_name)
        # if not skip_test:
        #      render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background,output_name)

if __name__ == '__main__':
    # Parse command-line arguments --root /home/rg0775/QingHong/MM/3dgs/output/0902/ --output /home/rg0775/QingHong/MM/3dgs/output/0904 --baseline_distance 0.01 --judder_angle 360  
    parser = ArgumentParser(description="batch render script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--format", default='png', type=str)
    parser.add_argument("--baseline_distance", default=0, type=float)
    parser.add_argument("--judder_angle","--ja", default=0, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--root", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)

    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_ = parser.parse_args(cmdlne_string)
    print("Rendering " + args_.root)
    safe_state(args_.quiet)
    folders = jhelp_folder(args_.root)
    assert len(folders)>0,'error root'
    if len(folders) == 1 and os.path.basename(folders[0]) == 'point_cloud':
        folders = [args_.root]
    for folder in folders:
        source_ = os.path.join(folder,'source_path.txt')
        assert os.path.isfile(source_),f'can not find source_path.txt,please check your data{source_}'
        tmp = read_txt(source_)
        source,name,cur = tmp[0].rstrip(),tmp[1].rstrip(),int(tmp[2].rstrip())
        name += f'.{args_.format}'
        cmdlne_string_step = cmdlne_string.copy()
        cmdlne_string_step.append('--model_path')
        cmdlne_string_step.append(folder)
        cmdlne_string_step.append('--source_path')
        cmdlne_string_step.append(source)
        args = get_combined_args(parser,cmdlne_string_step)
        render_sets_mid(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test,args.baseline_distance,args.judder_angle,os.path.join(args.output,'{}',name),cur)