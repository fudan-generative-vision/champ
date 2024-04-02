import cv2
from pathlib import Path
import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import pyrender
from pathlib import Path

if 'PYOPENGL_PLATFORM' not in os.environ:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
from hmr2.models import HMR2, download_models, load_hmr2, DEFAULT_CHECKPOINT

if __name__ == "__main__":
    # Replace 'input_video_path' and 'output_video_path' with the actual paths to your video files
    #input_video_path = 'input_video.mp4'
    #output_video_path = '00337_transfer_test/output.mp4'
    parser = argparse.ArgumentParser(description='HMR2 demo code')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    parser.add_argument('--driving_path', type=str, default="driving_videos/001", help='Folder path to driving imgs sequence')
    parser.add_argument('--reference_path', type=str, default="reference_imgs/images/ref.png", help='Path to reference img')
    parser.add_argument('--output_folder', type=str, default="output", help='Path to result imgs')
    parser.add_argument('--figure_transfer', dest='figure_transfer', action='store_true', default=False, help='If true, transfer SMPL shape parameter.')
    parser.add_argument('--view_transfer', dest='view_transfer', action='store_true', default=False, help='If true, transfer camera parameter.')

    args = parser.parse_args()
    
    os.makedirs(args.output_folder, exist_ok=True)
    #reference_img = cv2.imread(str(args.reference_file_folder))
    have_smpl_results = False
    
    
    model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)
    model = model.to(args.device)
    
    os.makedirs(os.path.join(args.output_folder), exist_ok=True)
    os.makedirs(os.path.join(args.output_folder, "visualized_imgs"), exist_ok=True)
    # os.makedirs(os.path.join(args.output_folder,"mesh"), exist_ok=True)
    os.makedirs(os.path.join(args.output_folder,"mask"), exist_ok=True)
    os.makedirs(os.path.join(args.output_folder,"semantic_map"), exist_ok=True)
    os.makedirs(os.path.join(args.output_folder,"images"), exist_ok=True)
    os.makedirs(os.path.join(args.output_folder,"normal"), exist_ok=True)
    os.makedirs(os.path.join(args.output_folder,"depth"), exist_ok=True)
    os.makedirs(os.path.join(args.output_folder,"smpl_results"), exist_ok=True)
    
    driving_folder = args.driving_path
    reference_file = args.reference_path
    
    print(os.listdir(driving_folder))
    if "smpl_results" in os.listdir(driving_folder):
        have_smpl_results = True
        driving_paths = os.listdir(os.path.join(driving_folder, "smpl_results"))
        driving_paths = [path for path in driving_paths if os.path.splitext(path)[1].lower() == ".npy"]
        driving_paths.sort(key=lambda x: int(x.split('.')[0]))
        driving_paths = [os.path.join(driving_folder, "smpl_results", path) for path in driving_paths]
    if not have_smpl_results:  
        print("No SMPLS found in driving folder.")
    else:
        
        referece_dict = np.load(str(reference_file), allow_pickle=True).item()
        reference_path = Path(str(reference_file))
        reference_img = cv2.imread(os.path.join(reference_path.parent.parent, "images", reference_path.name.split(".")[0]+".png") )
        
        #print(driving_folder)
        #print(driving_paths)
        smooth_smpl_path = os.path.join(driving_folder, "smpl_results", "smpls_group.npz")
        if os.path.exists(smooth_smpl_path):
            result_dict_list = np.load(smooth_smpl_path, allow_pickle=True)
            result_dict_first = np.load(driving_paths[0], allow_pickle=True).item()
            i = 0
            for smpl_outs, cam_t, file_path in tqdm(zip(result_dict_list["smpl"], result_dict_list["camera"], driving_paths)):
                img_fn, _ = os.path.splitext(os.path.basename(file_path))
                result_dict = {key: value for key, value in result_dict_first.items()}
                result_dict["smpls"] = smpl_outs
                result_dict["cam_t"] = cam_t
                if args.view_transfer:
                    scaled_focal_length = referece_dict["scaled_focal_length"]
                    result_dict["cam_t"] = referece_dict["cam_t"]
                    result_dict["scaled_focal_length"] = scaled_focal_length
                # transfer reference SMPL shape to driving SMPLs
                if args.figure_transfer:
                    result_dict["smpls"]['betas'] = referece_dict['smpls']['betas']
                    
                smpl_output = model.smpl(**{k: torch.Tensor(v[[0]]).to(args.device).float() for k,v in result_dict["smpls"].items()}, pose2rot=False)
                pred_vertices = smpl_output.vertices
                result_dict["verts"][0] = pred_vertices.reshape(-1, 3).detach().cpu().numpy()
                result_dict["render_res"] =  referece_dict["render_res"]
                if i == 0:
                    cv2.imwrite(os.path.join(args.output_folder, "reference_img", f'{img_fn}.png'), reference_img)
                np.save(str(os.path.join(args.output_folder,"smpl_results", f'{img_fn}.npy')),
                    result_dict)
                i += 1