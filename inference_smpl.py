import cv2
from pathlib import Path
import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import pyrender
if 'PYOPENGL_PLATFORM' not in os.environ:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import HMR2, download_models, load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils import recursive_to
from hmr2.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hmr2.utils.renderer import Renderer, cam_crop_to_full
from hmr2.utils.semantic_renderer import SemanticRenderer
LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

def process_frame(frame):
    # Replace this function with your processing logic
    # For demonstration purposes, we'll just resize the frame
    processed_frame = cv2.resize(frame, (640, 480))  # Resize to 640x480
    return processed_frame

def process_video(input_video_path, output_video_path):
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)
    
    # Get the video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs as well
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Process each frame of the video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Process the frame
        processed_frame = process_frame(frame)
        # Write the processed frame to the output video
        out.write(processed_frame)
    # Release the video capture and writer objects
    cap.release()
    out.release()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HMR2 demo code')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    parser.add_argument('--reference_imgs_folder', type=str, default="reference_imgs", help='Folder path to reference imgs')
    parser.add_argument('--driving_videos_folder', type=str, default="driving_videos", help='Folder path to driving videos')
    parser.add_argument('--figure_scale', type=int, default=None, help='Adjust the figure scale to better fit extreme shape')

    # parser.add_argument('--transfer_shape_to_driving', type=bool, default=True, help='If True, transfer reference shape to driving smpl. Otherwise, transfer driving poses to reference shape to 3.')

    args = parser.parse_args()
    
    os.makedirs(args.reference_imgs_folder, exist_ok=True)
    os.makedirs(os.path.join(args.reference_imgs_folder, "visualized_imgs"), exist_ok=True)
    os.makedirs(os.path.join(args.reference_imgs_folder, "mask"), exist_ok=True)
    os.makedirs(os.path.join(args.reference_imgs_folder, "semantic_map"), exist_ok=True)
    os.makedirs(os.path.join(args.reference_imgs_folder, "depth"), exist_ok=True)
    os.makedirs(os.path.join(args.reference_imgs_folder, "smpl_results"), exist_ok=True)

    reference_img_paths = [ path for path in os.listdir(os.path.join(args.reference_imgs_folder, "images"))]
    driving_videos_paths = [ path for path in os.listdir(os.path.join(args.driving_videos_folder))]
    
    model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)

    # Load detector
    from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
    from detectron2.config import LazyConfig
    import hmr2
    cfg_path = Path(hmr2.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = str(Path(hmr2.__file__).parent.parent/'detectron2'/"model_final_f05665.pkl")
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)
    

    model = model.to(args.device)
    detector.model.to(args.device)
    renderer = SemanticRenderer(model_cfg, faces=model.smpl.faces, 
                        lbs=model.smpl.lbs_weights, viewport_size=(720,720))  
    for img_path in tqdm(reference_img_paths, desc ="Processing Reference Images:"):
        img_cv2 = cv2.imread(str(os.path.join(args.reference_imgs_folder, "images", img_path)))

        renderer.renderer.delete()
        renderer.renderer = pyrender.OffscreenRenderer(viewport_width=img_cv2.shape[:2][::-1][0],
                                                viewport_height=img_cv2.shape[:2][::-1][1],
                                                point_size=1.0)
        img_fn, _ = os.path.splitext(os.path.basename(img_path))
        
        # Detect humans in image
        det_out = detector(img_cv2)

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
        boxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        
        # Run HMR2.0 on all detected humans
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        
        for batch in dataloader:
            batch = recursive_to(batch, args.device)
            with torch.no_grad():
                out = model(batch)

            pred_cam = out['pred_cam']
            pred_smpl_parameter = out['pred_smpl_params']
            # transfer reference SMPL shape to driving SMPLs
            if args.figure_scale is not None:
                pred_smpl_parameter['betas'][0][1] = args.figure_scale
                print("Adjust the figure scale to:", args.figure_scale)
            
            
            smpl_output = model.smpl(**{k: v.float() for k,v in pred_smpl_parameter.items()}, pose2rot=False)
            pred_vertices = smpl_output.vertices
            out['pred_vertices'] = pred_vertices.reshape(batch['img'].shape[0], -1, 3)
            
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()
            # Render the result
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                # Add all verts and cams to list
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                cam_t = pred_cam_t_full[n]
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            rendering_results_dict = renderer.render_all_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], **misc_args)
            # Overlay image
            valid_mask = rendering_results_dict["Image"][:,:,-1][:, :, np.newaxis]
            cam_view = valid_mask * rendering_results_dict["Image"][:,:,[2,1,0]] +  (1 - valid_mask) * img_cv2.astype(np.float32)[:,:,::-1] / 255
            smpl_outs = {k: v.detach().cpu().numpy() for k, v in pred_smpl_parameter.items()}
            results_dict_for_rendering = {"verts":all_verts, "cam_t":all_cam_t, 
                "render_res":img_size[n].cpu().numpy(), "smpls":smpl_outs,
                "scaled_focal_length":scaled_focal_length.cpu().numpy()}
            cv2.imwrite(os.path.join(args.reference_imgs_folder, "visualized_imgs", f'{img_fn}.png'), 255*cam_view[:, :, ::-1])
            cv2.imwrite(os.path.join(args.reference_imgs_folder, "mask", f'{img_fn}.png'), 
                        255*rendering_results_dict.get("Mask")[:,:,0])
            cv2.imwrite(os.path.join(args.reference_imgs_folder, "semantic_map", f'{img_fn}.png'), 
                        255*rendering_results_dict.get("SemanticMap"))
            np.save(str(os.path.join(args.reference_imgs_folder, "smpl_results", f'{img_fn}.npy')),
                results_dict_for_rendering)
    for video_path in tqdm(driving_videos_paths, desc ="Processing Driving Videos:"):
        video_path = os.path.join(args.driving_videos_folder, video_path)
        os.makedirs(video_path, exist_ok=True)
        # os.makedirs(os.path.join(video_path, "visualized_imgs"), exist_ok=True)
        # os.makedirs(os.path.join(video_path, "mask"), exist_ok=True)
        # os.makedirs(os.path.join(video_path, "semantic_map"), exist_ok=True)
        # os.makedirs(os.path.join(video_path, "depth"), exist_ok=True)
        os.makedirs(os.path.join(video_path, "smpl_results"), exist_ok=True)

        driving_img_paths = [ path for path in os.listdir(os.path.join(video_path, "images"))]
        driving_img_paths.sort(key=lambda x: int(x.split('.')[0]))
        smpls = []
        cams = []
        for img_path in tqdm(driving_img_paths):
            img_cv2 = cv2.imread(str(os.path.join(video_path, "images", img_path)))

            # renderer.renderer.delete()
            # renderer.renderer = pyrender.OffscreenRenderer(viewport_width=img_cv2.shape[:2][::-1][0],
            #                                         viewport_height=img_cv2.shape[:2][::-1][1],
            #                                         point_size=1.0)
            img_fn, _ = os.path.splitext(os.path.basename(img_path))
            
            # Detect humans in image
            det_out = detector(img_cv2)

            det_instances = det_out['instances']
            valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
            boxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
            
            # Run HMR2.0 on all detected humans
            dataset = ViTDetDataset(model_cfg, img_cv2, boxes)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

            all_verts = []
            all_cam_t = []
            
            for batch in dataloader:
                batch = recursive_to(batch, args.device)
                with torch.no_grad():
                    out = model(batch)

                pred_cam = out['pred_cam']
                pred_smpl_parameter = out['pred_smpl_params']
                # transfer reference SMPL shape to driving SMPLs
                if args.figure_scale is not None:
                    pred_smpl_parameter['betas'][0][1] = args.figure_scale
                    print("Adjust the figure scale to:", args.figure_scale)
                
                
                smpl_output = model.smpl(**{k: v.float() for k,v in pred_smpl_parameter.items()}, pose2rot=False)
                pred_vertices = smpl_output.vertices
                out['pred_vertices'] = pred_vertices.reshape(batch['img'].shape[0], -1, 3)
                
                box_center = batch["box_center"].float()
                box_size = batch["box_size"].float()
                img_size = batch["img_size"].float()
                
                scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
                pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()
                # Render the result
                batch_size = batch['img'].shape[0]
                for n in range(batch_size):
                    # Add all verts and cams to list
                    verts = out['pred_vertices'][n].detach().cpu().numpy()
                    cam_t = pred_cam_t_full[n]
                    all_verts.append(verts)
                    all_cam_t.append(cam_t)
                    
                misc_args = dict(
                    mesh_base_color=LIGHT_BLUE,
                    scene_bg_color=(1, 1, 1),
                    focal_length=scaled_focal_length,
                )
                smpl_outs = {k: v.detach().cpu().numpy() for k, v in pred_smpl_parameter.items()}
                cams.append(all_cam_t[0])
                smpls.append(smpl_outs)
                results_dict_for_rendering = {"verts":all_verts, "cam_t":all_cam_t, 
                    "render_res":img_size[n].cpu().numpy(), "smpls":smpl_outs,
                    "scaled_focal_length":scaled_focal_length.cpu().numpy()}
                np.save(str(os.path.join(video_path, "smpl_results", f'{img_fn}.npy')),
                    results_dict_for_rendering)
        np.savez(str(os.path.join(video_path, f'smpls_group.npz')), smpl=smpls, camera=cams)