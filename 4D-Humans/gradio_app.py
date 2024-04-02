from pathlib import Path
import tempfile
import torch
import argparse
import os
import numpy as np
import gradio as gr
from PIL import Image

from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import HMR2, download_models, load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils import recursive_to
from hmr2.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hmr2.utils.renderer import Renderer, cam_crop_to_full

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

def main():
    parser = argparse.ArgumentParser(description='HMR2 demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    # parser.add_argument('--img_folder', type=str, default='example_data/images', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='demo_out', help='Output folder to save rendered results')
    # parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    # parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=False, help='If set, render all people together also')
    # parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for inference/fitting')

    args = parser.parse_args()

    # Download and load checkpoints
    download_models(CACHE_DIR_4DHUMANS)
    model, model_cfg = load_hmr2(args.checkpoint)

    # Setup HMR2.0 model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Load detector
    from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
    from detectron2.config import LazyConfig
    import hmr2
    cfg_path = Path(hmr2.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.smpl.faces)

    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)

    # # Iterate over all images in folder
    # for img_path in Path(args.img_folder).glob('*.jpg'):
    #     img_cv2 = cv2.imread(str(img_path))
    def infer(in_pil_img, in_threshold=0.8, out_pil_img=None):
        # Convert RGB to BGR
        img_cv2 = np.array(in_pil_img)[:, :, ::-1].copy()

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
        all_mesh_paths = []
        
        temp_name = next(tempfile._get_candidate_names())

        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            pred_cam = out['pred_cam']
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            # Render the result
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                # Get filename from path img_path
                # img_fn, _ = os.path.splitext(os.path.basename(img_path))
                person_id = int(batch['personid'][n])
                white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:,None,None]/255) / (DEFAULT_STD[:,None,None]/255)
                input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
                input_patch = input_patch.permute(1,2,0).numpy()

                # regression_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                #                         out['pred_cam_t'][n].detach().cpu().numpy(),
                #                         batch['img'][n],
                #                         mesh_base_color=LIGHT_BLUE,
                #                         scene_bg_color=(1, 1, 1),
                #                         )

                # if args.side_view:
                #     side_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                #                             out['pred_cam_t'][n].detach().cpu().numpy(),
                #                             white_img,
                #                             mesh_base_color=LIGHT_BLUE,
                #                             scene_bg_color=(1, 1, 1),
                #                             side_view=True)
                #     final_img = np.concatenate([input_patch, regression_img, side_img], axis=1)
                # else:
                #     final_img = np.concatenate([input_patch, regression_img], axis=1)

                # cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_{person_id}.png'), 255*final_img[:, :, ::-1])

                # Add all verts and cams to list
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                cam_t = pred_cam_t_full[n]
                all_verts.append(verts)
                all_cam_t.append(cam_t)

                # Save all meshes to disk
                # if args.save_mesh:
                if True:
                    camera_translation = cam_t.copy()
                    tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_BLUE)

                    temp_path = os.path.join(args.out_folder, f'{temp_name}_{person_id}.obj')
                    tmesh.export(temp_path)
                    all_mesh_paths.append(temp_path)

        # Render front view
        if len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], **misc_args)

            # Overlay image
            input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
            input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

            # cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_all.png'), 255*input_img_overlay[:, :, ::-1])

            # convert to PIL image
            out_pil_img =  Image.fromarray((input_img_overlay*255).astype(np.uint8))
            return out_pil_img, all_mesh_paths
        else:
            return None, []

    with gr.Blocks(title="4DHumans", css=".gradio-container") as demo:

        gr.HTML("""<div style="font-weight:bold; text-align:center; color:royalblue;">HMR 2.0</div>""")

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input image", type="pil")
            with gr.Column():
                output_image = gr.Image(label="Reconstructions", type="pil")
                output_meshes = gr.File(label="3D meshes")

        gr.HTML("""<br/>""")

        with gr.Row():
            threshold = gr.Slider(0, 1.0, value=0.6, label='Detection Threshold')
            send_btn = gr.Button("Infer")
            send_btn.click(fn=infer, inputs=[input_image, threshold], outputs=[output_image, output_meshes])

        gr.Examples([
                ['example_data/images/pexels-anete-lusina-4793258.jpg'], 
                ['example_data/images/skates.png'],
            ], 
            inputs=[input_image, 0.6])

    demo.launch(debug=True)

if __name__ == '__main__':
    main()
