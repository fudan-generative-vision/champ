import argparse
import logging
import os
import os.path as osp
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.checkpoint
from torchvision import transforms
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from PIL import Image
from transformers import CLIPVisionModelWithProjection

from models.unet_2d_condition import UNet2DConditionModel
from models.unet_3d import UNet3DConditionModel
from models.mutual_self_attention import ReferenceAttentionControl
from models.guidance_encoder import GuidanceEncoder
from models.champ_model import ChampModel

from pipelines.pipeline_aggregation import MultiGuidance2LongVideoPipeline

from utils.video_utils import resize_tensor_frames, save_videos_grid, pil_list_to_tensor


def setup_savedir(cfg):
    time_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    if cfg.exp_name is None:
        savedir = f"results/exp-{time_str}"
    else:
        savedir = f"results/{cfg.exp_name}-{time_str}"

    os.makedirs(savedir, exist_ok=True)

    return savedir


def setup_guidance_encoder(cfg):
    guidance_encoder_group = dict()

    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    for guidance_type in cfg.guidance_types:
        guidance_encoder_group[guidance_type] = GuidanceEncoder(
            guidance_embedding_channels=cfg.guidance_encoder_kwargs.guidance_embedding_channels,
            guidance_input_channels=cfg.guidance_encoder_kwargs.guidance_input_channels,
            block_out_channels=cfg.guidance_encoder_kwargs.block_out_channels,
        ).to(device="cuda", dtype=weight_dtype)

    return guidance_encoder_group


def process_semantic_map(semantic_map_path: Path):
    image_name = semantic_map_path.name
    mask_path = semantic_map_path.parent.parent / "mask" / image_name
    semantic_array = np.array(Image.open(semantic_map_path))
    mask_array = np.array(Image.open(mask_path).convert("RGB"))
    semantic_pil = Image.fromarray(np.where(mask_array > 0, semantic_array, 0))

    return semantic_pil


def combine_guidance_data(cfg):
    guidance_types = cfg.guidance_types
    guidance_data_folder = cfg.data.guidance_data_folder

    guidance_pil_group = dict()
    for guidance_type in guidance_types:
        guidance_pil_group[guidance_type] = []
        guidance_image_lst = sorted(
            Path(osp.join(guidance_data_folder, guidance_type)).iterdir()
        )
        guidance_image_lst = (
            guidance_image_lst
            if not cfg.data.frame_range
            else guidance_image_lst[cfg.data.frame_range[0], cfg.data.frame_range[1]]
        )

        for guidance_image_path in guidance_image_lst:
            # Add black background to semantic map
            if guidance_type == "semantic_map":
                guidance_pil_group[guidance_type] += [
                    process_semantic_map(guidance_image_path)
                ]
            else:
                guidance_pil_group[guidance_type] += [
                    Image.open(guidance_image_path).convert("RGB")
                ]

    # get video length from the first guidance sequence
    first_guidance_length = len(list(guidance_pil_group.values())[0])
    # ensure all guidance sequences are of equal length
    assert all(
        len(sublist) == first_guidance_length
        for sublist in list(guidance_pil_group.values())
    )

    return guidance_pil_group, first_guidance_length


def inference(
    cfg,
    vae,
    image_enc,
    model,
    scheduler,
    ref_image_pil,
    guidance_pil_group,
    video_length,
    width,
    height,
    device="cuda",
    dtype=torch.float16,
):
    reference_unet = model.reference_unet
    denoising_unet = model.denoising_unet
    guidance_types = cfg.guidance_types
    guidance_encoder_group = {
        f"guidance_encoder_{g}": getattr(model, f"guidance_encoder_{g}")
        for g in guidance_types
    }

    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed)
    pipeline = MultiGuidance2LongVideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        **guidance_encoder_group,
        scheduler=scheduler,
    )
    pipeline = pipeline.to(device, dtype)

    video = pipeline(
        ref_image_pil,
        guidance_pil_group,
        width,
        height,
        video_length,
        num_inference_steps=cfg.num_inference_steps,
        guidance_scale=cfg.guidance_scale,
        generator=generator,
    ).videos

    del pipeline
    torch.cuda.empty_cache()

    return video


def main(cfg):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    save_dir = setup_savedir(cfg)
    logging.info(f"Running inference ...")

    # setup pretrained models
    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    noise_scheduler = DDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        cfg.image_encoder_path,
    ).to(dtype=weight_dtype, device="cuda")

    vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(
        dtype=weight_dtype, device="cuda"
    )

    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        cfg.base_model_path,
        cfg.motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=cfg.unet_additional_kwargs,
    ).to(dtype=weight_dtype, device="cuda")

    reference_unet = UNet2DConditionModel.from_pretrained(
        cfg.base_model_path,
        subfolder="unet",
    ).to(device="cuda", dtype=weight_dtype)

    guidance_encoder_group = setup_guidance_encoder(cfg)

    ckpt_dir = cfg.ckpt_dir
    denoising_unet.load_state_dict(
        torch.load(
            osp.join(ckpt_dir, f"denoising_unet.pth"),
            map_location="cpu",
        ),
        strict=False,
    )
    reference_unet.load_state_dict(
        torch.load(
            osp.join(ckpt_dir, f"reference_unet.pth"),
            map_location="cpu",
        ),
        strict=False,
    )

    for guidance_type, guidance_encoder_module in guidance_encoder_group.items():
        guidance_encoder_module.load_state_dict(
            torch.load(
                osp.join(ckpt_dir, f"guidance_encoder_{guidance_type}.pth"),
                map_location="cpu",
            ),
            strict=False,
        )

    reference_control_writer = ReferenceAttentionControl(
        reference_unet,
        do_classifier_free_guidance=False,
        mode="write",
        fusion_blocks="full",
    )
    reference_control_reader = ReferenceAttentionControl(
        denoising_unet,
        do_classifier_free_guidance=False,
        mode="read",
        fusion_blocks="full",
    )

    model = ChampModel(
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        reference_control_writer=reference_control_writer,
        reference_control_reader=reference_control_reader,
        guidance_encoder_group=guidance_encoder_group,
    ).to("cuda", dtype=weight_dtype)

    if cfg.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            reference_unet.enable_xformers_memory_efficient_attention()
            denoising_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    ref_image_path = cfg.data.ref_image_path
    ref_image_pil = Image.open(ref_image_path)
    ref_image_w, ref_image_h = ref_image_pil.size

    guidance_pil_group, video_length = combine_guidance_data(cfg)

    result_video_tensor = inference(
        cfg=cfg,
        vae=vae,
        image_enc=image_enc,
        model=model,
        scheduler=noise_scheduler,
        ref_image_pil=ref_image_pil,
        guidance_pil_group=guidance_pil_group,
        video_length=video_length,
        width=cfg.width,
        height=cfg.height,
        device="cuda",
        dtype=weight_dtype,
    )  # (1, c, f, h, w)

    result_video_tensor = resize_tensor_frames(
        result_video_tensor, (ref_image_h, ref_image_w)
    )
    save_videos_grid(result_video_tensor, osp.join(save_dir, "animation.mp4"))

    ref_video_tensor = transforms.ToTensor()(ref_image_pil)[None, :, None, ...].repeat(
        1, 1, video_length, 1, 1
    )
    guidance_video_tensor_lst = []
    for guidance_pil_lst in guidance_pil_group.values():
        guidance_video_tensor_lst += [
            pil_list_to_tensor(guidance_pil_lst, size=(ref_image_h, ref_image_w))
        ]
    guidance_video_tensor = torch.stack(guidance_video_tensor_lst, dim=0)

    grid_video = torch.cat([ref_video_tensor, result_video_tensor], dim=0)
    grid_video_wguidance = torch.cat(
        [ref_video_tensor, result_video_tensor, guidance_video_tensor], dim=0
    )

    save_videos_grid(grid_video, osp.join(save_dir, "grid.mp4"))
    save_videos_grid(grid_video_wguidance, osp.join(save_dir, "grid_wguidance.mp4"))

    logging.info(f"Inference completed, results saved in {save_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/inference.yaml")
    args = parser.parse_args()

    if args.config[-5:] == ".yaml":
        cfg = OmegaConf.load(args.config)
    else:
        raise ValueError("Do not support this format config file")

    main(cfg)
