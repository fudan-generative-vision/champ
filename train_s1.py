import argparse
import logging
import math
import os
import os.path as osp
import random
import warnings
from pathlib import Path

import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from torchvision.utils import save_image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from tqdm.auto import tqdm
from transformers import CLIPVisionModelWithProjection

from models.champ_model import ChampModel
from models.guidance_encoder import GuidanceEncoder
from models.unet_2d_condition import UNet2DConditionModel
from models.unet_3d import UNet3DConditionModel
from models.mutual_self_attention import ReferenceAttentionControl

from datasets.image_dataset import ImageDataset
from datasets.data_utils import mask_to_bkgd
from utils.tb_tracker import TbTracker
from utils.util import seed_everything, delete_additional_ckpt, compute_snr

from pipelines.pipeline_guidance2image import MultiGuidance2ImagePipeline

warnings.filterwarnings("ignore")
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def padding_pil(img_pil, img_size):
    # resize a PIL image and zero padding the short edge
    W, H = img_pil.size
    resize_ratio = img_size / max(W, H)
    new_W, new_H = int(W * resize_ratio), int(H * resize_ratio)
    img_pil = img_pil.resize((new_W, new_H))
    
    left = (img_size - new_W) // 2
    right = img_size - new_W - left
    top = (img_size - new_H) // 2
    bottom = img_size - new_H - top
    
    padding_border = (left, top, right, bottom)
    img_pil = ImageOps.expand(img_pil, border=padding_border, fill=0)
    
    return img_pil

def concat_pil(img_pil_lst):
    # horizontally concat PIL images 
    # NOTE(ZSH): assume all images are of same size
    W, H = img_pil_lst[0].size
    num_img = len(img_pil_lst)
    new_width = num_img * W
    new_image = Image.new("RGB", (new_width, H), color=0)
    for img_idx, img in enumerate(img_pil_lst):
        new_image.paste(img, (W * img_idx, 0))  
    
    return new_image

def validate(
    ref_img_path,
    guid_folder,
    guid_types,
    guid_idx,
    width, height,
    pipe,
    generator,
    denoising_steps=20,
    guidance_scale=3.5,
    aug_type="Resize",
):
    ref_img_pil = Image.open(ref_img_path)
    if aug_type =="Padding":
        ref_img_pil = padding_pil(ref_img_pil, height)
    guid_folder = Path(guid_folder)
    guid_img_pil_lst = []
    for guid_type in guid_types:
        guid_img_lst = sorted((guid_folder / guid_type).iterdir())
        guid_img_path = guid_img_lst[guid_idx]
        if guid_type == "semantic_map":
            mask_img_path = guid_folder / "mask" / guid_img_path.name
            guid_img_pil = mask_to_bkgd(guid_img_path, mask_img_path)
        else:
            guid_img_pil = Image.open(guid_img_path).convert("RGB")
        if aug_type == "Padding":
            guid_img_pil = padding_pil(guid_img_pil, height)
        guid_img_pil_lst += [guid_img_pil]
    
    val_images = pipe(
        ref_img_pil,
        guid_img_pil_lst,
        guid_types,
        width,
        height,
        denoising_steps,
        guidance_scale,
        generator=generator,
    ).images
    
    return val_images, ref_img_pil, guid_img_pil_lst

def log_validation(
    cfg,
    vae,
    image_enc,
    model,
    scheduler,
    accelerator,
    width,
    height,
    seed=42,
    dtype=torch.float32,
):
    logger.info("Running validation ...")
    unwrap_model = accelerator.unwrap_model(model)
    reference_unet = unwrap_model.reference_unet
    denoising_unet = unwrap_model.denoising_unet
    guid_types = unwrap_model.guidance_types
    guidance_encoder_group = {
        f"guidance_encoder_{g}": getattr(unwrap_model, f"guidance_encoder_{g}") for g in guid_types
    }
    
    generator = torch.manual_seed(seed)
    vae = vae.to(dtype=dtype)
    image_enc = image_enc.to(dtype=dtype)
    
    pipeline = MultiGuidance2ImagePipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        **guidance_encoder_group,
        scheduler=scheduler,
    )
    pipeline = pipeline.to(accelerator.device)
    
    ref_img_lst = cfg.validation.ref_images
    guid_folder_lst = cfg.validation.guidance_folders
    guid_idxes = cfg.validation.guidance_indexes
    
    val_results = []
    for val_idx, (ref_img_path, guid_folder, guid_idx) in enumerate(
        zip(ref_img_lst, guid_folder_lst, guid_idxes)):
        
        image_tensor, ref_img_pil, guid_img_pil_lst = validate(
            ref_img_path=ref_img_path,
            guid_folder=guid_folder,
            guid_types=guid_types,
            guid_idx=guid_idx,
            width=width,
            height=height,
            pipe=pipeline,
            generator=generator,
            aug_type=cfg.data.aug_type,
        )
        
        image_tensor = image_tensor[0, :, 0].permute(1, 2, 0).cpu().numpy()
        W, H = ref_img_pil.size
        result_img_pil = Image.fromarray((image_tensor * 255).astype(np.uint8))
        result_img_pil = result_img_pil.resize((W, H))
        guid_img_pil_lst = [img.resize((W, H)) for img in guid_img_pil_lst]
        result_pil_lst = [result_img_pil, ref_img_pil, *guid_img_pil_lst]
        concated_pil = concat_pil(result_pil_lst)
        
        val_results.append({"name": f"val_{val_idx}", "img": concated_pil})
    
    vae = vae.to(dtype=torch.float16)
    image_enc = image_enc.to(dtype=torch.float16)

    del pipeline
    torch.cuda.empty_cache()

    return val_results

def setup_guidance_encoder(cfg):
    guidance_encoder_group = dict()

    for guidance_type in cfg.data.guids:
        guidance_encoder_group[guidance_type] = GuidanceEncoder(
            guidance_embedding_channels=cfg.guidance_encoder_kwargs.guidance_embedding_channels,
            guidance_input_channels=cfg.guidance_encoder_kwargs.guidance_input_channels,
            block_out_channels=cfg.guidance_encoder_kwargs.block_out_channels,
        )

    return guidance_encoder_group


def main(cfg):
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    tb_tracker = TbTracker(cfg.exp_name, cfg.output_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        mixed_precision=cfg.solver.mixed_precision,
        log_with=tb_tracker,
        project_dir=f'{cfg.output_dir}/{cfg.exp_name}',
        kwargs_handlers=[kwargs],
    )
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=True)
    
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
        
    if cfg.seed is not None:
        seed_everything(cfg.seed)

    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif cfg.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        raise ValueError(
            f"Do not support weight dtype: {cfg.weight_dtype} during training"
        )

    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})
    train_noise_scheduler = DDIMScheduler(**sched_kwargs)
    
    vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(
        "cuda", dtype=weight_dtype
    )

    reference_unet = UNet2DConditionModel.from_pretrained(
        cfg.base_model_path,
        subfolder="unet",
    ).to(device="cuda")
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        cfg.base_model_path,
        "",
        subfolder="unet",
        unet_additional_kwargs={
            "use_motion_module": False,
            "unet_use_temporal_attention": False,
        },
    ).to(device="cuda")

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        cfg.image_encoder_path,
    ).to(dtype=weight_dtype, device="cuda")    
    
    guidance_encoder_group = setup_guidance_encoder(cfg)
    
    # Freeze some modules
    vae.requires_grad_(False)
    image_enc.requires_grad_(False)
    denoising_unet.requires_grad_(True)
    for name, param in reference_unet.named_parameters():
        if "up_blocks.3" in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)
            
    for module in guidance_encoder_group.values():
        module.requires_grad_(True)
            
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
        reference_unet,
        denoising_unet,
        reference_control_writer,
        reference_control_reader,
        guidance_encoder_group,
    )
    
    if cfg.solver.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            reference_unet.enable_xformers_memory_efficient_attention()
            denoising_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if cfg.solver.gradient_checkpointing:
        reference_unet.enable_gradient_checkpointing()
        denoising_unet.enable_gradient_checkpointing()

    if cfg.solver.scale_lr:
        learning_rate = (
            cfg.solver.learning_rate
            * cfg.solver.gradient_accumulation_steps
            * cfg.data.train_bs
            * accelerator.num_processes
        )
    else:
        learning_rate = cfg.solver.learning_rate
        
    if cfg.solver.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = optimizer_cls(
        trainable_params,
        lr=learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        cfg.solver.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.solver.lr_warmup_steps
        * cfg.solver.gradient_accumulation_steps,
        num_training_steps=cfg.solver.max_train_steps
        * cfg.solver.gradient_accumulation_steps,
    )            
    
    train_dataset = ImageDataset(
        video_folder=cfg.data.video_folder,
        image_size=cfg.data.image_size,
        sample_margin=cfg.data.sample_margin,
        data_parts=cfg.data.data_parts,
        guids=cfg.data.guids,
        extra_region=None,
        bbox_crop=cfg.data.bbox_crop,
        bbox_resize_ratio=tuple(cfg.data.bbox_resize_ratio),
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.data.train_bs, shuffle=True, num_workers=16
    )
    
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.solver.gradient_accumulation_steps
    )
    num_train_epochs = math.ceil(
        cfg.solver.max_train_steps / num_update_steps_per_epoch
    )
    
    logger.info("Start training ...")
    logger.info(f"Num Samples: {len(train_dataset)}")
    logger.info(f"Train Batchsize: {cfg.data.train_bs}")
    logger.info(f"Num Epochs: {num_train_epochs}")
    logger.info(f"Total Steps: {cfg.solver.max_train_steps}")
    
    global_step, first_epoch = 0, 0
    
    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint != "latest":
            resume_dir = cfg.resume_from_checkpoint
        else:
            resume_dir = f"{cfg.output_dir}/{cfg.exp_name}/checkpoints"
        dirs = os.listdir(resume_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1]
        accelerator.load_state(os.path.join(resume_dir, path))
        accelerator.print(f"Resuming from checkpoint {path}")
        global_step = int(path.split("-")[1])

        first_epoch = global_step // num_update_steps_per_epoch
        
    progress_bar = tqdm(
        range(global_step, cfg.solver.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")
    
    # Training Loop
    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.
        for _, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                pixel_values = batch["tgt_img"].to(weight_dtype)
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents.unsqueeze(2)  # (b, c, 1, h, w)
                    latents = latents * 0.18215
                    
                noise = torch.randn_like(latents)
                if cfg.noise_offset > 0.0:
                    noise += cfg.noise_offset * torch.randn(
                        (noise.shape[0], noise.shape[1], 1, 1, 1),
                        device=noise.device,
                    )
                    
                bsz = latents.shape[0]
                # Sample a random timestep for each video
                timesteps = torch.randint(
                    0,
                    train_noise_scheduler.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()
            
                tgt_guid_imgs = batch["tgt_guid"]
                tgt_guid_imgs = tgt_guid_imgs.unsqueeze(2)
                
                uncond_fwd = random.random() < cfg.uncond_ratio
                clip_image_list = []
                ref_image_list = []
                for batch_idx, (ref_img, clip_img) in enumerate(
                    zip(
                        batch["ref_img"],
                        batch["clip_img"],
                    )
                ):
                    if uncond_fwd:
                        clip_image_list.append(torch.zeros_like(clip_img))
                    else:
                        clip_image_list.append(clip_img)
                    ref_image_list.append(ref_img)
                                   
                with torch.no_grad():
                    ref_img = torch.stack(ref_image_list, dim=0).to(
                        dtype=vae.dtype, device=vae.device
                    )
                    ref_image_latents = vae.encode(
                        ref_img
                    ).latent_dist.sample()  # (bs, d, 64, 64)
                    ref_image_latents = ref_image_latents * 0.18215

                    clip_img = torch.stack(clip_image_list, dim=0).to(
                        dtype=image_enc.dtype, device=image_enc.device
                    )
                    clip_image_embeds = image_enc(
                        clip_img.to("cuda", dtype=weight_dtype)
                    ).image_embeds
                    image_prompt_embeds = clip_image_embeds.unsqueeze(1)  # (bs, 1, d)
             
                noisy_latents = train_noise_scheduler.add_noise(
                    latents, noise, timesteps
                )

                if train_noise_scheduler.prediction_type == "epsilon":
                    target = noise
                elif train_noise_scheduler.prediction_type == "v_prediction":
                    target = train_noise_scheduler.get_velocity(
                        latents, noise, timesteps
                    )
                else:
                    raise ValueError(
                        f"Unknown prediction type {train_noise_scheduler.prediction_type}"
                    )
                    
                model_pred = model(
                    noisy_latents,
                    timesteps,
                    ref_image_latents,
                    image_prompt_embeds,
                    tgt_guid_imgs,
                    uncond_fwd,
                )
                
                if cfg.snr_gamma == 0:
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )
                else:
                    snr = compute_snr(train_noise_scheduler, timesteps)
                    if train_noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective requires that we add one to SNR values before we divide by them.
                        snr = snr + 1
                    mse_loss_weights = (
                        torch.stack(
                            [snr, cfg.snr_gamma * torch.ones_like(timesteps)], dim=1
                        ).min(dim=1)[0]
                        / snr
                    )
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    )
                    loss = (
                        loss.mean(dim=list(range(1, len(loss.shape))))
                        * mse_loss_weights
                    )
                    loss = loss.mean()
                    
                avg_loss = accelerator.gather(loss.repeat(cfg.data.train_bs)).mean()
                train_loss += avg_loss.item() / cfg.solver.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        trainable_params,
                        cfg.solver.max_grad_norm,
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Logging
            save_dir = f"{cfg.output_dir}/{cfg.exp_name}"
            if accelerator.sync_gradients:
                reference_control_reader.clear()
                reference_control_writer.clear()
                progress_bar.update(1)
                global_step += 1
                tb_tracker.add_scalar(tag='train loss', scalar_value=train_loss, global_step=global_step)
                train_loss = 0.0
                #　save checkpoints
                if global_step % cfg.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
                        delete_additional_ckpt(save_dir, 6)
                        accelerator.save_state(save_path)                
                # check data
                if global_step % cfg.checkpointing_steps == 0 or global_step == 1:
                    img_forcheck = batch['tgt_img'] * 0.5 + 0.5
                    ref_forcheck = batch['ref_img'] * 0.5 + 0.5
                    guid_forcheck = list(torch.chunk(batch['tgt_guid'], batch['tgt_guid'].shape[1]//3, dim=1))
                    batch_forcheck = torch.cat([ref_forcheck, img_forcheck] + guid_forcheck, dim=0)
                    save_image(batch_forcheck, f'{cfg.output_dir}/{cfg.exp_name}/sanity_check/data-{global_step:06d}-rank{accelerator.device.index}.png', nrow=4)
                # log validation                      
                if global_step % cfg.validation.validation_steps == 0 or global_step == 1:
                    if accelerator.is_main_process:
                        
                        sample_dicts = log_validation(
                            cfg=cfg,
                            vae=vae,
                            image_enc=image_enc,
                            model=model,
                            scheduler=val_noise_scheduler,
                            accelerator=accelerator,
                            width=cfg.data.image_size,
                            height=cfg.data.image_size,
                            seed=cfg.seed
                        )
                        
                        for sample_dict in sample_dicts:
                            sample_name = sample_dict["name"]
                            img = sample_dict["img"]
                            img.save(f"{save_dir}/validation/{global_step:06d}-{sample_name}.png")
                            
            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "stage": 1,
            }
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.solver.max_train_steps:
                break
            
        # save model after each epoch
        if (
            epoch + 1
        ) % cfg.save_model_epoch_interval == 0 and accelerator.is_main_process:
            unwrap_model = accelerator.unwrap_model(model)
            save_checkpoint(
                unwrap_model.reference_unet,
                f"{save_dir}/saved_models",
                "reference_unet",
                global_step,
                total_limit=None,
            )
            save_checkpoint(
                unwrap_model.denoising_unet,
                f"{save_dir}/saved_models",
                "denoising_unet",
                global_step,
                total_limit=None,
            )
            for guid_type in unwrap_model.guidance_types:
                save_checkpoint(
                    getattr(unwrap_model, f"guidance_encoder_{guid_type}"),
                    f"{save_dir}/saved_models",
                    f"guidance_encoder_{guid_type}",
                    global_step,
                    total_limit=None,
                )

    accelerator.wait_for_everyone()
    accelerator.end_training()                                    
                            
def save_checkpoint(model, save_dir, prefix, ckpt_num, total_limit=None):
    save_path = osp.join(save_dir, f"{prefix}-{ckpt_num}.pth")

    if total_limit is not None:
        checkpoints = os.listdir(save_dir)
        checkpoints = [d for d in checkpoints if d.startswith(prefix)]
        checkpoints = sorted(
            checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0])
        )

        if len(checkpoints) >= total_limit:
            num_to_remove = len(checkpoints) - total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]
            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(save_dir, removing_checkpoint)
                os.remove(removing_checkpoint)

    state_dict = model.state_dict()
    torch.save(state_dict, save_path)           
    
    
if __name__ == "__main__":
    import shutil
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train/stage1.yaml")
    args = parser.parse_args()

    if args.config[-5:] == ".yaml":
        config = OmegaConf.load(args.config)
    else:
        raise ValueError("Do not support this format config file")
    
    save_dir = os.path.join(config.output_dir, config.exp_name)
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'sanity_check'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'saved_models'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'validation'), exist_ok=True)
    # save config, script
    shutil.copy(args.config, os.path.join(save_dir, 'sanity_check', f'{config.exp_name}.yaml'))
    shutil.copy(os.path.abspath(__file__), os.path.join(save_dir, 'sanity_check'))
    
    main(config)             
    