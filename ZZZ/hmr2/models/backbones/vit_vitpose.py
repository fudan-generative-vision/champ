# import mmcv
# import mmpose
# from mmpose.models import build_posenet
# from mmcv.runner import load_checkpoint
# from pathlib import Path

# def vit(cfg):
#     vitpose_dir = Path(mmpose.__file__).parent.parent
#     config = f'{vitpose_dir}/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py'
#     # checkpoint = f'{vitpose_dir}/models/vitpose-h-multi-coco.pth'

#     config = mmcv.Config.fromfile(config)
#     config.model.pretrained = None
#     model = build_posenet(config.model)
#     # load_checkpoint(model, checkpoint, map_location='cpu')

#     return model.backbone
