from .vit import vit

def create_backbone(cfg):
    if cfg.MODEL.BACKBONE.TYPE == 'vit':
        return vit(cfg)
    else:
        raise NotImplementedError('Backbone type is not implemented')
