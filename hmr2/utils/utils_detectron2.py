import detectron2.data.transforms as T
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, instantiate
from detectron2.data import MetadataCatalog
from omegaconf import OmegaConf


class DefaultPredictor_Lazy:
    """Create a simple end-to-end predictor with the given config that runs on single device for a
    single input image.

    Compared to using the model directly, this class does the following additions:

    1. Load checkpoint from the weights specified in config (cfg.MODEL.WEIGHTS).
    2. Always take BGR image as the input and apply format conversion internally.
    3. Apply resizing defined by the config (`cfg.INPUT.{MIN,MAX}_SIZE_TEST`).
    4. Take one input image and produce a single output, instead of a batch.

    This is meant for simple demo purposes, so it does the above steps automatically.
    This is not meant for benchmarks or running complicated inference logic.
    If you'd like to do anything more complicated, please refer to its source code as
    examples to build and use the model manually.

    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            test dataset name in the config.


    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg):
        """
        Args:
            cfg: a yacs CfgNode or a omegaconf dict object.
        """
        if isinstance(cfg, CfgNode):
            self.cfg = cfg.clone()  # cfg can be modified by model
            self.model = build_model(self.cfg)  # noqa: F821
            if len(cfg.DATASETS.TEST):
                test_dataset = cfg.DATASETS.TEST[0]

            checkpointer = DetectionCheckpointer(self.model)
            checkpointer.load(cfg.MODEL.WEIGHTS)

            self.aug = T.ResizeShortestEdge(
                [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
            )

            self.input_format = cfg.INPUT.FORMAT
        else:  # new LazyConfig
            self.cfg = cfg
            self.model = instantiate(cfg.model)
            test_dataset = OmegaConf.select(cfg, "dataloader.test.dataset.names", default=None)
            if isinstance(test_dataset, (list, tuple)):
                test_dataset = test_dataset[0]

            checkpointer = DetectionCheckpointer(self.model)
            checkpointer.load(OmegaConf.select(cfg, "train.init_checkpoint", default=""))

            mapper = instantiate(cfg.dataloader.test.mapper)
            self.aug = mapper.augmentations
            self.input_format = mapper.image_format

        self.model.eval().cuda()
        if test_dataset:
            self.metadata = MetadataCatalog.get(test_dataset)
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():
            if self.input_format == "RGB":
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug(T.AugInput(original_image)).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions
