import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from loguru import logger
from torch import Tensor

from app.exps.yolox_s import Exp
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.models.yolox import YOLOX
from yolox.utils.boxes import postprocess


class Predictor:
    model: YOLOX
    exp: Exp
    cls_names: Tuple[str, str]
    device: str = "cpu"
    fp16: bool = False
    legacy: bool = False

    def __init__(
        self,
        exp: Exp,
        device: str = "cpu",
        fp16: bool = False,
        legacy: bool = False,
    ):
        self.exp = exp
        self.model = self.exp.get_model().eval()

        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)

    # FIXME
    def __load_exp(self) -> None:
        self.num_classes = self.exp.num_classes
        self.confthre = self.exp.test_conf
        self.nmsthre = self.exp.nmsthre
        self.test_size = self.exp.test_size

        model = self.exp.get_model()
        self.model = model.eval()

    def __preprocess(self, img: np.ndarray) -> Tensor:
        img, _ = self.preproc(img, None, self.test_size)
        tensor: Tensor = torch.from_numpy(img).unsqueeze(0)
        tensor = tensor.float()
        if self.device == "gpu":
            tensor = tensor.cuda()
            if self.fp16:
                tensor = tensor.half()  # to FP16
        return tensor

    def load_model(self, ckpt_path: str) -> None:
        self.__load_exp()
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    def inference(self, img: np.ndarray):
        tensor = self.__preprocess(img)
        outputs = []
        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(tensor)

            outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre, class_agnostic=True)
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs

    def post_process(self, outputs: List[Tensor]):
        final: List[Dict] = []
        for p in outputs:
            for *xyxy, conf, cls in p:
                output = {}
                output["class_id"] = self.cls_names[int(cls)]
                output["confidence"] = float(conf)
                output["bbox"] = list(map(float, xyxy))
                final.append(output)
        return final


class yolox:
    _model: YOLOX

    def __init__(self, exp: Exp):
        self._model = exp.get_model().eval()

    def inference(self, tensor: Tensor) -> List[Tensor]:
        with torch.no_grad():
            outputs = self._model(tensor)
        return outputs

    def post_process(self, outputs: List[Tensor], classes: List[str]):
        final: List[Dict] = []
        for p in outputs:
            for *xyxy, conf, cls in p:
                output = {}
                output["class_id"] = classes[int(cls)]
                output["confidence"] = float(conf)
                output["bbox"] = list(map(float, xyxy))
                final.append(output)
        return final


@dataclass(eq=False)
class ImgProc:
    test_size: Tuple[int, int]
    preproc: ValTransform = ValTransform()
    device: str = "cpu"
    fp16: bool = False

    def preprocess(self, img: np.ndarray) -> Tensor:
        img, _ = self.preproc(img, None, self.test_size)
        tensor: Tensor = torch.from_numpy(img).unsqueeze(0)
        tensor = tensor.float()
        if self.device == "gpu":
            tensor = tensor.cuda()
            if self.fp16:
                tensor = tensor.half()  # to FP16
        return tensor


def post_process(outputs: List[Tensor], classes: Tuple = COCO_CLASSES):
    final: List[Dict] = []
    for p in outputs:
        for *xyxy, conf, cls in p:
            output = {}
            output["class_id"] = classes[int(cls)]
            output["confidence"] = float(conf)
            output["bbox"] = list(map(float, xyxy))
            final.append(output)
    return final


def load_exp(exp: Exp) -> YOLOX:
    return exp.get_model().eval()


def load_weights(ckpt_path: str, device: str = "cpu") -> Any:
    try:
        ckpt: dict = torch.load(ckpt_path, map_location=device)
        return ckpt
    except Exception as e:
        print(e)
        raise Exception("model weights not found")


def inference(model: YOLOX, tensor: Tensor) -> List[Tensor]:
    with torch.no_grad():
        outputs = model(tensor)
    return outputs


def load_model(model: YOLOX, ckpt: Any):
    try:
        model.load_state_dict(ckpt["model"])
        return model
    except Exception as e:
        print(e)
        raise Exception("model weights not found")


if __name__ == "__main__":
    yolox_model = torch.hub.load("Megvii-BaseDetection/YOLOX", "yolox_s")
    import cv2

    img = cv2.imread("/workspaces/yolox-aws-lambda/YOLOX/assets/dog.jpg")
    img_proc = ImgProc(test_size=(416, 416))

    tensor = img_proc.preprocess(img=img)
    output = inference(model=yolox_model, tensor=tensor)

    print(post_process(output))
