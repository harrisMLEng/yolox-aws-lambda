import cv2
from numpy import ndarray
import pytest
from torch import Tensor

from app.predictor.predictor import ImgProc


@pytest.fixture
def image1() -> ndarray:
    img = cv2.imread("/workspaces/yolox-aws-lambda/YOLOX/assets/demo.png")
    return img


# FIXME pytest-lazy-fixture not compatible fix this
def test_image_preprocessor(image1):
    img_preproc = ImgProc(test_size=(416, 416))

    tensor = img_preproc.preprocess(image1)

    assert isinstance(tensor, Tensor)
