import numpy as np
import cv2

from ...model_loader import ModelLoader


class ImageEncoder:
    def __init__(self, model_loader: ModelLoader, input_size=(224, 224)):
        self.model = model_loader
        self.input_size = input_size

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        img = cv2.resize(img, self.input_size)
        img = img[:, :, ::-1]
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = np.transpose(img, (2, 0, 1))
        return img

    def _dummy_text_inputs(self, batch_size: int):
        return {
            self.model.inputs["ids"]: np.zeros((batch_size, 77), dtype=np.int64),
            self.model.inputs["mask"]: np.zeros((batch_size, 77), dtype=np.int64),
        }

    def encode(self, images: list[np.ndarray]) -> np.ndarray:
        batch = np.stack([self._preprocess(i) for i in images])
        batch_size = batch.shape[0]

        inputs = {
            self.model.inputs["pixel"]: batch,
        }

        # ДОБАВЛЯЕМ dummy text inputs
        inputs.update(self._dummy_text_inputs(batch_size))

        out = self.model.session.run(
            [self.model.outputs["image"]],
            inputs,
        )[0]

        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / norms