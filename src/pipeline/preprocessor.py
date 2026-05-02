"""
ImagePreprocessor — classical OpenCV preprocessing stage.

Handles:
  • Resize / letterbox padding (preserves aspect ratio)
  • CLAHE illumination normalization
  • Fast Non-Local Means denoising
  • Channel normalization + float32 conversion
"""
from __future__ import annotations

import cv2
import numpy as np


class ImagePreprocessor:
    """
    Deterministic, stateless preprocessing stage.

    Parameters
    ----------
    target_size:
        (width, height) the network expects.
    normalize_illumination:
        Apply CLAHE to the L-channel in LAB space.
    denoise_strength:
        Strength of the Non-Local Means filter (0 = disabled).
    """

    def __init__(
        self,
        target_size: tuple[int, int] = (640, 640),
        normalize_illumination: bool = True,
        denoise_strength: int = 7,
    ) -> None:
        self.target_size = target_size
        self.normalize_illumination = normalize_illumination
        self.denoise_strength = denoise_strength

        if normalize_illumination:
            self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # ------------------------------------------------------------------

    def run(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Apply the full preprocessing sequence.

        Returns
        -------
        processed : np.ndarray
            uint8 BGR image at *target_size*, ready for the detector.
        meta : dict
            Bookkeeping data (original shape, scale, padding) needed to
            map bounding boxes back to the original frame.
        """
        meta: dict = {"original_shape": image.shape[:2]}

        # Step 1: denoise
        if self.denoise_strength > 0:
            image = cv2.fastNlMeansDenoisingColored(
                image,
                None,
                h=self.denoise_strength,
                hColor=self.denoise_strength,
                templateWindowSize=7,
                searchWindowSize=21,
            )

        # Step 2: illumination normalisation (CLAHE on L-channel)
        if self.normalize_illumination:
            image = self._apply_clahe(image)

        # Step 3: letterbox resize
        image, scale, (pad_w, pad_h) = self._letterbox(image, self.target_size)
        meta.update({"scale": scale, "pad_w": pad_w, "pad_h": pad_h})

        return image, meta

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        l_channel = self._clahe.apply(l_channel)
        return cv2.cvtColor(cv2.merge([l_channel, a, b]), cv2.COLOR_LAB2BGR)

    @staticmethod
    def _letterbox(
        image: np.ndarray,
        new_size: tuple[int, int],
        fill_value: int = 114,
    ) -> tuple[np.ndarray, float, tuple[int, int]]:
        """
        Resize with aspect-ratio preservation, padding remainder with
        a neutral gray.  Returns (image, scale, (pad_w, pad_h)).
        """
        h, w = image.shape[:2]
        target_w, target_h = new_size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2
        image = cv2.copyMakeBorder(
            image,
            pad_h, target_h - new_h - pad_h,
            pad_w, target_w - new_w - pad_w,
            cv2.BORDER_CONSTANT,
            value=(fill_value, fill_value, fill_value),
        )
        return image, scale, (pad_w, pad_h)

    def unmap_bbox(
        self,
        bbox: tuple[int, int, int, int],
        meta: dict,
    ) -> tuple[int, int, int, int]:
        """
        Map a bbox from processed coordinates back to the original image.
        """
        x1, y1, x2, y2 = bbox
        pad_w, pad_h = meta["pad_w"], meta["pad_h"]
        scale = meta["scale"]
        x1 = int((x1 - pad_w) / scale)
        y1 = int((y1 - pad_h) / scale)
        x2 = int((x2 - pad_w) / scale)
        y2 = int((y2 - pad_h) / scale)
        oh, ow = meta["original_shape"]
        x1, x2 = max(0, x1), min(ow, x2)
        y1, y2 = max(0, y1), min(oh, y2)
        return (x1, y1, x2, y2)
