import cv2
import numpy as np

import triton_python_backend_utils as pb_utils

PATCH_SIZE = 200
TARGET_SIZE = 224

class TritonPythonModel:
    def execute(self, requests):
        def _process(img, patch_size=PATCH_SIZE, target_size=TARGET_SIZE):
            _, img_h, img_w = img.shape

            num_patches_h = img_h // patch_size + (1 if img_h % patch_size != 0 else 0)
            num_patches_w = img_w // patch_size + (1 if img_w % patch_size != 0 else 0)

            grid_x, grid_y = np.meshgrid(np.arange(num_patches_w), np.arange(num_patches_h))
            grid_x = grid_x.flatten() * patch_size
            grid_y = grid_y.flatten() * patch_size

            cropped_images = []
            cropped_images_original = []

            for x1, y1 in zip(grid_x, grid_y):
                x2 = min(x1 + patch_size, img_w)
                y2 = min(y1 + patch_size, img_h)

                cropped_image_original = img[:, y1:y2, x1:x2]
                cropped_image_original = np.transpose(cropped_image_original, (1, 2, 0))
                cropped_image_original = cv2.resize(cropped_image_original, (target_size, target_size))
                cropped_image_original = np.transpose(cropped_image_original, (2, 0, 1))
                cropped_images_original.append(cropped_image_original)

                mean = np.array([0.485 * 255, 0.456 * 255, 0.406 * 255])
                std = np.array([0.229 * 255, 0.224 * 255, 0.225 * 255])
                cropped_image = (cropped_image_original - mean[:, None, None]) / std[:, None, None]
                cropped_images.append(cropped_image)

            return np.asarray(cropped_images), np.asarray(cropped_images_original)

        responses = []
        for request in requests:
            img = pb_utils.get_input_tensor_by_name(request, "original").as_numpy()
            cropped_images_list, cropped_images_original_list = [], []
            num_samples = img.shape[0]
            for i in range(num_samples):
                cropped_images, cropped_images_original = _process(img[i])
                if len(cropped_images.shape) == 4:
                    cropped_images_list.append(cropped_images)
                else:
                    print(f"========== JAS DEBUG: Bad cropped_images: {cropped_images.shape}")
                if len(cropped_images_original.shape) == 4:
                    cropped_images_original_list.append(cropped_images_original)
                else:
                    print(f"========== JAS DEBUG: Bad cropped_images: {cropped_images_original.shape}")
            assert len(cropped_images_list) == len(cropped_images_original_list)

            if not cropped_images_list:
                cropped_images = np.zeros((1, 3, 224, 224))
                cropped_images_original = np.zeros((1, 3, 224, 224))
            else:
                cropped_images = np.vstack(cropped_images_list)
                cropped_images_original = np.vstack(cropped_images_original_list)

            out_tensor_0 = pb_utils.Tensor(
                "cropped_images", cropped_images.astype(np.float32)
            )
            out_tensor_1 = pb_utils.Tensor(
                "snippets", cropped_images_original.astype(np.uint8)
            )
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0, out_tensor_1]
            )
            responses.append(inference_response)

        return responses
