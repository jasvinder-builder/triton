import cv2
import numpy as np

import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def execute(self, requests):
        def _postprocess(img, num_dets, det_boxes, det_scores, det_classes, input_shape):
            num_dets_scalar = num_dets[0]
            boxes = det_boxes[:num_dets_scalar] / np.array(
                [input_shape[0], input_shape[1], input_shape[0], input_shape[1]], dtype=np.float32)
            _, w, h = img.shape
            det_boxes_original = boxes * np.array([h, w, h, w], dtype=np.float32)
            scores = det_scores[:num_dets_scalar].astype(np.float32)
            classes = det_classes[:num_dets_scalar].astype(np.int32)
            person_indices = np.where((classes == 0) & (scores > 0.5))[0]

            cropped_images = []
            cropped_images_original = []
            for box, box_original, score in zip(det_boxes[person_indices], det_boxes_original[person_indices], scores[person_indices]):
                x1, y1, x2, y2 = box_original
                x1 = max(x1, 0)
                y1 = max(y1, 0)
                cropped_image_original = img[:, int(y1):int(y2), int(x1):int(x2)]
                if score > 0.5:
                    cropped_image_original = np.transpose(cropped_image_original, (1, 2, 0))
                    cropped_image_original = cv2.resize(cropped_image_original, (224, 224))
                    cropped_image_original = np.transpose(cropped_image_original, (2, 0, 1))
                    cropped_images_original.append(cropped_image_original)

                    mean=np.array([0.485 * 255, 0.456 * 255, 0.406 * 255])
                    std=np.array([0.229 * 255, 0.224 * 255, 0.225 * 255])
                    cropped_image = (cropped_image_original - mean[:, None, None]) / std[:, None, None]
                    cropped_images.append(cropped_image)

            return np.asarray(cropped_images), np.asarray(cropped_images_original)

        responses = []
        for request in requests:
            img = pb_utils.get_input_tensor_by_name(request, "original").as_numpy()
            num_dets = pb_utils.get_input_tensor_by_name(request, "num_dets").as_numpy()
            det_boxes = pb_utils.get_input_tensor_by_name(request, "det_boxes").as_numpy()
            det_scores = pb_utils.get_input_tensor_by_name(request, "det_scores").as_numpy()
            det_classes = pb_utils.get_input_tensor_by_name(request, "det_classes").as_numpy()
            roi_det_boxes_list, cropped_images_list, cropped_images_original_list = [], [], []
            num_samples = img.shape[0]
            for i in range(num_samples):
                cropped_images, cropped_images_original = _postprocess(img[i], num_dets[i], det_boxes[i], det_scores[i], det_classes[i], (640, 640))
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
