#!/usr/bin/env python

import argparse
import cv2
import numpy as np
import sys

from functools import partial
from pathlib import Path
import queue

from PIL import Image

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

INPUT_NAMES = ["video"]
OUTPUT_NAMES = ["embeddings", "snippets"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input',
                        type=str,
                        nargs='?',
                        help='Input video file')
    parser.add_argument('-m',
                        '--model',
                        type=str,
                        required=False,
                        default='video_grid_processing_service',
                        help='Inference model name, default: video_grid_processing_service')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8001',
                        help='Inference server URL, default localhost:8001')

    FLAGS = parser.parse_args()

    # Create server context
    try:
        triton_client = grpcclient.InferenceServerClient(
            url=FLAGS.url,
            verbose=False,
            ssl=False,
            root_certificates=None,
            private_key=None,
            certificate_chain=None)
    except Exception as e:
        print("context creation failed: " + str(e))
        sys.exit()

    # Health check
    if not triton_client.is_server_live():
        print("FAILED : is_server_live")
        sys.exit(1)

    if not triton_client.is_server_ready():
        print("FAILED : is_server_ready")
        sys.exit(1)

    if not triton_client.is_model_ready(FLAGS.model):
        print("FAILED : is_model_ready")
        sys.exit(1)

    # Queue for results
    completed_requests = queue.Queue()
    def callback(queue, result, error):
        if error:
            queue.put(error)
        else:
            queue.put(result)

    triton_client.start_stream(callback=partial(callback, completed_requests))

    # Configure inputs and outputs
    buffer = np.fromfile(FLAGS.input, dtype=np.uint8)
    buffer = np.expand_dims(buffer, axis=0)
    input_shape = list(buffer.shape)
    inputs, outputs = [], []
    print(f"input shape = {input_shape}")
    inputs.append(grpcclient.InferInput(INPUT_NAMES[0], input_shape, "UINT8"))
    inputs[0].set_data_from_numpy(buffer)
    for i in range(len(OUTPUT_NAMES)):
        outputs.append(grpcclient.InferRequestedOutput(OUTPUT_NAMES[i]))

    triton_client.async_stream_infer(model_name=FLAGS.model, inputs=inputs, request_id="0", outputs=outputs, enable_empty_final_response=True)

    while True:
        result = completed_requests.get()
        if type(result) == InferenceServerException:
            raise result
        response = result.get_response()
        if response.parameters.get('triton_final_response').bool_param:
            break
        embeddings = result.as_numpy(OUTPUT_NAMES[0])
        snippets = result.as_numpy(OUTPUT_NAMES[1])
        assert len(embeddings) == len(snippets)
        print(f"Number of embeddings = {len(embeddings)}")

    print("================ Finished processing ============")
    triton_client.stop_stream()
