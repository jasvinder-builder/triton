import sys
import queue

from functools import partial
from pathlib import Path
from typing import Tuple

import faiss
import numpy as np
from PIL import Image

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

# TODO: Fix the way these params are defined both on the triton side and here
VIDEO_PIPELINE = {
    "model": "video_processing_service",
    "input": "video",
    "input_type": "UINT8",
    "outputs": ["embeddings", "snippets"],
}

VIDEO_GRID_PIPELINE = {
    "model": "video_grid_processing_service",
    "input": "video",
    "input_type": "UINT8",
    "outputs": ["embeddings", "snippets"],
}

IMAGE_PIPELINE = {
    "model": "image_embedding_service",
    "input": "image",
    "input_type": "UINT8",
    "outputs": ["embedding"],
}

EMB_DIM = 512


class ProcessingEngine:
    def __init__(self, url: str = "localhost:8001"):
        try:
            self.client = grpcclient.InferenceServerClient(
                url=url,
                verbose=False,
                ssl=False,
                root_certificates=None,
                private_key=None,
                certificate_chain=None,
            )
        except Exception as e:
            print("Client creation failed: " + str(e))
            sys.exit(1)

        if not self.client.is_server_live():
            print("FAILED : is_server_live")
            sys.exit(1)

        if not self.client.is_server_ready():
            print("FAILED : is_server_ready")
            sys.exit(1)

        # queue for storing streaming results
        self.results = queue.Queue()

    def process_image(self, image_file_path: str | Path):
        inputs, outputs = self._prepare_inputs_and_outputs_for_inference(
            file_path=image_file_path, pipeline=IMAGE_PIPELINE
        )

        # Run the inference
        result = self.client.infer(
            model_name=IMAGE_PIPELINE["model"], inputs=inputs, outputs=outputs
        )
        embedding = result.as_numpy(IMAGE_PIPELINE["outputs"][0])
        return embedding

    def process_video(self, video_file_path: str | Path, grid=False) -> Tuple:
        def callback(q, res, err):
            if err:
                q.put(err)
            else:
                q.put(res)

        # Start the stream
        self.client.start_stream(callback=partial(callback, self.results))

        inputs, outputs = self._prepare_inputs_and_outputs_for_inference(
            file_path=video_file_path, pipeline=VIDEO_PIPELINE
        )

        # Run the async inference
        model = VIDEO_GRID_PIPELINE["model"] if grid else VIDEO_PIPELINE["model"]
        inference_type = "grid_based_clip" if grid else "person_based_clip"
        print(f"Grid is {grid} therefore running {inference_type}")
        self.client.async_stream_infer(
            model_name=model,
            inputs=inputs,
            request_id="0",
            outputs=outputs,
            enable_empty_final_response=True,
        )

        # Make folder to save snippets
        video_path = Path(video_file_path)
        snippets_folder = video_path.parent / f"{video_path.stem}_snippets"
        snippets_folder.mkdir(parents=True, exist_ok=True)

        # Fetch results as they come
        all_embeddings = []
        emb_index = 0
        while True:
            result = self.results.get()
            if isinstance(result, InferenceServerException):
                raise result
            response = result.get_response()
            if response.parameters.get("triton_final_response").bool_param:
                break
            embeddings = result.as_numpy(VIDEO_PIPELINE["outputs"][0])
            snippets = result.as_numpy(VIDEO_PIPELINE["outputs"][1])
            assert len(embeddings) == len(snippets)
            for i, emb in enumerate(embeddings):
                image = Image.fromarray(
                    snippets[i].transpose(1, 2, 0).astype(np.uint8), "RGB"
                )
                image.save(snippets_folder / f"{emb_index}.jpg")
                all_embeddings.append(emb)
                emb_index += 1

        # At this point we have saved all snippets and have all embeddings ready,
        # let's form a faiss index and save it
        # index = faiss.IndexFlatL2(EMB_DIM)
        index = faiss.IndexHNSWFlat(EMB_DIM, 16)
        index.add(np.stack(all_embeddings))
        video_path = Path(video_file_path)
        index_file_path = video_path.parent / f"{video_path.stem}_faiss.index"
        faiss.write_index(index, str(index_file_path))
        assert Path.exists(index_file_path)
        return snippets_folder, index_file_path

    @staticmethod
    def search_embeddings(query: np.ndarray, index_path: str | Path, k=100):
        print(f"query={query.shape}, index_path = {index_path}")
        index = faiss.read_index(str(index_path))
        D, I = index.search(query, k=k)
        return np.squeeze(I), np.squeeze(D)

    @staticmethod
    def _prepare_inputs_and_outputs_for_inference(
        file_path: str | Path, pipeline
    ) -> Tuple:
        buffer = np.fromfile(file_path, dtype=np.uint8)
        buffer = np.expand_dims(buffer, axis=0)
        input_shape = list(buffer.shape)
        inputs, outputs = [], []
        inputs.append(
            grpcclient.InferInput(
                pipeline["input"], input_shape, pipeline["input_type"]
            )
        )
        inputs[0].set_data_from_numpy(buffer)
        for output_name in pipeline["outputs"]:
            outputs.append(grpcclient.InferRequestedOutput(output_name))
        return inputs, outputs
