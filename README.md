# Triton Server Based Prototyping

This repository showcases use of open source tools like nvidia [triton-server](https://github.com/triton-inference-server/server) [[FAQ](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/faq.html#)], [dali](https://github.com/triton-inference-server/dali_backend), [faiss](https://github.com/facebookresearch/faiss), [deepstream](https://developer.nvidia.com/blog/building-iva-apps-using-deepstream-5-0-updated-for-ga/) to implement a bare-bones video analytics app (using [streamlit](https://streamlit.io/)) with the following functionality:
1. Upload videos for processing (where processing involves detecting persons (and ~80 other [COCO classes](https://github.com/amikelive/coco-labels/blob/master/coco-labels-2014_2017.txt)) and then computing and saving CLIP embeddings for detected persons.
2. Upload person identities in the form of images.
3. Search a person identity in a video.
4. [WIP] Processing a live camera stream and alerting for person identities of interest at ingest time.
5. [UPDATE] Videos can be processed using the grid version of CLIP as well, where instead of running a detector, we will simply divide the image into 200x200 blocks and compute the CLIP embedding for each block.

## Triton Pipelines
- Video ingestion happens through this [ensemble pipeline](https://github.com/jasvinder-builder/triton/blob/master/model_repository/video_processing_service/config.pbtxt)
- Image/identity processing happens through this [ensemble pipeline](https://github.com/jasvinder-builder/triton/blob/master/model_repository/image_embedding_service/config.pbtxt)
- The individual building blocks of these pipelines are:
    - `DALI` based [detector preprocessing block](https://github.com/jasvinder-builder/triton/blob/master/model_repository/detector_preprocess/1/dali.py) that also does [efficient video decoding 
using a mix of CPU and GPU](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/operations/nvidia.dali.fn.experimental.inputs.video.html)
    - [YOLOV7 TensorRT model](https://github.com/jasvinder-builder/triton/blob/master/model_repository/yolov7/config.pbtxt) for doing detections where the model was downloaded and prepared using these [instructions](https://github.com/jasvinder-builder/triton/blob/master/model_repository/yolov7/1/model.plan.notes)
    - [Detector post-processing](https://github.com/jasvinder-builder/triton/blob/master/model_repository/detector_postprocess/1/model.py) based on triton python-backend
    - [WIP] Alternative detector post-processing blocks based on `CUPY` and `DALI` that obviate the need of bringing data to the CPU
    - [CLIP TensorRT model](https://github.com/jasvinder-builder/triton/blob/master/model_repository/clip_visual/config.pbtxt) downloaded and prepared using these [instructions](https://github.com/jasvinder-builder/triton/blob/master/model_repository/clip_visual/1/model.plan.notes)
    - `DALI` based [CLIP Preprocessing block](https://github.com/jasvinder-builder/triton/blob/master/model_repository/clip_preprocess/1/dali.py) (only used in image identity pipeline)
- For enabling search, all the embeddings per video are [added to a FAISS index and saved to disk](https://github.com/jasvinder-builder/triton/blob/master/processing_engine.py#L113), then during search time, the index is reloaded from disk and used to compare against the query embedding.
 
## Running Triton

- First prepare a Docker image using these [instructions](https://github.com/jasvinder-builder/triton/blob/master/Dockerfile.notes)
- Get a GPU machine with nvidia drivers installed (example: 192.168.11.91)
- Sample run command:
```
docker run --gpus=all --shm-size=6g --rm --net=host --ipc=host -p8000:8000 -p8001:8001 -p8002:8002 -v /home/jasvindersingh/work/triton/model_repository/:/models my-triton-dali tritonserver --cuda-memory-pool-byte-size 0:8589934592 --model-repository=/models
```
- [WIP] Add more notes

## Running the App
- [WIP] Update the reuirements.txt and add virtual env instructions
- Run ``` streamlit run app.py --server.maxUploadSize 2000```

## More Detailed Notes
- [WIP]
- **Video Decoding**: [blaze](https://github.com/jasvinder-builder/blaze-worker/blob/master/src/framesources/video_capture.cpp#L62) versus [DALI](https://github.com/NVIDIA/DALI/blob/708af6048e71e55eb117718a53c8bf7649eecb38/dali/operators/reader/loader/video/frames_decoder.cc#L455)- the latter supports [mixed/GPU assisted decoding](https://github.com/NVIDIA/DALI/blob/708af6048e71e55eb117718a53c8bf7649eecb38/dali/operators/reader/loader/video/frames_decoder_gpu.cc#L592) using nvdecode as well. One current limitation of `DALI` seems to be that the encoded video must be first [loaded entirely in memory](https://github.com/jasvinder-builder/triton/blob/master/processing_engine.py#L129) before decoding (note that the entire decoded video will not be kept in memory, decoding will be done in a [streaming fashion](https://github.com/jasvinder-builder/triton/blob/master/model_repository/detector_preprocess/1/dali.py#L9) and frames will be returned).

## Deepstream
- Deepstream is running as a separate docker and [acting as a triton-client](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinferserver.html#deepstream-triton-grpc-support) communicating with the triton-server which is serving a lightweight face detection [centerface](https://github.com/jasvinder-builder/triton/tree/master/model_repository/centerface) model.
- Starting deepstream: ```docker run --gpus "device=0" -it --net=host -p 8554:8554 --rm -v /home/jasvindersingh/work/triton:/my_triton -e CUDA_CACHE_DISABLE=0 my-deepstream7 ```
- Running the deepstream app from inside the above docker: ```deepstream-app -c /my_triton/deepstream/config/app_config.txt ``` (output gets stored as *out.mp4* inside the config folder, but can also be served as an RTSP stream).
- Also, we can change the source in the app_config.txt to an RTSP stream as well.
- [WIP] Instead of simply displaying the output, if we want to do something more interesting like live `CLIP queries based detection`, then we need to add probes and get access to the returned inference metadata, something along the lines of [this](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/blob/8178b5e611ccdc23bef39caf2f1f07b14d39a7cb/apps/deepstream-ssd-parser/deepstream_ssd_parser.py#L264)
 
## Architecture Diagrams

### Triton Server
![image](https://github.com/jasvinder-builder/triton/assets/15894486/72f7c120-ae69-461a-b257-f9ccd7e16869)

### Triton And Deepstream
![image](https://github.com/jasvinder-builder/triton/assets/15894486/15ae155c-2ba2-4d29-8970-33e6aab1bf3b)


