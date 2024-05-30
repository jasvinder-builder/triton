import nvidia.dali as dali
import nvidia.dali.types as types
from nvidia.dali.plugin.triton import autoserialize

@autoserialize
@dali.pipeline_def(batch_size=8, num_threads=4, device_id=0, output_dtype=[types.UINT8, types.FLOAT], output_ndim=[3, 3])
def video_pipeline():
    # https://docs.nvidia.com/deeplearning/dali/user-guide/docs/operations/nvidia.dali.fn.experimental.inputs.video.html#nvidia.dali.fn.experimental.inputs.video
    # streams and decodes encoded video from a memory buffer, returns a batch of sequences of frames with layout (F, H, W, C), see above link for a picture
    sequence_length = 5
    original = dali.fn.experimental.inputs.video(name="video", sequence_length=sequence_length, device="mixed", last_sequence_policy='pad')
    # fn.readers.video reads from disk instead of a memory buffer
    # fn.decoders.video decodes the full video in one go instead of streaming
    original = original[:1, :]  # keep only 1 out of sequence_length frames

    # resize to 640x640 for yolo and normalize pixels to 0-1 range
    frames = dali.fn.resize(original, resize_x=640, resize_y=640, device="gpu")
    frames = dali.fn.color_space_conversion(frames, image_type=types.BGR, output_type=types.RGB, device="gpu")
    frames = frames / 255.0
    frames = dali.fn.transpose(frames, perm=[0, 3, 1, 2])
    frames = dali.fn.squeeze(frames, axes=[0], name="frames")

    # transpose and squeeze original too
    original = dali.fn.transpose(original, perm=[0, 3, 1, 2])
    original = dali.fn.squeeze(original, axes=[0], name="original")

    return original, frames
