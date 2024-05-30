import nvidia.dali as dali
import nvidia.dali.types as types
from nvidia.dali.plugin.triton import autoserialize

@autoserialize
@dali.pipeline_def(batch_size=4, num_threads=4, device_id=0, output_dtype=[types.UINT8], output_ndim=[3])
def video_pipeline():
    sequence_length = 10
    original = dali.fn.experimental.inputs.video(name="video", sequence_length=sequence_length, device="mixed", last_sequence_policy='pad')
    original = original[:1, :]  # keep only 1 out of sequence_length frames
    original = dali.fn.transpose(original, perm=[0, 3, 1, 2])
    original = dali.fn.squeeze(original, axes=[0], name="original")

    return original
