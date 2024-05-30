from nvidia.dali import fn
from nvidia.dali import pipeline_def
from nvidia.dali.plugin.triton import autoserialize
import nvidia.dali.types as types


@autoserialize
@pipeline_def(batch_size=256, num_threads=4, device_id=0)
def pipe(hw_decoder_load=0.8):
    images = fn.external_source(device="cpu", name="image")
    images = fn.decoders.image(images, device="mixed", output_type=types.RGB, hw_decoder_load=hw_decoder_load)
    images = fn.resize(images, resize_x=224, resize_y=224)
    images = fn.crop_mirror_normalize(images,
                                           dtype=types.FLOAT,
                                           output_layout="CHW",
                                           crop=(224, 224),
                                           mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                           std=[0.229 * 255, 0.224 * 255, 0.225 * 255], name="preprocessed")
    return images
