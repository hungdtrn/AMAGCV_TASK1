from .datasets import LoadImagesAndLabels

class CustomLoadImagesAndLabels(LoadImagesAndLabels):
    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False, cache_images=False, single_cls=False, stride=32, pad=0, prefix=''):
        super().__init__(path, img_size, batch_size, augment, hyp, rect, image_weights, cache_images, single_cls, stride, pad, prefix)
        