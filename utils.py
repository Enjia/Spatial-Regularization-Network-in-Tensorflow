import numpy as np

def inv_preprocess(imgs, num_images):
    n, h, w, c = imgs.shape
    assert(n >= num_images), "Batch size %d should be greater or equal than number of images to save %d." % (n, num_images)
    outputs = np.zeros((num_images, h, w, c), dtype=np.uint8)
    for i in range(num_images):
        outputs[i] = (imgs[i]+127)[:, :, ::-1].astype(np.uint8)
    return outputs