from skimage.segmentation import mark_boundaries
from lime import lime_image

def load_img(path, img_size):
    '''preprocessess img to be used by the explainer'''
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_size)

    return img