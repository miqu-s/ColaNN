import tensorflow as tf

# this stuff was used for debugging lol

def preprocess_image(image):
    image = image
    return image

def preprocess(color, depth, rendered):
    color = preprocess_image(color)
    depth = preprocess_image(depth)
    rendered = preprocess_image(rendered)
    return color, depth, rendered