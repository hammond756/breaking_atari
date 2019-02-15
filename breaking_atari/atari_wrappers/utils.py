from cv2 import matchTemplate, TM_CCORR_NORMED, imread
import numpy as np
import os

class Template(object):
    def __init__(self, path, obj_type, stamp_dir=False):

        assert os.path.isfile(path), "Invalid image path in Template: {}".format(path)

        self.path = path
        self.obj_type = obj_type
        self.image = imread(path, 0)

        if stamp_dir:
            self._load_stamp(stamp_dir)
        else:
            self.stamp = 255

    def _load_stamp(self, stamp_dir):
        self.stamp = imread(stamp_dir + self.obj_type + '.png', 0)

    @property
    def has_stamp(self):
        return type(self.stamp) != int

    @property
    def channel(self):
        return {
            'barrier' : 2,
            'enemy' : 0,
            'agent' : 1
        }[self.obj_type]

    @property
    def shape(self):
        if self.has_stamp:
            return self.stamp.shape
        else:
            return self.image.shape

def tile(coordinates, grid):
    """
    Transform from pixel-level coordinate to location in a more coarse grid
    """
    new_y = np.floor(coordinates[0] / grid[0]).astype(np.int)
    new_x = np.floor(coordinates[1] / grid[1]).astype(np.int)

    return new_y, new_x

def object_locations(template_im, image, threshold):
    assert len(template_im.shape) == len(image.shape), "Template and image don't have matching dimensions: Template {} - Image {}".format(template_im.shape, image.shape)

    correlations = matchTemplate(image, template_im, TM_CCORR_NORMED)
    return np.where(correlations > threshold)

def create_stamps(template, image, coordinates):

    canvas = np.zeros_like(image)
    height, width = template.shape

    y, x = coordinates

    # calculate offsets based on image boundaries
    y_offset = np.minimum(y + height, image.shape[0])
    x_offset = np.minimum(x + width, image.shape[1])

    for i in range(y.size):
        canvas[y[i]:y_offset[i], x[i]:x_offset[i]] = template.stamp

    return canvas

def mask(templates, image, threshold):
    channels = np.zeros((image.shape[0], image.shape[1], 3))
    for template in templates:
        coordinates = object_locations(template.image, image.squeeze(), threshold)
        obj_channel = create_stamps(template, image.squeeze(), coordinates)
        channels[:,:,template.channel] += obj_channel

    return channels.astype(np.uint8)

def location_features(template, observation, grid, threshold):

    assert type(grid) == type(np.array([0])), type(grid)
    assert observation.shape[0] % grid[0] == 0
    assert observation.shape[1] % grid[1] == 0

    loc = object_locations(observation, template, threshold)
    tiled = tile(loc, grid=grid)

    new_shape = np.divide(observation.shape, grid).astype(np.int)
    zeros = np.zeros(new_shape)
    zeros[tiled] = 1
    flat = zeros.flatten()

    return flat

def compute_bunker_health(template, image):
    health = []
    bunker_y, bunker_x = np.array([156, 156, 156]), np.array([ 42,  74, 106])

    y_offset = bunker_y + template.shape[0]
    x_offset = bunker_x + template.shape[1]

    for i in range(3):
        bunker_cutout = image[bunker_y[i]:y_offset[i], bunker_x[i]:x_offset[i]]
        health.append(matchTemplate(bunker_cutout, template, TM_CCORR_NORMED)[0,0])

    return np.array(health)

def compute_agent_x_position(template, image):
    agent_y = 185
    agent_cutout = image[agent_y:agent_y+10, :]
    correlations = matchTemplate(agent_cutout, template, TM_CCORR_NORMED)
    y, x = np.where(correlations > 0.98)
    return x

def extract_features(observation, templates, grid, threshold):

    locs = []
    for k, v in templates.items():
        _features = location_features(v, observation, grid, threshold)
        locs.append(_features)

    locs = np.concatenate(locs)

    return locs