from cv2 import matchTemplate, TM_CCORR_NORMED, imread
import numpy as np

class Template(object):
    def __init__(self, path, obj_type, stamp_dir=False):
        self.path = path
        self.obj_type = obj_type
        self.image = imread(path, 0)

        if stamp_dir:
            self._load_stamp(stamp_dir)
        else:
            self.stamp = 1

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

def tile(coords, grid):
    """
    Transform from pixel-level coordinate to location in a more coarse grid
    """
    new_y = np.floor(coords[0] / grid[0]).astype(np.int)
    new_x = np.floor(coords[1] / grid[1]).astype(np.int)

    return new_y, new_x

def object_locations(template, image, threshold):
    correlations = matchTemplate(image, template, TM_CCORR_NORMED)
    return np.where(correlations > threshold)

def _create_stamps(template, image, coordinates):

    canvas = np.zeros_like(image)
    height, width = template.shape

    # calculate offsets based on image boundaries
    y_offset = np.minimum(coordinates[0] + height, image.shape[0])
    x_offset = np.minimum(coordinates[1] + width, image.shape[1])

    for i, (y, x) in enumerate(coordinates):
        canvas[y:y_offset[i], x:x_offset[i]] = template.stamp

    return canvas

def mask(templates, image, threshold):
    channels = np.zeros((image.shape[0], image.shape[1], 3))
    for template in templates:
        coords = object_locations(template.image, image, threshold)
        obj_channel = _create_stamps(template, image, coords)
        channels[:,:,template.channel] += obj_channel

    return channels

def location_features(template, observation, grid, threshold):

    assert type(grid) == type(np.array([0])), type(grid)
    assert observation.shape[0] % grid[0] == 0
    assert observation.shape[1] % grid[1] == 0

    loc = object_locations(observation, template, threshold)
    tiled = tile(loc, grid=grid)

    _new_shape = np.divide(observation.shape, grid).astype(np.int)
    _zeros = np.zeros(_new_shape)
    _zeros[tiled] = 1
    flat = _zeros.flatten()

    return flat

def extract_features(observation, templates, grid, threshold):

    locs = []
    for k, v in templates.items():
        _features = location_features(v, observation, grid, threshold)
        locs.append(_features)

    locs = np.concatenate(locs)

    return locs