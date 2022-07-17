import numbers
import numpy as np
import functional as F

from PIL import Image, ImageFilter


class Map(object):
    """Re-maps a number from one range to another.
    Args:
        mapping (dict): Dictionary with the new range of values.
    """

    def __init__(self, mapping=None):
        self.mapping = mapping

    def __call__(self, value):
        """
        Args:
            value (int): Value to be mapped.
        Returns:
            int: Mapped value.
        """
        return F.map(value, self.mapping)

    def __repr__(self):
        return self.__class__.__name__ + '(mapping={0})'.format(self.mapping)


class GaussianNoise(object):

    def __init__(self, mean=0.0, stdv=0.1):
        self.mean = mean
        self.stdv = stdv

    def __call__(self, image):
        npimg = np.array(image) / 255.0
        noise = np.random.normal(loc=self.mean, scale=self.stdv, size=npimg.shape)
        npimg = np.clip(npimg + noise, 0.0, 1.0) * 255.0
        return Image.fromarray(npimg.astype(np.uint8))

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, stdv={1})'.format(self.mean, self.stdv)


class GaussianBlur(object):

    def __init__(self, sigma):
        self.sigma = sigma

        if isinstance(sigma, numbers.Number):
            self.min_sigma = sigma
            self.max_sigma = sigma
        elif isinstance(sigma, list):
            if len(sigma) != 2:
                raise Exception("`sigma` should be a number or a list of two numbers")
            if sigma[1] < sigma[0]:
                raise Exception(
                    "radius[0] should be <= radius[1]")
            self.min_sigma = sigma[0]
            self.max_sigma = sigma[1]
        else:
            raise Exception("`sigma` should be a number or a list of two numbers")

    def __call__(self, image):
        sigma = np.random.uniform(self.min_sigma, self.max_sigma)
        sigma = max(0.0, sigma)
        ksize = int(sigma+0.5) * 8 + 1
        return image.filter(ImageFilter.GaussianBlur(ksize))

    def __repr__(self):
        return self.__class__.__name__ + '(sigma={0})'.format(self.sigma)


class RandomRotation(object):

    def __init__(self, mean=0.0, stdv=0.1):
        self.mean = mean
        self.stdv = stdv

    def __call__(self, image):
        theta = np.random.normal(loc=self.mean, scale=self.stdv)
        cos = np.cos(theta)
        sin = np.sin(theta)
        return image.transform(image.size,
                               Image.AFFINE,
                               (cos, sin, 0.0, -sin, cos, 0.0),
                               resample=Image.BICUBIC)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, stdv={1})'.format(self.mean, self.stdv)


class RandomAffine(object):

    def __init__(self, mean=0.0, stdv=0.1):
        self.mean = mean
        self.stdv = stdv

    def __call__(self, image):
        a = 1 + np.random.normal(loc=self.mean, scale=self.stdv)
        b = np.random.normal(loc=self.mean, scale=self.stdv)
        c = np.random.normal(loc=self.mean, scale=self.stdv)
        d = 1 + np.random.normal(loc=self.mean, scale=self.stdv)
        return image.transform(image.size,
                               Image.AFFINE,
                               (a, b, 0.0, c, d, 0.0),
                               resample=Image.BICUBIC)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, stdv={1})'.format(self.mean, self.stdv)

