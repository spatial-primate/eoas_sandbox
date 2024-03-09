import PIL
import numpy as np
# import pycwt as wavelet


class FourierPreprocessing:
    def __init__(self, filename, dx):
        self.image_filename = filename
        self.image = PIL.Image.open(filename)
        self.data = np.array(self.image)
        self.section = None
        self.no_mean = self.data - np.mean(self.data)
        self.polynomial_fit = None
        self.size = np.size(self.image)
        self.shape_x, self.shape_y = np.shape(self.data)
        self.dx = dx

    def taper(self, taper="tukey"):
        pass

    def make_bright_dark(self):
        """
        ideas:
            - linear 2%
            - edge: black, subject: white (bone-headed base case)
        :return: stretched-intensity image array

        """
        # todo: scipy's peak_local_max to locate markers for stretching
        pass

    def spatial_subset(self, x_range, y_range, along='x'):
        if 'y' in along.lower():
            self.section = self.image[y_range[0]:y_range[1], x_range[0]:]
        elif 'x' in along.lower():
            self.section = self.image[y_range[0]:, x_range[0]:x_range[1]]
        elif 'xy' in along.lower():  # cut out a box
            self.section = self.image[y_range[0]:y_range[1], x_range[0]:x_range[1]]
        else:
            raise Exception('Invalid range to subset image.')

    def demean(self, secular="linear"):
        if "linear" in secular.lower():
            self.no_mean = self.data - np.mean(self.data)
        elif "polynomial" in secular.lower():  # todo: polynomial degree depends on image...
            self.polynomial_fit = np.polyfit(np.arange(self.shape_x) + 1, self.data, deg=6)  # todo: include units
            self.no_mean = self.data - self.polynomial_fit
        else:
            raise NotImplementedError(f"Method {secular} not implemented")
