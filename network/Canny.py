"""
Canny Edge Detector
By Carlos Santiago Bañón
canny_edge_detector.py
Defines the Canny Edge Detector.
"""


import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from utils.util import convolve1d


class CannyEdgeDetector:

  def __init__(self, I, dimension, kernel_size=5, sigma=1, weak_value = 75,
               strong_value = 255, low_threshold_ratio = 0.05,
               high_threshold_ratio = 0.15):
    
    # Convolution Hyperparameters
    self.I = I
    self.dimension = dimension
    self.kernel_size = kernel_size
    self.sigma = sigma

    # Thresholding Hyperparameters
    self.weak_value = weak_value
    self.strong_value = strong_value
    self.low_threshold_ratio = low_threshold_ratio
    self.high_threshold_ratio = high_threshold_ratio

    # Intermediate Images
    self.I_smooth = None
    self.I_smooth_x = None
    self.I_smooth_y = None
    self.I_prime = None
    self.I_prime_x = None
    self.I_prime_y = None
    self.I_theta = None
    self.I_non_max = None
    self.I_hysteresis = None


  def gaussian_smoothing(self, I):
    """
    Perform 1-D Gaussian smoothing.
    :I: Input image.
    """

    # Define the range of the Gaussian distribution.
    r = np.linspace(-(self.kernel_size // 2), self.kernel_size // 2, self.kernel_size)

    # Calculate the Gaussian mask.
    G = [(np.exp((-x ** 2.0) / (2.0 * (self.sigma ** 2.0)))
          * (1 / (np.sqrt(2.0 * np.pi * (self.sigma ** 2.0))))) for x in r]

    # Convolve for both axes.
    I_smooth_x = convolve1d(I, G, axis=0)
    I_smooth_y = convolve1d(I, G, axis=1)
    
    # Combine both components.
    I_smooth = np.hypot(I_smooth_x, I_smooth_y).astype(int)

    return (I_smooth, I_smooth_x, I_smooth_y)


  def get_derivatives(self, I_x, I_y):
    """
    Get the edges using derivative masks.
    :I_x: x-component of the input image.
    :I_y: y-component of the input image.
    """

    if (self.dimension == 0):

      # Define the 1-D derivative mask.
      G = np.array([-1, 0, 1])

      # Compute the derivatives in both components.
      I_prime_x = convolve1d(I_x, G, axis=0)
      I_prime_y = convolve1d(I_y, G, axis=1)

    elif (self.dimension == 1):

      # Define the 2-D derivative masks.
      G_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
      G_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

      # Calculate magnitude of the input.
      I = np.hypot(I_x, I_y).astype(int)
      
      # Compute the derivatives in both components.
      I_prime_x = ndimage.filters.convolve(I, G_x)
      I_prime_y = ndimage.filters.convolve(I, G_y)

    else:
      print("Error: Invalid dimension.")
      return

    # Calculate the magnitude.
    I_prime = np.hypot(I_prime_x, I_prime_y).astype(int)
    I_prime = I_prime / I_prime.max() * 255

    # Calculate the direction.
    I_theta = np.arctan2(I_prime_y, I_prime_x).astype(int)

    return (I_prime, I_prime_x, I_prime_y, I_theta)


  def non_maximum_suppression(self, I, I_theta):
    """ 
    Perform non-maximum suppression.
    :I: Input image.
    :I_theta: Direction matrix for input image I.
    """

    # Set up the non-max matrix.
    height, weight = I.shape
    non_max = np.zeros((height, weight), dtype=np.int32)

    # Set up the relative angles.
    angle = I_theta * 180 / np.pi
    angle[angle < 0] += 180
    
    # Check the pixels in the directions relative to the angles.
    for i in range(1, height - 1):
      for j in range(1, weight - 1):
      
        pixel_1 = 255
        pixel_2 = 255
        
        # Check 0 degrees relative to the angle.
        if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
          pixel_1 = I[i, j + 1]
          pixel_2 = I[i, j - 1]

        # Check 45 degrees relative to the angle.
        elif (22.5 <= angle[i, j] < 67.5):
          pixel_1 = I[i + 1, j - 1]
          pixel_2 = I[i - 1, j + 1]

        # Check 90 degrees relative to the angle.
        elif (67.5 <= angle[i, j] < 112.5):
          pixel_1 = I[i + 1, j]
          pixel_2 = I[i - 1, j]

        # Check 135 degrees relative to the angle.
        elif (112.5 <= angle[i, j] < 157.5):
          pixel_1 = I[i - 1, j - 1]
          pixel_2 = I[i + 1, j + 1]

        # Update the non-max matrix accordingly.
        if (I[i, j] >= pixel_1) and (I[i, j] >= pixel_2):
          non_max[i, j] = I[i, j]
        else:
          non_max[i, j] = 0

    return non_max


  def hysteresis_threshold(self, I):
    """
    Perform hysteresis thresholding.
    :I: Input image.
    """

    # Set up the hysteresis matrix.
    height, width = I.shape
    I_hysteresis = np.zeros((height, width), dtype=np.int32)

    # Calculate the thresholds from the threshold ratios.
    high_threshold = I.max() * self.high_threshold_ratio;
    low_threshold = high_threshold * self.low_threshold_ratio;

    # Find the locations of weak and strong pixels.
    strong_values_x, strong_values_y = np.where(I >= high_threshold)
    weak_values_x, weak_values_y = np.where((I <= high_threshold) & (I >= low_threshold))

    # Update the hysteresis matrix accordingly.
    I_hysteresis[strong_values_x, strong_values_y] = self.strong_value
    I_hysteresis[weak_values_x, weak_values_y] = self.weak_value

    # Check the connected components and update the hysteresis matrix accordingly.
    for i in range(1, height - 1):
      for j in range(1, width - 1):
        if (I_hysteresis[i, j] == self.weak_value):

          # If a neighbor is strong, define the pixel as strong.
          if ((I_hysteresis[i + 1, j - 1] == self.strong_value)
            or (I_hysteresis[i + 1, j] == self.strong_value)
            or (I_hysteresis[i + 1, j + 1] == self.strong_value)
            or (I_hysteresis[i, j - 1] == self.strong_value)
            or (I_hysteresis[i, j + 1] == self.strong_value)
            or (I_hysteresis[i - 1, j - 1] == self.strong_value)
            or (I_hysteresis[i - 1, j] == self.strong_value)
            or (I_hysteresis[i - 1, j + 1] == self.strong_value)):
              I_hysteresis[i, j] = self.strong_value
          
          # If a neighbor is weak, turn off the pixel.
          else:
            I_hysteresis[i, j] = 0

    return I_hysteresis


  def detect_edges(self):

    # Perform Gaussian smoothing.
    self.I_smooth, self.I_smooth_x, self.I_smooth_y = self.gaussian_smoothing(self.I)
    plt.imshow(self.I_smooth_x, cmap='gray')
    plt.title('Image After Gaussian Smoothing (x)')
    plt.show()
    plt.imshow(self.I_smooth_y, cmap='gray')
    plt.title('Image After Gaussian Smoothing (y)')
    plt.show()
    plt.imshow(self.I_smooth, cmap='gray')
    plt.title('Image After Gaussian Smoothing')
    plt.show()

    # Get the derivatives using a 1-D derivative mask.
    self.I_prime, self.I_prime_x, self.I_prime_y, self.I_theta = self.get_derivatives(self.I_smooth_x, self.I_smooth_y)
    plt.imshow(self.I_prime_x, cmap='gray')
    plt.title('Image After Derivative Mask (x)')
    plt.show()
    plt.imshow(self.I_prime_y, cmap='gray')
    plt.title('Image After Derivative Mask (y)')
    plt.show()
    plt.imshow(self.I_prime, cmap='gray')
    plt.title('Image After Derivative Mask')
    plt.show()
    plt.imshow(self.I_theta, cmap='magma')
    plt.title('Image After Derivative Mask (Direction)')
    plt.show()

    # Perform non-maximum suppression.
    self.I_non_max = self.non_maximum_suppression(self.I_prime, self.I_theta)
    plt.imshow(self.I_non_max, cmap='gray')
    plt.title('Image After Non-Maximum Suppression')
    plt.show()

    # Perform hysteresis thresholding.
    self.I_hysteresis = self.hysteresis_threshold(self.I_non_max)
    plt.imshow(self.I_hysteresis, cmap='gray')
    plt.title('Image After Hysteresis Thresholding')
    plt.show()

    return self.I_hysteresis