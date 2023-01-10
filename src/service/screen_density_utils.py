import numpy as np
from service import constants
from service.utils import get_screen_resolution_in_xml

def calculate_ppi_of_screen(width, height, display_screen_size):
  """
  Calculate the PPI of the display
  base on width and height in pixels, display screen size in inches.
  """
  return np.sqrt(width ** 2 + height ** 2) / display_screen_size


def convert_pixels2inches(pixels, ppi):
  return pixels / ppi


def convert_font_height_px2dp(pixels, ppi):
  fontsize = round((pixels*160)/ppi, 1)
  return fontsize


def convert_font_height_px2pt(pixels, ppi):
  fontsize = round((pixels*163)/ppi, 1)
  return fontsize


def convert_font_height_px2mm(pixels):
  return round(pixels * 25.4 / 160, 1)


def convert_pixel2mm(d, ppi):
  return round(d/ppi * 25.4,1)


def inch_to_cm(value):
  return value * 2.54


def get_screen_density(xml_tree, screen_img, platform, pixel_map, display_screen_size, device):
  if platform == constants.Platform.ANDROID:
    height, width = screen_img.shape[:2]
  else:
    width, height = get_screen_resolution_in_xml(xml_tree, platform, pixel_map)
    if device in constants.EXCEPT_RESOLUTION_IPHONES:
      width, height = 1080, 1920
  return calculate_ppi_of_screen(width, height, display_screen_size)