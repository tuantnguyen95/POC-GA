import os
import argparse
from imgaug import augmenters as iaa
import cv2
from tqdm import tqdm


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('img_dir',
                      help='Parent directory that contains sub-dirs represent classes')
  parser.add_argument('--saturate',
                      action='store_true',
                      help='Increase saturation of the images')
  parser.add_argument('--jpeg_compress',
                      action='store_true',
                      help='Apply jpeg compression to the images')
  parser.add_argument('--bright',
                      action='store_true',
                      help='Increase brightness of the images')
  args = parser.parse_args()
  return args


def get_imgs(absolute_path, files):
  img_paths = []
  imgs = []
  for file in files:
    if file.endswith('.png') and not file.startswith('augmented'):
      img_path = os.path.join(absolute_path, file)
      img = cv2.imread(img_path)
      img_paths.append(file)
      imgs.append(img)
  return imgs, img_paths


def apply_augmentation(imgs, saturate=False, bright=False, jpeg_compress=False):
  augmented_imgs = {}
  if saturate:
    saturate_aug = iaa.imgcorruptlike.Saturate(severity=1)
    saturate_imgs = saturate_aug(images=imgs)
    augmented_imgs['saturate'] = saturate_imgs
  if bright:
    bright_aug = iaa.imgcorruptlike.Brightness(severity=1)
    bright_imgs = bright_aug(images=imgs)
    augmented_imgs['bright'] = bright_imgs
  if jpeg_compress:
    jpeg_compress_aug = iaa.imgcorruptlike.JpegCompression(severity=1)
    jpeg_compress_imgs = jpeg_compress_aug(images=imgs)
    augmented_imgs['jpeg_compress'] = jpeg_compress_imgs
  return augmented_imgs


def write_imgs(absolute_path, img_paths, augmented_imgs):
  jpeg_compresses = augmented_imgs.get('jpeg_compress')
  saturates = augmented_imgs.get('saturate')
  brights = augmented_imgs.get('bright')
  pbar = tqdm(img_paths)
  class_name = absolute_path.split('/')[-1]
  for idx, filename in enumerate(pbar):
    pbar.set_description("Processing class %s" % class_name)
    if jpeg_compresses:
      jpeg_compress_path = os.path.join(absolute_path, 'augmented_jpg_compress_' + filename)
      cv2.imwrite(jpeg_compress_path, jpeg_compresses[idx])
    if saturates:
      saturate_path = os.path.join(absolute_path, 'augmented_saturate_' + filename)
      cv2.imwrite(saturate_path, saturates[idx])
    if brights:
      bright_path = os.path.join(absolute_path, 'augmented_bright_' + filename)
      cv2.imwrite(bright_path, brights[idx])


def augment_images():
  args = parse_args()
  for r, d, f in os.walk(args.img_dir):
    imgs, img_paths = get_imgs(r, f)
    augmented_imgs = apply_augmentation(imgs, saturate=args.saturate, bright=args.bright,
                                        jpeg_compress=args.jpeg_compress)
    write_imgs(r, img_paths, augmented_imgs)


if __name__ == '__main__':
  augment_images()
