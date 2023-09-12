import sys
import numpy

import bob.io.base
import bob.io.image
import bob.ip.facedetect
from bob.ip.skincolorfilter import SkinColorFilter
from PIL import Image

#face_image = bob.io.base.load('test.jpg')
face_image = Image.open('/homes/bacharya/skin/test.jpg')
face_image = numpy.asarray(face_image)
face_image = numpy.moveaxis(face_image, -1, 0)

detection = bob.ip.facedetect.detect_single_face(face_image)
bounding_box, quality = bob.ip.facedetect.detect_single_face(face_image)
face = face_image[:, bounding_box.top:bounding_box.bottom, bounding_box.left:bounding_box.right]
skin_filter = SkinColorFilter()
skin_filter.estimate_gaussian_parameters(face)
skin_mask = skin_filter.get_skin_mask(face, 0.3)
skin_image = numpy.copy(face)
skin_image[:, numpy.logical_not(skin_mask)] = 0

from matplotlib import pyplot

f, ax = pyplot.subplots(1, 1)
ax.set_title('Original Image')
ax.set_xticks([])
ax.set_yticks([])
ax.imshow(numpy.rollaxis(numpy.rollaxis(face, 2),2))
pyplot.show()

f, ax = pyplot.subplots(1, 1)
ax.set_title('Detected skin pixels')
ax.set_xticks([])
ax.set_yticks([])
ax.imshow(numpy.rollaxis(numpy.rollaxis(skin_image, 2),2))
pyplot.show()

