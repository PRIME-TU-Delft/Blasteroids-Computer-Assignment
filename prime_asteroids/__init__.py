"""
This package provides the framework that should be used for the Asteroids Computer Assignment.
A solution code package is also provided.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# Class: AsteroidsImage
#
# Provides a frame around the concept of an image and the operations applied to it.
class AsteroidsImage:
    """
    This class is meant as a wrapper around an image and provides an abstraction
    that allows the students to think of the image as its mathematical representation,
    instead of the way computers normally represent images.

    Keep in mind that the operations that the students want to perform use cartesian
    coordinates, instead of the normal computer-graphics coordinate-system. This class
    provides translation functionality.
    """

    def __init__(self):
        self.image = np.zeros((100, 100, 3))
        self.width = 100
        self.height = 100

    # Provide a function for loading an image
    def load_image(self, image_name):
        # Start by opening the requested image-file as a PIL.Image
        # If the file doesn't exist, PIL will throw an exception.
        pil_image = Image.open(image_name).convert("RGB")

        # Now represent the image as a numpy-array
        numpy_image = np.asarray(pil_image)

        # During this assignment, we will often treat this image as if
        # it lived in a cartesian coordinate system, where the center pixel
        # is located at coordinate (0, 0). Since we don't want to bother
        # students with unpredictable effects of sampling the image with
        # the new coordinate system, we simply make sure that the number of
        # pixel-rows and -columns is odd. Only then, a center pixel exists.
        # 
        # If number of rows is even
        if numpy_image.shape[0] % 2 == 0:
            # Remove one row to make it odd
            numpy_image = np.delete(numpy_image, numpy_image.shape[0]-1, 0)
        
        # If number of columns is even
        if numpy_image.shape[1] % 2 == 0:
            # Remove one column to make it odd
            numpy_image = np.delete(numpy_image, numpy_image.shape[1]-1, 1)
            

        # And finally: normalise the image's color values.
        # Since color is stored 8-bit numbers, the maximum color value is 255 (2^8 - 1).
        # After normalising, store the image in this class instance as 'self.image'.
        self.image = numpy_image / 255.0

        # Also store the dimensions of this image (width and height)
        self.width = self.image.shape[1]
        self.height = self.image.shape[0]

        if self.width > 300 or self.height > 300:
            print("For quicker results use images of size 300x300 or smaller.")

        # Once again assert that this width and height are odd
        assert self.width % 2 != 0 and self.height % 2 != 0, \
            "Expected odd image dimensions, but got: width=" + \
            str(self.width) + " and height=" + str(self.height)
        
        print(self.image.shape)

    def __cartesian_to_viewport(self, x, y):
        # Careful: in the initialisation of this class, we made sure that the
        # dimensions of the image are odd, to make this operation straight-forward.

        viewport_x =  x + math.floor(self.width / 2)
        viewport_y = -y + math.floor(self.height / 2)

        return viewport_x, viewport_y

    def __viewport_to_cartesian(self, x, y):
        # Careful: in the initialisation of this class, we made sure that the
        # dimensions of the image are odd, to make this operation straight-forward.

        cartesian_x = x - math.floor(self.width / 2)
        cartesian_y = -(y - math.floor(self.height / 2))

        return cartesian_x, cartesian_y
    
    def copy(self):
        """
        Returns a copy of the object
        """
        result = AsteroidsImage()
        result.image = self.image.copy()
        result.width = self.width
        result.height = self.height

        return result
    
    def clear_image(self):
        self.image = np.zeros(self.image.shape)

    def is_inside_image(self, x, y):
        """
        Return true iff these coordinates fall within the bounds of the image
        """

        vp_x, vp_y = self.__cartesian_to_viewport(x, y)
        return 0 <= vp_x < self.width and 0 <= vp_y < self.height
    
    def all_x_values(self):
        """
        Returns a list of all x coordinates.
        """
        return(range(-math.floor(self.width / 2), math.floor(self.width / 2) + 1))
    
    def all_y_values(self):
        """
        Returns a list of all y coordinates.
        """
        return range(-math.floor(self.height / 2), math.floor(self.height / 2) + 1)

    def get_pixel_at(self, x, y):
        """
        This method provides an important abstraction layer. It translates the given
        (x, y) from cartesian coordinates to the viewport coordinate-system and returns
        the corresponding pixel-value.
        """

        vp_x, vp_y = self.__cartesian_to_viewport(x, y)
        return self.image[vp_y, vp_x]
    
    def set_pixel_at(self, x, y, value):
        """
        This method provides an important abstraction layer. It translates the given
        (x, y) from cartesian coordinates to the viewport coordinate-system and updates
        the corresponding pixel value
        """
        vp_x, vp_y = self.__cartesian_to_viewport(x, y)
        self.image[vp_y, vp_x] = value

    def plot(self):
        """
        Use Matplotlib to display the (rotated) image in a plot
        """
        plt.xticks([])
        plt.yticks([])
        plt.imshow(self.image)
        plt.show()

    def clamp(self, x, y):
        return \
            min(math.floor(self.width / 2), max(-math.floor(self.width / 2), x)),\
            min(math.floor(self.height / 2), max(-math.floor(self.height / 2), y))

    def compute_rotation_matrix(self, theta):
        """
        This method creates the rotation matrix for the given angle and returns it.
        """

        # Rotating an image is done by multiplying each pixel-position by a rotation-matrix.
        # It is important to realise that we therefore view pixel-positions as vectors with
        # an (x, y) coordinate. The resulting vector tells us where the pixel ends up after rotation.
        rotation_matrix = np.array([
            [math.cos(theta), -math.sin(theta)],
            [math.sin(theta),  math.cos(theta)],
        ])

        return rotation_matrix
    

    def rotate_forward(self, theta):

        # Make a copy of the current image, because we will be overwriting values in the image
        image_copy = self.copy()

        # Clear the stored image
        self.clear_image()

        # Calculate the rotation matrix
        rotation_matrix = self.compute_rotation_matrix(theta)

        # We are doing a FORWARD ROTATION. This means that we iterate over all pixels in the
        # original image, find the new location for those pixels and insert them accordingly.
        for x in self.all_x_values():
            for y in self.all_y_values():

                # Grab the pixel that is stored at this location
                pixel_value = image_copy.get_pixel_at(x, y)

                # Create a vector with these coordinates
                vector = np.array([x, y])

                # Multiply this vector by the rotation matrix to obtain this
                # pixel's location in the rotated image
                vector_rotated = np.matmul(rotation_matrix, vector)

                # The rotated x, y coordinates can be extracted
                x_rotated = vector_rotated[0]
                y_rotated = vector_rotated[1]

                # These rotated coordinates will likely not exactly fall exactly on a pixel.
                # Find the pixel that is closest to these coordinates (round the numbers)
                x_closest = round(x_rotated)
                y_closest = round(y_rotated)

                # It may occur that this new pixel location falls outside the bounds of our image!
                # Try rotating two pieces of paper on top of each other. You'll see that the corners
                # will fall 'outside of the bounds of the other sheet of paper'
                if self.is_inside_image(x_closest, y_closest):
                    # Now, insert this pixel's value in the new location
                    self.set_pixel_at(x_closest, y_closest, pixel_value)



    def rotate_backward(self, theta):

        # Make a copy of the current image, because we will be overwriting values in the image
        image_copy = self.copy()

        # Clear the stored image
        self.clear_image()

        # Calculate the INVERSE rotation matrix
        # We can simply use negative theta to obtain the inverse
        # We'll need the inverse, because this time we're approaching this problem backwards!
        inv_rotation_matrix = self.compute_rotation_matrix(-theta)

        # We are doing a BACKWARD ROTATION. This means that we iterate over all pixels in the
        # RESULT image, find the ORIGINAL location for those pixels and determine the resulting
        # color that way. Use bilinear sampling to sample the original image.
        for x in self.all_x_values():
            for y in self.all_y_values():

                # Create a vector with these coordinates
                vector = np.array([x, y])

                # Multiply this vector by the rotation matrix to obtain this
                # pixel's location in the ORIGINAL image
                vector_inv_rotated = np.matmul(inv_rotation_matrix, vector)

                # The rotated x, y coordinates can be extracted
                x_rotated = vector_inv_rotated[0]
                y_rotated = vector_inv_rotated[1]

                #
                # Imagine that coordinate (x_rotated, y_rotated) falls between four pixel-centers:
                #       alpha
                #       O -|--- O
                #   beta|  |    |
                #       ---X-----
                #       |  |    |
                #       O--|----O
                #
                # We can compute alpha and beta by the integer parts of x and y from their
                # floating-point-counterparts, respectively. But what happens if X falls
                # exactly between two pixels in either x or y direction? Then this calculation
                # for alpha or beta will result in a negative value! Take this into account
                # by taking the maximum with zero. Alpha and beta are both never smaller
                # than zero.
                #

                # Compute alpha and beta
                alpha   = max(0, x_rotated - math.floor(x_rotated))
                beta    = max(0, y_rotated - math.floor(y_rotated))

                # Now find the values that are stored in each of the corner pixels
                # It is important to realise that one of these corners may fall
                # outside of the image! In that case, fall back to the pixel that
                # lies closest to it. (Search for 'image border-padding' for more details).
                top_left_x, top_left_y = self.clamp(
                    math.floor(x_rotated),
                    math.ceil(y_rotated)
                )

                top_right_x, top_right_y = self.clamp(
                    math.ceil(x_rotated),
                    math.ceil(y_rotated)
                )

                bottom_left_x, bottom_left_y = self.clamp(
                    math.floor(x_rotated),
                    math.floor(y_rotated)
                )

                bottom_right_x, bottom_right_y = self.clamp(
                    math.ceil(x_rotated),
                    math.floor(y_rotated)
                )

                # Notice how it is possible to select the same pixel for two corners.
                # This happens when (x_rotated, y_rotated) falls exactly between two (not four)
                # or exactly on one pixel of the original image. By selecting the multiple times
                # above here, we have already accounted for this fact!
                #
                # Example: say we select the same pixel for top_left and top_right. Then, during
                # interpolation, we will add (alpha*top_right + (1-alpha)*top_left), which is
                # identical to 1*top_left which is identical to 1*top_right.
                #
                # Therefore, in all below calculations, we can assume that we have selected four
                # distinct pixels
                
                # Now, obtain the pixel values for these coordinates
                c1 = image_copy.get_pixel_at(top_right_x, top_right_y)
                c2 = image_copy.get_pixel_at(top_left_x, top_left_y)
                c3 = image_copy.get_pixel_at(bottom_left_x, bottom_left_y)
                c4 = image_copy.get_pixel_at(bottom_right_x, bottom_right_y)

                # And, finally, calculate the interpolated pixel value for this location in the
                # resulting image.
                result = alpha * beta * c1 + \
                                    (1-alpha) * beta * c2 + \
                                    (1-alpha) * (1-beta) * c3 + \
                                    alpha * (1-beta) * c4
                
                # Insert the value we found in the resulting image
                self.set_pixel_at(x, y, result)
