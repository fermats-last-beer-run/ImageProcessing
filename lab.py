#!/usr/bin/env python3

"""
Image Processing Project
- Takes a PNG image and performs various linear transformations on it
Author: Dylan Espey
"""

import math

from PIL import Image

def get_1d_location(image, row, col):       
    return (image['width'] * row) + col

def get_pixel(image, row, col):
    loc = get_1d_location(image, row, col)
    return image["pixels"][loc]

def get_pixel_wrap(image, row, col):
    row = row % image['height']
    col = col % image['width']
    return get_pixel(image, row, col)

def get_pixel_zero(image, row, col):
    if(row < 0 or col < 0 or row >= image['height'] or col >= image['width']):
        return 0
    return get_pixel(image, row, col)

def get_pixel_extend(image, row, col):
    if row < 0:
        row = 0
    if col < 0:
        col = 0
    if row >= image['height']:
        row = image['height'] - 1
    if col >= image['width']:
        col = image['width'] - 1
    return get_pixel(image, row, col)

def get_pixel_mode(image, row, col, mode="zero"):
    """
    Get pixel using edge modes "zero", "wrap", "extend"
    row relates to height, col relates to width
    i.e; the number of rows and the number of cols
    """
    modes = {'zero':get_pixel_zero, 'wrap':get_pixel_wrap, 'extend':get_pixel_extend}
    return modes[mode](image,row,col)



def set_pixel(image, row, col, color):
    loc = get_1d_location(image, row, col)
    image["pixels"][loc] = color


def apply_per_pixel(image, func):
    result = {
        "height": image["height"],
        "width": image["width"],
        "pixels": image['pixels'].copy(),
    }
    for row in range(image["height"]):
        for col in range(image["width"]):
            color = get_pixel(image, row, col)
            new_color = func(color)
            set_pixel(result, row, col, new_color)
    return result


def inverted(image):
    return apply_per_pixel(image, lambda color: 255-color)


# HELPER FUNCTIONS

def correlate(image, kernel, boundary_behavior):
    """
    Computes the result of correlating the given image with the given kernel.

    `boundary_behavior` is a string of "zero", "extend", or "wrap". Else, return None.

    kernel: Dictionary ('height', 'width' 'pixels')
    width is redundant, but allows us to use the other helper functions intended for the base image representation
    """
    
    def kernel_op(x, y, midpoint):
        ''' Multiplies image Pixel(x+ x offset, y+ y offset) by corresponding kernel (midpoint + y offset, midpoint + x offset)'''
        k_value = get_pixel_mode(kernel, midpoint + y, midpoint + x, boundary_behavior)
        image_value = get_pixel_mode(image, row + y, col + x, boundary_behavior)
        return image_value * k_value
    
    
    def apply_kernel(kernel):
        ''' Iterates across (n x n) kernel range, selects corresponding kernel values in image '''
        growth = kernel['height'] // 2
        final = 0
        for y_offset in range(growth * -1, growth + 1):
            for x_offset in range(growth * -1, growth + 1):
                final += kernel_op(x_offset, y_offset, growth)
        return final

        
    # Exit function if edge boundary behaviour is not a valid option
    if boundary_behavior not in ['zero', 'wrap', 'extend']:
        return None
    
    # Image Data frame
    result = {
        "height": image["height"],
        "width": image["width"],
        "pixels": image['pixels'].copy(),
    }

    # Iterate over each row and column
    for row in range(0, image["height"]):
        for col in range(0, image["width"]):
            # Perform a kernel operation
            kValue = apply_kernel(kernel)
            # Set Pixel to kernel value
            set_pixel(result, row, col, kValue)

    return result
    
    



def round_and_clip_image(image):
    """
    Given a dictionary, ensures that the values in the "pixels" list are all
    integers in the range [0, 255].

    Any locations with values higher than 255 in the input should have value
    255 in the output; and any locations with values lower than 0 in the input
    should have value 0 in the output.
    """
    for pixel in range(0, len(image['pixels'])):
        # Rounds each pixel, then guarantees it falls within 0, 255 range
        image['pixels'][pixel] = max( 0, min(255, round(image['pixels'][pixel]) ))
    return image


def generate_kernel(size, f):
    kernel = {'height':size, 'width':size, 'pixels': []}
    for _ in range(0, size*size):
        kernel['pixels'].append(f)
    return kernel

# FILTERS

def blurred(image, kernel_size):
    """
    Returns a new image representing the result of applying a box blur (with the
    given kernel size) to the given input image.

    """
    # Generates a gaussian blur kernel
    kernel = generate_kernel(kernel_size, 1/(kernel_size**2))

    # Compute the correlation of the image for a given kernel
    result = correlate(image, kernel, 'extend')
    
    # Return validated image
    return round_and_clip_image(result)

def sharpened(image, kernel_size):
    """
    Sharpening Formula: S(r,c) = 2*I(r,c) - Blur(r,c) = - Blur(r,c) + 2*I(r,c)
    Returns a new image representing the result of applying an unsharp mask (with the
    given kernel size) to the given input image.

    """
    # Generates a gaussian kernel with negative values, and add 2 * Identity kernel to it (add 2 to the midpoint)
    kernel = generate_kernel(kernel_size, -1/(kernel_size**2))
    midpoint = get_1d_location(kernel, kernel['height'] // 2, kernel['height'] // 2)
    kernel['pixels'][midpoint] += 2


    # Compute the correlation of the image for a given kernel
    result = correlate(image, kernel, 'extend')
    # Return validated image
    return round_and_clip_image(result)

def edges(image):
    ''' 
        Edge Detection using Sobel operator
        Returns an edge mask where 255 white represents an edge. 
    '''
    sobel_kernel_first = {'height': 3, 'width': 3, 'pixels':
                          [
                              -1,-2,-1,
                              0,0,0,
                              1,2,1
                          ]}
    
    sobel_kernel_second = {'height': 3, 'width': 3, 'pixels':
                        [
                            -1,0,1,
                            -2,0,2,
                            -1,0,1
                        ]}
    
    result = {'height':image['height'], 'width':image['width'], 'pixels': [] }
    
    result_first = correlate(image, sobel_kernel_first, 'extend')
    result_second = correlate(image, sobel_kernel_second, 'extend')

    for i, j in zip(result_first['pixels'], result_second['pixels']):
        result['pixels'].append(math.sqrt((i**2) + (j**2)))
        
    result = round_and_clip_image(result)
    return result

# HELPER FUNCTIONS FOR LOADING AND SAVING IMAGES

def load_greyscale_image(filename):
    """
    Loads an image from the given file and returns a dictionary
    representing that image.  This also performs conversion to greyscale.

    Invoked as, for example:
       i = load_greyscale_image("test_images/cat.png")
    """
    with open(filename, "rb") as img_handle:
        img = Image.open(img_handle)
        img_data = img.getdata()
        if img.mode.startswith("RGB"):
            pixels = [round(.299 * p[0] + .587 * p[1] + .114 * p[2])
                      for p in img_data]
        elif img.mode == "LA":
            pixels = [p[0] for p in img_data]
        elif img.mode == "L":
            pixels = list(img_data)
        else:
            raise ValueError(f"Unsupported image mode: {img.mode}")
        width, height = img.size
        return {"height": height, "width": width, "pixels": pixels}


def save_greyscale_image(image, filename, mode="PNG"):
    """
    Saves the given image to disk or to a file-like object.  If filename is
    given as a string, the file type will be inferred from the given name.  If
    filename is given as a file-like object, the file type will be determined
    by the "mode" parameter.
    """
    out = Image.new(mode="L", size=(image["width"], image["height"]))
    out.putdata(image["pixels"])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


if __name__ == "__main__":
    image_in = load_greyscale_image("test_images/mario.png")
    image_out = edges(image_in)
    save_greyscale_image(image_out, "edges.png")

