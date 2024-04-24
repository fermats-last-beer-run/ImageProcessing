#!/usr/bin/env python3

import os
import pickle
import hashlib

import lab
import pytest

TEST_DIRECTORY = os.path.dirname(__file__)


def object_hash(x):
    return hashlib.sha512(pickle.dumps(x)).hexdigest()


def compare_images(im1, im2):
    assert set(im1.keys()) == {'height', 'width', 'pixels'}, 'Incorrect keys in dictionary'
    assert im1['height'] == im2['height'], 'Heights must match'
    assert im1['width'] == im2['width'], 'Widths must match'
    assert len(im1['pixels']) == im1['height']*im1['width'], 'Incorrect number of pixels'
    assert all(isinstance(i, int) for i in im1['pixels']), 'Pixels must all be integers'
    assert all(0<=i<=255 for i in im1['pixels']), 'Pixels must all be in the range from [0, 255]'
    pix_incorrect = (None, None)
    for ix, (i, j) in enumerate(zip(im1['pixels'], im2['pixels'])):
        if i != j:
            pix_incorrect = (ix, abs(i-j))
    assert pix_incorrect == (None, None), 'Pixels must match.  Incorrect value at location %s (differs from expected by %s)' % pix_incorrect



def test_load():
    result = lab.load_greyscale_image(os.path.join(TEST_DIRECTORY, 'test_images', 'centered_pixel.png'))
    expected = {
        'height': 11,
        'width': 11,
        'pixels': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    }
    compare_images(result, expected)

def test_round_and_clip():
    result = lab.load_greyscale_image(os.path.join(TEST_DIRECTORY, 'test_images', 'centered_pixel.png'))
    if not result:
        print("No image")

    result['pixels'][0] = 1000
    result['pixels'][1] = -1999

    lab.round_and_clip_image(result)
    
    assert result['pixels'][0] == 255
    assert result['pixels'][1] == 0

   

def test_edge_mode_zero():
    result = lab.load_greyscale_image(os.path.join(TEST_DIRECTORY, 'test_images', 'cat.png'))
    if not result:
        print("No image")

    assert lab.get_pixel_mode(result, result['height'], 0) == 0
    assert lab.get_pixel_mode(result, 0, result['width']) == 0
    assert lab.get_pixel_mode(result, result['height'], result['width']) == 0
    assert lab.get_pixel_mode(result, -1, 0) == 0
    assert lab.get_pixel_mode(result, 0, -1) == 0

    assert lab.get_pixel_mode(result, 0, 0) != 0

def test_edge_mode_wrap():
    result = lab.load_greyscale_image(os.path.join(TEST_DIRECTORY, 'test_images', 'cat.png'))
    if not result:
        print("No image")

    assert lab.get_pixel_mode(result, result['height'], 0, mode='wrap') == result['pixels'][0]
    assert lab.get_pixel_mode(result, 0, result['width'], mode='wrap') == result['pixels'][0]
    assert lab.get_pixel_mode(result, result['height'], result['width'], mode='wrap') == result['pixels'][0]

    assert lab.get_pixel_mode(result, -1, -1, mode='wrap') != result['pixels'][0]

def test_edge_mode_extend():
    result = lab.load_greyscale_image(os.path.join(TEST_DIRECTORY, 'test_images', 'cat.png'))
    if not result:
        print("No image")
    
    expected_pixel = lab.get_1d_location(result, result['height'] - 1, 0)
    # EXTENDING BEYOND THE HEIGHT RETURNS THE BOUND OF THE HEIGHT
    assert lab.get_pixel_mode(result, result['height'], 0, mode='extend') == result['pixels'][expected_pixel]
    # EXTENDING BEYOND THE HEIGHT FURTHER STILL RETURNS THE BOUND OF THE HEIGHT
    assert lab.get_pixel_mode(result, result['height'] + 1, 0, mode='extend') == result['pixels'][expected_pixel]
    # EXTENDING BEYOND THE HEIGHT AND WIDTH RETURNS THE FINAL PIXEL
    assert lab.get_pixel_mode(result, result['height'], result['width'], mode='extend') == result['pixels'][-1]
    # EXTENDING BEYOND THE HEIGHT AND WIDTH EVEN MORE RETURNS THE FINAL PIXEL
    assert lab.get_pixel_mode(result, result['height'] + 1, result['width'] + 1, mode='extend') == result['pixels'][-1]

    assert lab.get_pixel_mode(result, result['height'], result['width'], mode='extend') != result['pixels'][0]
    
def test_inverted_1():
    im = lab.load_greyscale_image(os.path.join(TEST_DIRECTORY, 'test_images', 'centered_pixel.png'))
    result = lab.inverted(im)
    expected = {
        'height': 11,
        'width': 11,
        'pixels': [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                   255, 255, 255, 255, 255, 0, 255, 255, 255, 255, 255,
                   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
    }
    compare_images(result, expected)

def test_inverted_2():
    im = lab.load_greyscale_image(os.path.join(TEST_DIRECTORY, 'test_images', 'centered_pixel.png'))
    result = lab.inverted(lab.inverted(im))
    expected = {
        'height': 11,
        'width': 11,
        'pixels':  [0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,255,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0],
    }
    compare_images(result, expected)

def test_correlate_identity():
    im = lab.load_greyscale_image(os.path.join(TEST_DIRECTORY, 'test_images', 'centered_pixel.png'))

    kernel = {
        'height': 3,
        'width':3,
        'pixels': [0,0,0,
                   0,1,0,
                   0,0,0],
    }

    result = lab.correlate(im, kernel, 'zero')

    expected = {
        'height': 11,
        'width': 11,
        'pixels':  [0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,255,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0],
    }

    compare_images(result, expected)

def test_correlate_translate():
    im = lab.load_greyscale_image(os.path.join(TEST_DIRECTORY, 'test_images', 'centered_pixel.png'))

    kernel = {
        'height': 5,
        'width':5,
        'pixels': [0,0,0,0,0,
                   0,0,0,0,0,
                   1,0,0,0,0,
                   0,0,0,0,0,
                   0,0,0,0,0],
    }

    result = lab.correlate(im, kernel, 'extend')

    expected = {
        'height': 11,
        'width': 11,
        'pixels':  [0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,255,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0],
    }
    compare_images(result, expected)

def test_correlate_avg():
    im = lab.load_greyscale_image(os.path.join(TEST_DIRECTORY, 'test_images', 'centered_pixel.png'))

    kernel = {
        'height': 3,
        'width':3,
        'pixels': [0.0,0.2,0.0,
                   0.2,0.2,0.2,
                   0.0,0.2,0.0]
    }

    result = lab.correlate(im, kernel, 'extend')

    expected = {
        'height': 11,
        'width': 11,
        'pixels':  [0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,51,0,0,0,0,0,
                    0,0,0,0,51,51,51,0,0,0,0,
                    0,0,0,0,0,51,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0],
    }

    lab.round_and_clip_image(result)
    compare_images(result, expected)


@pytest.mark.parametrize("fname", ['mushroom', 'twocats', 'chess'])
def test_inverted_images(fname):
    inpfile = os.path.join(TEST_DIRECTORY, 'test_images', '%s.png' % fname)
    expfile = os.path.join(TEST_DIRECTORY, 'test_results', '%s_invert.png' % fname)
    im = lab.load_greyscale_image(inpfile)
    oim = object_hash(im)
    result = lab.inverted(im)
    expected = lab.load_greyscale_image(expfile)
    assert object_hash(im) == oim, 'Be careful not to modify the original image!'
    compare_images(result, expected)


@pytest.mark.parametrize("kernsize", [1, 3, 7])
@pytest.mark.parametrize("fname", ['mushroom', 'twocats', 'chess'])
def test_blurred_images(kernsize, fname):
    inpfile = os.path.join(TEST_DIRECTORY, 'test_images', '%s.png' % fname)
    expfile = os.path.join(TEST_DIRECTORY, 'test_results', '%s_blur_%02d.png' % (fname, kernsize))
    input_img = lab.load_greyscale_image(inpfile)
    input_hash = object_hash(input_img)
    result = lab.blurred(input_img, kernsize)
    expected = lab.load_greyscale_image(expfile)
    assert object_hash(input_img) == input_hash, "Be careful not to modify the original image!"
    compare_images(result, expected)

def test_blurred_black_image():
    im = {
        'height':5,
        'width':6,
        'pixels':[255,255,255,255,255,255,
                  255,255,255,255,255,255,
                  255,255,255,255,255,255,
                  255,255,255,255,255,255,
                  255,255,255,255,255,255],
    }

    result = lab.blurred(im, 3)

    expected = {
        'height':5,
        'width':6,
        'pixels':[255,255,255,255,255,255,
                  255,255,255,255,255,255,
                  255,255,255,255,255,255,
                  255,255,255,255,255,255,
                  255,255,255,255,255,255],
    }

    result_2 = lab.blurred(im, 9)

    lab.round_and_clip_image(result)
    lab.round_and_clip_image(result_2)

    compare_images(result, expected)
    compare_images(result_2, expected)


def test_blurred_centered_pixel():
    im =  lab.load_greyscale_image(os.path.join(TEST_DIRECTORY, 'test_images', 'centered_pixel.png'))
    result = lab.blurred(im, 3)
    result_2 = lab.blurred(im, 11)
    print(result_2)
    expected = {
        'height': 11,
        'width': 11,
        'pixels':  [0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,28,28,28,0,0,0,0,
                    0,0,0,0,28,28,28,0,0,0,0,
                    0,0,0,0,28,28,28,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0],
    }

    expected_9x9 = {
        'height': 11,
        'width': 11,
        'pixels':  [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    }


    lab.round_and_clip_image(result)
    lab.round_and_clip_image(result_2)
    compare_images(result, expected)
    compare_images(result_2, expected_9x9)

@pytest.mark.parametrize("kernsize", [1, 3, 9])
@pytest.mark.parametrize("fname", ['mushroom', 'twocats', 'chess'])
def test_sharpened_images(kernsize, fname):
    inpfile = os.path.join(TEST_DIRECTORY, 'test_images', '%s.png' % fname)
    expfile = os.path.join(TEST_DIRECTORY, 'test_results', '%s_sharp_%02d.png' % (fname, kernsize))
    input_img = lab.load_greyscale_image(inpfile)
    input_hash = object_hash(input_img)
    result = lab.sharpened(input_img, kernsize)
    expected = lab.load_greyscale_image(expfile)
    assert object_hash(input_img) == input_hash, "Be careful not to modify the original image!"
    compare_images(result, expected)


@pytest.mark.parametrize("fname", ['mushroom', 'twocats', 'chess'])
def test_edges_images(fname):
    inpfile = os.path.join(TEST_DIRECTORY, 'test_images', '%s.png' % fname)
    expfile = os.path.join(TEST_DIRECTORY, 'test_results', '%s_edges.png' % fname)
    input_img = lab.load_greyscale_image(inpfile)
    input_hash = object_hash(input_img)
    result = lab.edges(input_img)
    expected = lab.load_greyscale_image(expfile)
    assert object_hash(input_img) == input_hash, "Be careful not to modify the original image!"
    compare_images(result, expected)

def test_edges_centered_pixel():
    im =  lab.load_greyscale_image(os.path.join(TEST_DIRECTORY, 'test_images', 'centered_pixel.png'))
    print(im)
    result = lab.edges(im)

    expected = {
        'height': 11,
        'width': 11,
        'pixels': [0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,255,255,255,0,0,0,0,
                    0,0,0,0,255,0,255,0,0,0,0,
                    0,0,0,0,255,255,255,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0]
    }
    print(result)
    compare_images(result, expected)
