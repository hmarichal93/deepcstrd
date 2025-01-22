from cross_section_tree_ring_detection.preprocessing import resize, NONE



def preprocessing(im_in, height_output=None, width_output=None, cy=None, cx=None):
    """
    Image preprocessing steps. Following actions are made
    - image resize
    - image is converted to gray scale
    - gray scale image is equalized
    Implements Algorithm 7 in the paper.
    @param im_in: segmented image
    @param height_output: new image img_height
    @param width_output: new image img_width
    @param cy: pith y's coordinate
    @param cx: pith x's coordinate
    @return:
    - im_pre: equalized image
    - cy: pith y's coordinate after resize
    - cx: pith x's coordinate after resize
    """
    # Line 1 to 6
    if NONE in [height_output, width_output]:
        im_r, cy_output, cx_output = ( im_in, cy, cx)
    else:
        im_r, cy_output, cx_output = resize( im_in, height_output, width_output, cy, cx)

    return im_r, cy_output, cx_output