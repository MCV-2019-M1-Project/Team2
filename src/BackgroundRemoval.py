import cv2
import numpy as np


"""
Annotations
-----------
- Function BackgroundMask2 is way slower than BackgroundMask1, but it works better. To
  simply try out BackgroundMask2, downsample the image and upsample the mask later.
"""


def MSE(a,b,axis):
    """
    This function computes the MSE between a and b along the specified axis.

    Parameters
    ----------
    a : Numpy array.

    b : Numpy array.

    Returns
    -------
    Numpy array containing the MSE computation between a and b along the specified axis.
    """
    return ((a-b)**2).mean(axis=axis)

def GetForegroundPixels(img,mask):
    """
    This function returns the foreground pixels of an image according to a mask.

    Parameters
    ----------
    img : Image as numpy array.

    mask : Mask for img as numpy array of depth 1.

    Returns
    -------
    Flat numpy array containing the masked pixel values.
    """
    img_flat = np.reshape(img,(img.shape[0]*img.shape[1],img.shape[2]))
    mask_flat = np.reshape(mask,(mask.shape[0]*mask.shape[1]))
    return np.asarray([item for ind,item in enumerate(img_flat) if mask_flat[ind] != 0], dtype=np.uint8)

def BackgroundMask1(img):
    """
    This function returns a binary mask for the background/foreground of an image.

    Parameters
    ----------
    img : Image as numpy array.

    Returns
    -------
    mask : Binary mask for the background/foreground (0/1) of the input image.
    """

    # Convert img colorspace from RGB to YUV
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # Average estimate bg color, from the 4 corners, only using U and V channels
    img_uv = img_yuv[:,:,1:]
    corners = [img_uv[0,0,:],img_uv[0,-1,:],img_uv[-1,0,:],img_uv[-1,-1,:]]
    average_estimate_bg_color = np.mean(corners,axis=0)

    # Create bg image, compute MSE between img_uv and bg_img at each pixel, threshold at min_distance so that
    # mask at > min_distance = 1
    bg_img = np.ones(shape=img_uv.shape,dtype=np.uint8)*average_estimate_bg_color
    mask = MSE(img_uv,bg_img,axis=2)
    min_distance = 5
    _,mask = cv2.threshold(mask, min_distance, 255, cv2.THRESH_BINARY)

    return np.asarray(mask,dtype=np.uint8)

def BackgroundMask2(img):
    """
    This function returns a binary mask for the background/foreground of an image.

    Parameters
    ----------
    img : Image as numpy array.

    Returns
    -------
    mask : Binary mask for the background/foreground (0/1) of the input image.
    """

    # Convert img colorspace from RGB to YUV
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # Create bg_image, taking the average of the closest %percent% square from the borders of the image for each pixel,
    # using only U and V channels
    img_uv = img_yuv[:,:,1:]
    bg_img = img_uv.copy()
    percent = 0.02
    # ...BRACE YOURSELVES, BAD CODING IS COMING...
    for row_ind in range(int(percent*bg_img.shape[0]),int((1-percent)*bg_img.shape[0])):
        for col_ind in range(int(percent*bg_img.shape[1]),int((1-percent)*bg_img.shape[1])):
            if row_ind/img_uv.shape[0] < 0.5 and col_ind/img_uv.shape[1] < 0.5:
                if row_ind/img_uv.shape[0] < col_ind/img_uv.shape[1]:
                    from_ = col_ind-int(percent*img_uv.shape[1])
                    to_ = col_ind+int(percent*img_uv.shape[1])
                    if from_ < to_ and from_ >= 0 and to_ < img_uv.shape[1]:
                        region = img_uv[:int(percent*bg_img.shape[0]),from_:to_,:]
                        region = np.reshape(region,(region.shape[0]*region.shape[1],region.shape[2]))
                        average_estimate_bg_color = np.mean(region,axis=0)
                    else:
                        average_estimate_bg_color = img_uv[0,col_ind,:]
                else:
                    from_ = col_ind-int(percent*img_uv.shape[0])
                    to_ = col_ind+int(percent*img_uv.shape[0])
                    if from_ < to_ and from_ >= 0 and to_ < img_uv.shape[0]:
                        region = img_uv[from_:to_,:int(percent*bg_img.shape[1]),:]
                        region = np.reshape(region,(region.shape[0]*region.shape[1],region.shape[2]))
                        average_estimate_bg_color = np.mean(region,axis=0)
                    else:
                        average_estimate_bg_color = img_uv[row_ind,0,:]
            elif row_ind/img_uv.shape[0] > 0.5 and col_ind/img_uv.shape[1] < 0.5:
                if 1-row_ind/img_uv.shape[0] < col_ind/img_uv.shape[1]:
                    from_ = col_ind-int(percent*img_uv.shape[1])
                    to_ = col_ind+int(percent*img_uv.shape[1])
                    if from_ < to_ and from_ >= 0 and to_ < img_uv.shape[1]:
                        region = img_uv[int((1-percent)*bg_img.shape[0]):,from_:to_,:]
                        region = np.reshape(region,(region.shape[0]*region.shape[1],region.shape[2]))
                        average_estimate_bg_color = np.mean(region,axis=0)
                    else:
                        average_estimate_bg_color = img_uv[-1,col_ind,:]
                else:
                    from_ = col_ind-int(percent*img_uv.shape[0])
                    to_ = col_ind+int(percent*img_uv.shape[0])
                    if from_ < to_ and from_ >= 0 and to_ < img_uv.shape[0]:
                        region = img_uv[from_:to_,:int(percent*bg_img.shape[1]),:]
                        region = np.reshape(region,(region.shape[0]*region.shape[1],region.shape[2]))
                        average_estimate_bg_color = np.mean(region,axis=0)
                    else:
                        average_estimate_bg_color = img_uv[row_ind,0,:]
            elif row_ind/img_uv.shape[0] < 0.5 and col_ind/img_uv.shape[1] > 0.5:
                if row_ind/img_uv.shape[0] < 1-col_ind/img_uv.shape[1]:
                    from_ = col_ind-int(percent*img_uv.shape[1])
                    to_ = col_ind+int(percent*img_uv.shape[1])
                    if from_ < to_ and from_ >= 0 and to_ < img_uv.shape[1]:
                        region = img_uv[:int(percent*bg_img.shape[0]),from_:to_,:]
                        region = np.reshape(region,(region.shape[0]*region.shape[1],region.shape[2]))
                        average_estimate_bg_color = np.mean(region,axis=0)
                    else:
                        average_estimate_bg_color = img_uv[0,col_ind,:]
                else:
                    from_ = col_ind-int(percent*img_uv.shape[0])
                    to_ = col_ind+int(percent*img_uv.shape[0])
                    if from_ < to_ and from_ >= 0 and to_ < img_uv.shape[0]:
                        region = img_uv[from_:to_,int((1-percent)*bg_img.shape[1]):,:]
                        region = np.reshape(region,(region.shape[0]*region.shape[1],region.shape[2]))
                        average_estimate_bg_color = np.mean(region,axis=0)
                    else:
                        average_estimate_bg_color = img_uv[row_ind,-1,:]
            else:
                if row_ind/img_uv.shape[0] < 1-col_ind/img_uv.shape[1]:
                    from_ = col_ind-int(percent*img_uv.shape[1])
                    to_ = col_ind+int(percent*img_uv.shape[1])
                    if from_ < to_ and from_ >= 0 and to_ < img_uv.shape[1]:
                        region = img_uv[int((1-percent)*bg_img.shape[0]):,from_:to_,:]
                        region = np.reshape(region,(region.shape[0]*region.shape[1],region.shape[2]))
                        average_estimate_bg_color = np.mean(region,axis=0)
                    else:
                        average_estimate_bg_color = img_uv[-1,col_ind,:]
                else:
                    from_ = col_ind-int(percent*img_uv.shape[0])
                    to_ = col_ind+int(percent*img_uv.shape[0])
                    if from_ < to_ and from_ >= 0 and to_ < img_uv.shape[0]:
                        region = img_uv[from_:to_,int((1-percent)*bg_img.shape[1]):,:]
                        region = np.reshape(region,(region.shape[0]*region.shape[1],region.shape[2]))
                        average_estimate_bg_color = np.mean(region,axis=0)
                    else:
                        average_estimate_bg_color = img_uv[row_ind,-1,:]
            bg_img[row_ind,col_ind] = average_estimate_bg_color

    # Compute MSE between img_uv and bg_img at each pixel, threshold at min_distance so that
    # mask at > min_distance = 1
    mask = MSE(img_uv,bg_img,axis=2)
    min_distance = 5
    _,mask = cv2.threshold(mask, min_distance, 255, cv2.THRESH_BINARY)

    return np.asarray(mask,dtype=np.uint8)
