import os
from glob import glob
from distance_metrics import mse
import cv2
import numpy as np
import time


"""
Annotations
-----------
- Best is BackgroundMask4, pretty fast
- Do not use BackgroundMask5 or BackgroundMask6
- Function BackgroundMask2 is way slower than BackgroundMask1, but it works better.
"""

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
    return np.asarray([item for ind,item in enumerate(img_flat) if mask_flat[ind] != 0],dtype=np.uint8)

def BackgroundMask1(img):
    """
    This function returns a binary mask for the background/foreground of an image.

    Parameters
    ----------
    img : Image read with opencv.

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
    mask = mse(img_uv,bg_img,axis=2)
    min_distance = 5
    _,mask = cv2.threshold(mask, min_distance, 255, cv2.THRESH_BINARY)

    return np.asarray(mask,dtype=np.uint8)

def BackgroundMask2(img):
    """
    This function returns a binary mask for the background/foreground of an image.

    Parameters
    ----------
    img : Image read with opencv.

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
    percent = 0.02 # 0.02
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

    # Compute mse between img_uv and bg_img at each pixel, threshold at min_distance so that
    # mask at > min_distance = 1
    mask = mse(img_uv,bg_img,axis=2)
    min_distance = 5 # 5
    _,mask = cv2.threshold(mask, min_distance, 255, cv2.THRESH_BINARY)

    return np.asarray(mask,dtype=np.uint8)

def BackgroundMask3(img):
    """
    This function returns a binary mask for the background/foreground of an image.

    Parameters
    ----------
    img : Image read with opencv.

    Returns
    -------
    mask : Binary mask for the background/foreground (0/1) of the input image.
    """

    # Convert img colorspace from RGB to YUV
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # Find contours of the rectangle, using only UV channels
    img_uv = img_yuv[:,:,1:2]
    min_change = 10
    percent = 0.02
    x1,x2,y1,y2 = [int(img_uv.shape[0]*percent),int(img_uv.shape[0]*(1-percent)),int(img_uv.shape[1]*percent),int(img_uv.shape[1]*(1-percent))]
    # -- up
    for row_ind in range(int(img_uv.shape[0]*percent),img_uv.shape[0]):
        current = np.mean(img_uv[row_ind,int(img_uv.shape[1]*(0.5-percent/2)):int(img_uv.shape[1]*(0.5+percent/2))],axis=0)
        previous = np.mean(img_uv[row_ind-1,int(img_uv.shape[1]*0.49):int(img_uv.shape[1]*0.51)],axis=0)
        if mse(current,previous,axis=0) > min_change:
            x1 = row_ind
            break
    # -- down
    for row_ind in range(int(img_uv.shape[0]*(1-percent)),0,-1):
        current = np.mean(img_uv[row_ind,int(img_uv.shape[1]*(0.5-percent/2)):int(img_uv.shape[1]*(0.5+percent/2))],axis=0)
        previous = np.mean(img_uv[row_ind+1,int(img_uv.shape[1]*(0.5-percent/2)):int(img_uv.shape[1]*(0.5+percent/2))],axis=0)
        if mse(current,previous,axis=0) > min_change:
            x2 = row_ind
            break
    # -- left
    for col_ind in range(1,img_uv.shape[1]):
        current = np.mean(img_uv[int(img_uv.shape[0]*(0.5-percent/2)):int(img_uv.shape[0]*(0.5+percent/2)),col_ind],axis=0)
        previous = np.mean(img_uv[int(img_uv.shape[0]*(0.5-percent/2)):int(img_uv.shape[0]*(0.5+percent/2)),col_ind-1],axis=0)
        if mse(current,previous,axis=0) > min_change:
            y1 = col_ind
            break
    # -- right
    for col_ind in range(img_uv.shape[1]-2,0,-1):
        current = np.mean(img_uv[int(img_uv.shape[0]*(0.5-percent/2)):int(img_uv.shape[0]*(0.5+percent/2)),col_ind],axis=0)
        previous = np.mean(img_uv[int(img_uv.shape[0]*(0.5-percent/2)):int(img_uv.shape[0]*(0.5+percent/2)),col_ind+1],axis=0)
        if mse(current,previous,axis=0) > min_change:
            y2 = col_ind
            break

    mask = np.zeros(shape=(img_uv.shape[0],img_uv.shape[1]),dtype=np.uint8)
    mask[x1:x2,y1:y2] = 255
    return mask

#Hough lines
def BackgroundMask4(img):
    # Resize image
    original_size = img.shape[:2]
    resize_shape = (1000,1000)
    img = cv2.resize(img,resize_shape)

    # Obtain gray level image
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Obtain edges through Canny algorithm
    edges = cv2.Canny(gray,50,150,apertureSize=3)

    # Dilate edges
    kernel = np.ones((3,3),np.uint8)
    edges = cv2.dilate(edges,kernel,iterations=1)

    # Obtain straight lines through Hough transform
    large_number = 100000
    percent_border = 0.05
    pict_start_perc = 0.3
    lines_wanted = {"top": [-large_number,large_number,int(pict_start_perc*img.shape[0]),int(pict_start_perc*img.shape[0])], "bottom": [-large_number,large_number,int((1-pict_start_perc)*img.shape[0]),int((1-pict_start_perc)*img.shape[0])], "left": [int(pict_start_perc*img.shape[1]),int(pict_start_perc*img.shape[1]),-large_number,large_number], "right": [int((1-pict_start_perc)*img.shape[1]),int((1-pict_start_perc)*img.shape[1]),-large_number,large_number]}
    last_mask = np.zeros(shape=(original_size[0],original_size[1]),dtype=np.uint8)
    last_mean_points = {"top":None,"left":None,"bottom":None,"right":None}
    for degrees_margin in [2,5]:
        lines = cv2.HoughLines(edges,1,np.pi/180,150)
        for i in range(len(lines)):
            rho,theta = lines[i][0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + large_number*(-b))
            y1 = int(y0 + large_number*(a))
            x2 = int(x0 - large_number*(-b))
            y2 = int(y0 - large_number*(a))

            try:
                slope=(y2-y1)/(x2-x1)
            except:
                slope = 90
            # We want only vertical/horizontal lines with a certain degree deviation
            degrees = np.abs(np.arctan([slope])*180/np.pi)
            if degrees < degrees_margin or (90-degrees_margin) < degrees:
                # Horizontal line
                if degrees < degrees_margin:
                    # Line is in the %percent_border%
                    if np.mean([y1,y2]) < percent_border*img.shape[0] or np.mean([y1,y2]) > (1-percent_border)*img.shape[0]:
                        continue
                    # Horizontal line more to the top than registered one
                    if np.mean([y1,y2]) < np.mean(lines_wanted["top"][2:4]):
                        lines_wanted["top"] = [x1,x2,y1,y2,degrees]
                    # elif degrees < lines_wanted["top"][4] and np.mean([x1,x2]) < np.mean(lines_wanted["top"][2:4])+0.1*img.shape[0]:
                    #     lines_wanted["top"] = [x1,x2,y1,y2,degrees]
                    # Horizontal line more to the bottom than the registered one
                    elif np.mean([y1,y2]) > np.mean(lines_wanted["bottom"][2:4]):
                        lines_wanted["bottom"] = [x1,x2,y1,y2,degrees]
                    # elif degrees < lines_wanted["bottom"][4] and np.mean([x1,x2]) < np.mean(lines_wanted["bottom"][2:4])-0.1*img.shape[0]:
                    #     lines_wanted["bottom"] = [x1,x2,y1,y2,degrees]
                # Vertical line
                elif (90-degrees_margin) < degrees:
                    # Line is in the %percent_border%
                    if np.mean([x1,x2]) < percent_border*img.shape[1] or np.mean([x1,x2]) > (1-percent_border)*img.shape[1]:
                        continue
                    # Vertical line more to the left than the registered one
                    if np.mean([x1,x2]) < np.mean(lines_wanted["left"][:2]):
                        lines_wanted["left"] = [x1,x2,y1,y2,degrees]
                    # elif degrees > lines_wanted["left"][4] and np.mean([x1,x2]) < np.mean(lines_wanted["left"][:2])+0.1*img.shape[1]:
                    #     lines_wanted["left"] = [x1,x2,y1,y2,degrees]
                    # Vertical line more to the right than the registered one
                    elif np.mean([x1,x2]) > np.mean(lines_wanted["right"][:2]):
                        lines_wanted["right"] = [x1,x2,y1,y2,degrees]
                    # elif degrees > lines_wanted["right"][4] and np.mean([x1,x2]) < np.mean(lines_wanted["right"][:2])-0.1*img.shape[1]:
                    #     lines_wanted["right"] = [x1,x2,y1,y2,degrees]

                # print("x1",x1,"x2",x2,"y1",y1,"y2",y2,"theta",theta,"rho",rho)
                # print('slope',)

                # cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

        # Draw lines_wanted to an all_zeros image
        mask = np.zeros(shape=img.shape[:2],dtype=np.uint8)
        draw_perc_th = 0.15
        for k,v in lines_wanted.items():
            if k == "top":
                if np.mean(v[2:4]) >= int(pict_start_perc*img.shape[0]):
                    v[2] = int(draw_perc_th*img.shape[0])
                    v[3] = int(draw_perc_th*img.shape[0])
            elif k == "bottom":
                if np.mean(v[2:4]) <= int((1-pict_start_perc)*img.shape[0]):
                    v[2] = int((1-draw_perc_th)*img.shape[0])
                    v[3] = int((1-draw_perc_th)*img.shape[0])
            elif k == "left":
                if np.mean(v[:2]) >= int(pict_start_perc*img.shape[1]):
                    v[0] = int(draw_perc_th*img.shape[1])
                    v[1] = int(draw_perc_th*img.shape[1])
            elif k == "right":
                if np.mean(v[:2]) <= int((1-pict_start_perc)*img.shape[1]):
                    v[0] = int((1-draw_perc_th)*img.shape[1])
                    v[1] = int((1-draw_perc_th)*img.shape[1])
            # print(k,np.mean([v[0],v[1]]),np.mean([v[2],v[3]]))
            cv2.line(mask,(v[0],v[2]),(v[1],v[3]),(255,255,255),2)

        # Floodfill from center of lines drawn
        center_x = int((np.mean(lines_wanted["bottom"][2:4]+np.mean(lines_wanted["top"][2:4])))/2)
        center_y = int((np.mean(lines_wanted["right"][:2])+np.mean(lines_wanted["left"][:2]))/2)
        mask_flood = np.zeros(shape=(mask.shape[0]+2,mask.shape[1]+2),dtype=np.uint8)
        cv2.floodFill(mask, mask_flood, (center_y,center_x), (255,255,255));

        # # Floodfill from center of image
        # mask_flood = np.zeros(shape=(mask.shape[0]+2,mask.shape[1]+2),dtype=np.uint8)
        # cv2.floodFill(mask, mask_flood, (int(mask.shape[0]/2),int(mask.shape[1]/2)), (255,255,255));

        # Depaint lines
        for k,v in lines_wanted.items():
            cv2.line(mask,(v[0],v[2]),(v[1],v[3]),(0,0,0),2)

        # Resize mask to original size
        mask = cv2.resize(mask,(original_size[1],original_size[0]))

        # If new mask contains old mask, we stay with new mask
        if (np.bitwise_or(mask,last_mask) == mask).all():
            last_mask = mask
            last_mean_points["top"] = int(np.mean(lines_wanted["top"][2:4])*original_size[0]/resize_shape[0])
            last_mean_points["left"] = int(np.mean(lines_wanted["left"][:2])*original_size[1]/resize_shape[1])
            last_mean_points["bottom"] = int(np.mean(lines_wanted["bottom"][2:4])*original_size[0]/resize_shape[0])
            last_mean_points["right"] = int(np.mean(lines_wanted["right"][:2])*original_size[1]/resize_shape[1])

    return last_mask, last_mean_points

#Hough lines p
def BackgroundMask5(img):
    # Resize image
    original_size = img.shape[:2]
    img = cv2.resize(img,(1000,1000))
    # Obtain gray level image
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Obtain edges through Canny algorithm
    edges = cv2.Canny(gray,50,150,apertureSize=3)

    # Dilate edges
    kernel = np.ones((3,3),np.uint8)
    edges = cv2.dilate(edges,kernel,iterations=1)

    # Hough lines p
    lines = cv2.HoughLinesP(edges, 1, np.pi/180.0, 100,100,50)
    for i in range(len(lines)):
        x1,x2,y1,y2 = lines[i][0]
        print(x1,x2,y1,y2)
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

    return img

#Draw contours
def BackgroundMask6(img):
    # Resize image
    # original_size = img.shape[:2]
    # img = cv2.resize(img,(1000,1000))
    # Obtain gray level image
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    kernel = np.ones((5,5),np.float32)/25
    img2gray = cv2.filter2D(gray,-1,kernel)

    # threshold the image and extract contours
    _,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    contours,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # find the main island (biggest area)
    cnt = contours[0]
    max_area = cv2.contourArea(cnt)
    for cont in contours:
        if cv2.contourArea(cont) > max_area:
            cnt = cont
            max_area = cv2.contourArea(cont)

    # define main island contour approx. and hull
    perimeter = cv2.arcLength(cnt,True)
    epsilon = 0.01*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)

    hull = cv2.convexHull(cnt)

    # cv2.isContourConvex(cnt)

    mask = np.zeros(shape=img.shape,dtype=np.uint8)

    # cv2.drawContours(mask, cnt, -1, (0, 255, 0), 3)
    cv2.drawContours(mask, [approx], -1, (0, 0, 255), 3)

    return mask

#Watershed
def BackgroundMask7(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)
    _,sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    _,markers = cv2.connectedComponents(sure_fg)
    # markers[markers != 0] = 255
    markers[markers != 0] = 200

    markers[0:40,:] = 100
    sure_fg[0:40,:] = 100
    markers[len(markers)-41:len(markers)-1,:] = 100
    sure_fg[len(markers)-41:len(markers)-1,:] = 100
    markers[:,0:40] = 100
    sure_fg[:,0:40] = 100
    markers[:,len(markers[0,:])-41:len(markers[0,:])-1] = 100
    sure_fg[:,len(markers[0,:])-41:len(markers[0,:])-1] = 100

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[sure_fg==0] = 0
    
    mask = cv2.watershed(img,markers)
    mask[mask == 201] = 255
    mask[mask == 101] = 0
    mask[mask == -1] = 255
    return mask
