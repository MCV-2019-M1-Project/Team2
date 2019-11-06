import cv2
import numpy as np


def BackgroundMask4(img):
    # Resize image
    original_size = img.shape[:2]
    resize_shape = (1000, 1000)
    img = cv2.resize(img, resize_shape)

    # Obtain gray level image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Obtain edges through Canny algorithm
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Dilate edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Obtain straight lines through Hough transform
    large_number = 100000
    percent_border = 0.05
    pict_start_perc = 0.3
    lines_wanted = {
        "top": [-large_number, large_number, int(pict_start_perc * img.shape[0]), int(pict_start_perc * img.shape[0])],
        "bottom": [-large_number, large_number, int((1 - pict_start_perc) * img.shape[0]),
                   int((1 - pict_start_perc) * img.shape[0])],
        "left": [int(pict_start_perc * img.shape[1]), int(pict_start_perc * img.shape[1]), -large_number, large_number],
        "right": [int((1 - pict_start_perc) * img.shape[1]), int((1 - pict_start_perc) * img.shape[1]), -large_number,
                  large_number]}
    last_mask = np.zeros(shape=(original_size[0], original_size[1]), dtype=np.uint8)
    last_mean_points = {"top": None, "left": None, "bottom": None, "right": None}
    for degrees_margin in [2, 5]:
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
        if lines is None:
            return last_mask, None
        for i in range(len(lines)):
            rho, theta = lines[i][0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + large_number * (-b))
            y1 = int(y0 + large_number * (a))
            x2 = int(x0 - large_number * (-b))
            y2 = int(y0 - large_number * (a))

            try:
                slope = (y2 - y1) / (x2 - x1)
            except:
                slope = 90
            # We want only vertical/horizontal lines with a certain degree deviation
            degrees = np.abs(np.arctan([slope]) * 180 / np.pi)
            if degrees < degrees_margin or (90 - degrees_margin) < degrees:
                # Horizontal line
                if degrees < degrees_margin:
                    # Line is in the %percent_border%
                    if np.mean([y1, y2]) < percent_border * img.shape[0] or np.mean([y1, y2]) > (1 - percent_border) * \
                            img.shape[0]:
                        continue
                    # Horizontal line more to the top than registered one
                    if np.mean([y1, y2]) < np.mean(lines_wanted["top"][2:4]):
                        lines_wanted["top"] = [x1, x2, y1, y2, degrees]
                    # elif degrees < lines_wanted["top"][4] and np.mean([x1,x2]) < np.mean(lines_wanted["top"][2:4])+0.1*img.shape[0]:
                    #     lines_wanted["top"] = [x1,x2,y1,y2,degrees]
                    # Horizontal line more to the bottom than the registered one
                    elif np.mean([y1, y2]) > np.mean(lines_wanted["bottom"][2:4]):
                        lines_wanted["bottom"] = [x1, x2, y1, y2, degrees]
                    # elif degrees < lines_wanted["bottom"][4] and np.mean([x1,x2]) < np.mean(lines_wanted["bottom"][2:4])-0.1*img.shape[0]:
                    #     lines_wanted["bottom"] = [x1,x2,y1,y2,degrees]
                # Vertical line
                elif (90 - degrees_margin) < degrees:
                    # Line is in the %percent_border%
                    if np.mean([x1, x2]) < percent_border * img.shape[1] or np.mean([x1, x2]) > (1 - percent_border) * \
                            img.shape[1]:
                        continue
                    # Vertical line more to the left than the registered one
                    if np.mean([x1, x2]) < np.mean(lines_wanted["left"][:2]):
                        lines_wanted["left"] = [x1, x2, y1, y2, degrees]
                    # elif degrees > lines_wanted["left"][4] and np.mean([x1,x2]) < np.mean(lines_wanted["left"][:2])+0.1*img.shape[1]:
                    #     lines_wanted["left"] = [x1,x2,y1,y2,degrees]
                    # Vertical line more to the right than the registered one
                    elif np.mean([x1, x2]) > np.mean(lines_wanted["right"][:2]):
                        lines_wanted["right"] = [x1, x2, y1, y2, degrees]
                    # elif degrees > lines_wanted["right"][4] and np.mean([x1,x2]) < np.mean(lines_wanted["right"][:2])-0.1*img.shape[1]:
                    #     lines_wanted["right"] = [x1,x2,y1,y2,degrees]

                # print("x1",x1,"x2",x2,"y1",y1,"y2",y2,"theta",theta,"rho",rho)
                # print('slope',)

                # cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

        # Draw lines_wanted to an all_zeros image
        mask = np.zeros(shape=img.shape[:2], dtype=np.uint8)
        draw_perc_th = 0.15
        for k, v in lines_wanted.items():
            if k == "top":
                if np.mean(v[2:4]) >= int(pict_start_perc * img.shape[0]):
                    v[2] = int(draw_perc_th * img.shape[0])
                    v[3] = int(draw_perc_th * img.shape[0])
            elif k == "bottom":
                if np.mean(v[2:4]) <= int((1 - pict_start_perc) * img.shape[0]):
                    v[2] = int((1 - draw_perc_th) * img.shape[0])
                    v[3] = int((1 - draw_perc_th) * img.shape[0])
            elif k == "left":
                if np.mean(v[:2]) >= int(pict_start_perc * img.shape[1]):
                    v[0] = int(draw_perc_th * img.shape[1])
                    v[1] = int(draw_perc_th * img.shape[1])
            elif k == "right":
                if np.mean(v[:2]) <= int((1 - pict_start_perc) * img.shape[1]):
                    v[0] = int((1 - draw_perc_th) * img.shape[1])
                    v[1] = int((1 - draw_perc_th) * img.shape[1])
            # print(k,np.mean([v[0],v[1]]),np.mean([v[2],v[3]]))
            cv2.line(mask, (v[0], v[2]), (v[1], v[3]), (255, 255, 255), 2)

        # Floodfill from center of lines drawn
        center_x = int((np.mean(lines_wanted["bottom"][2:4] + np.mean(lines_wanted["top"][2:4]))) / 2)
        center_y = int((np.mean(lines_wanted["right"][:2]) + np.mean(lines_wanted["left"][:2])) / 2)
        mask_flood = np.zeros(shape=(mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        cv2.floodFill(mask, mask_flood, (center_y, center_x), (255, 255, 255));

        # # Floodfill from center of image
        # mask_flood = np.zeros(shape=(mask.shape[0]+2,mask.shape[1]+2),dtype=np.uint8)
        # cv2.floodFill(mask, mask_flood, (int(mask.shape[0]/2),int(mask.shape[1]/2)), (255,255,255));

        # Depaint lines
        for k, v in lines_wanted.items():
            cv2.line(mask, (v[0], v[2]), (v[1], v[3]), (0, 0, 0), 2)

        # Resize mask to original size
        mask = cv2.resize(mask, (original_size[1], original_size[0]))

        # If new mask contains old mask, we stay with new mask
        if (np.bitwise_or(mask, last_mask) == mask).all():
            last_mask = mask
            last_mean_points["top"] = int(np.mean(lines_wanted["top"][2:4]) * original_size[0] / resize_shape[0])
            last_mean_points["left"] = int(np.mean(lines_wanted["left"][:2]) * original_size[1] / resize_shape[1])
            last_mean_points["bottom"] = int(np.mean(lines_wanted["bottom"][2:4]) * original_size[0] / resize_shape[0])
            last_mean_points["right"] = int(np.mean(lines_wanted["right"][:2]) * original_size[1] / resize_shape[1])

    return last_mask, last_mean_points
