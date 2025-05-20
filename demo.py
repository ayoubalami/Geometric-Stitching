import cv2
import numpy as np
from typing import Dict, List, Tuple

# Configs for HSV thresholds + contour-area limits for the dot colours and edges for frames
CONFIG: dict[str, Dict[int, Dict[str, int]]] = {
    "edges_top": {
        1: dict(l_h=80,  l_s=90,  l_v=187,  u_h=139, u_s=189, u_v=255, min_area=170, max_area=3000),
        2: dict(l_h=0,   l_s=0,   l_v=0,    u_h=255, u_s=255, u_v=85,  min_area=35,  max_area=420),
    },
    "edges_bottom": {
        1: dict(l_h=94,  l_s=45,  l_v=0,  u_h=111, u_s=153, u_v=201, min_area=170, max_area=3000),
        2: dict(l_h=0,   l_s=0,   l_v=0,    u_h=255, u_s=255, u_v=85,  min_area=35,  max_area=420),
    },
    "white": {
        1: dict(l_h=0,   l_s=0,   l_v=250,  u_h=105, u_s=15,  u_v=255, min_area=40,  max_area=450),
        2: dict(l_h=0,   l_s=0,   l_v=245,  u_h=90,  u_s=21,  u_v=255, min_area=50,  max_area=470),
    },
    "black": {
        1: dict(l_h=33,  l_s=61,  l_v=68,   u_h=99,  u_s=206, u_v=197, min_area=50,  max_area=900),
        2: dict(l_h=33,  l_s=61,  l_v=68,   u_h=99,  u_s=206, u_v=197, min_area=50,  max_area=900),
    },
    "output_size": {1: (460, 900), 2: (460, 900)},  # width, height
    "resize":      {1: 0.60, 2: 0.90},              # display shrink factors
    "video": {
        "A": "100.mp4",  # left camera
        "B": "200.mp4",  # right camera
        "start_A": 23,   # frame offset
        "start_B": 5,
    },
}
 
DOT_CFG = {
    "white": {
        "lower": (0,   0,   250),
        "upper": (105, 15,  255),
     },
    "black": {
        "lower": (33,  61,   68),
        "upper": (99, 206,  197),
     },
}

DOT_CFG_2 = {
    "white": {
        "lower": (0,   0,   245),
        "upper": (90,  21,  255),
     },
    "black": {
        "lower": (33,  61,   68),
        "upper": (99, 206,  197),
     },
}

def order_points(points):

    # Convert to numpy array
    points = np.array(points, dtype="float32")
    
    # Sum and difference of points to find corners
    s = points.sum(axis=1)  # Sum of x + y
    diff = np.diff(points, axis=1)  # Difference x - y

    # Top-left: smallest sum
    top_left = points[np.argmin(s)]
    # Bottom-right: largest sum
    bottom_right = points[np.argmax(s)]
    # Top-right: smallest difference
    top_right = points[np.argmin(diff)]
    # Bottom-left: largest difference
    bottom_left = points[np.argmax(diff)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")

def warp_perspective(image, points, output_size):
  
    
    # Define the source and destination points
    src_points = np.array(points, dtype="float32")
    dst_points = np.array([
        [0, 0],
        [output_size[0] - 1, 0],
        [output_size[0] - 1, output_size[1] - 1],
        [0, output_size[1] - 1]
    ], dtype="float32")


    # Calculate the perspective transformation matrix

    start_tick = cv2.getTickCount()
    # matrix = cv2.getPerspectiveTransform(src_points, dst_points)  # avg time: 0.000021 seconds
    matrix, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC) #avg  time: 0.000108 seconds
    end_tick = cv2.getTickCount()
    total_PerspectiveTransform_time.append((end_tick - start_tick) / cv2.getTickFrequency())

    # Perform the perspective warp
    start_tick = cv2.getTickCount()
    warped_image = cv2.warpPerspective(image, matrix, output_size)
    end_tick = cv2.getTickCount()
    total_warpPerspective_time.append((end_tick - start_tick) / cv2.getTickFrequency())

    return warped_image

def homographic_stitch_image(img2, img1, white2, white1, black2, black1):
    # Compute transformation matrix
    try:
        src_points = np.array([white2, black2], dtype=np.float32)
        dst_points = np.array([white1, black1], dtype=np.float32)
        M, _ = cv2.estimateAffinePartial2D(src_points, dst_points)

        # Get dimensions of both images
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        # Transform corners of img2 to find output bounds
        corners_img2 = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]], dtype=np.float32)
        warped_corners = cv2.transform(corners_img2.reshape(-1, 1, 2), M).reshape(-1, 2)

        # Combine with img1 corners to get full bounds
        all_corners = np.vstack([warped_corners, np.array([[0, 0], [w1, 0], [0, h1], [w1, h1]])])
        x_min, y_min = np.floor(all_corners.min(axis=0)).astype(int)
        x_max, y_max = np.ceil(all_corners.max(axis=0)).astype(int)

        # Calculate canvas size and offset
        canvas_width = x_max - x_min
        canvas_height = y_max - y_min
        offset_x = -x_min
        offset_y = -y_min

        # Adjust transformation matrix for canvas placement
        M[0, 2] += offset_x
        M[1, 2] += offset_y

        # Warp img2 onto the canvas
        img2_warped = cv2.warpAffine(img2, M, (canvas_width, canvas_height))

        # Place img1 onto the canvas
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        canvas[offset_y:offset_y+h1, offset_x:offset_x+w1] = img1

        # Blend images (use np.maximum for simple overlay instead of addWeighted)
        result = cv2.addWeighted(canvas, 0.5, img2_warped, 0.5, 0)

        height, width = result.shape[:2]

        # Calculate half dimensions
        new_width = int(width * 0.7)
        new_height = int(height * 0.7)

        # Resize the image to half size
        result = cv2.resize(result, (new_width, new_height))

        # Alternative: Direct overlay (uncomment if preferwhite)
        # mask = (img2_warped > 0).astype(np.uint8)
        # result = canvas * (1 - mask) + img2_warped * mask
        cv2.imshow('Stitched Image with Homo matrice', result)
        return result    
    except:
        return

def stitch_images_manual(img2, img1, white2, white1, black2, black1):
    """Stitch images using manually calculated rotation, scale, and translation."""
    # Calculate vectors between dots
    try:
        vec1 = np.array(black1) - np.array(white1)
        vec2 = np.array(black2) - np.array(white2)
            
        # Calculate rotation angle
        angle1 = np.arctan2(vec1[1], vec1[0])
        angle2 = np.arctan2(vec2[1], vec2[0])
        theta = angle1 - angle2
        
        # Calculate scale factor
        scale = np.linalg.norm(vec1) / (np.linalg.norm(vec2) + 1e-6)  # Prevent division by zero
        
        # Calculate translation components
        x2, y2 = white2
        x1, y1 = white1
        tx = x1 - scale * (x2 * np.cos(theta) - y2 * np.sin(theta))
        ty = y1 - scale * (x2 * np.sin(theta) + y2 * np.cos(theta))
        
        # Build affine matrix
        M = np.array([
            [scale * np.cos(theta), -scale * np.sin(theta), tx],
            [scale * np.sin(theta),  scale * np.cos(theta), ty]
        ], dtype=np.float32)
        
        # Create canvas to hold both images
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Transform corners to find canvas bounds
        corners = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]], dtype=np.float32)
        warped_corners = cv2.transform(corners.reshape(-1, 1, 2), M).reshape(-1, 2)
        all_corners = np.vstack([warped_corners, [[0, 0], [w1, 0], [0, h1], [w1, h1]]])
        
        # Calculate canvas dimensions
        x_min, y_min = np.floor(all_corners.min(axis=0)).astype(int)
        x_max, y_max = np.ceil(all_corners.max(axis=0)).astype(int)
        canvas_w = x_max - x_min
        canvas_h = y_max - y_min
        
        # Adjust translation for canvas offset
        M[0, 2] -= x_min
        M[1, 2] -= y_min
        
        # Warp img2 onto canvas
        warped = cv2.warpAffine(img2, M, (canvas_w, canvas_h))
        
        # Place img1 onto canvas
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        offset_x = -x_min
        offset_y = -y_min
        canvas[offset_y:offset_y+h1, offset_x:offset_x+w1] = img1
        
        # Blend images
        result = cv2.addWeighted(canvas, 0.5, warped, 0.5, 0)
        
        height, width = result.shape[:2]

        # Calculate half dimensions
        new_width = int(width * 0.7)
        new_height = int(height * 0.7)

        # Resize the image to half size
        result = cv2.resize(result, (new_width, new_height))

        cv2.imshow('Stitched Image manualy', result)
        return result
    except:
        print("err")
        return

def extract_edges_and_anchors(frame,id_frame):
    current_frame_copy=frame.copy()
        # Get trackbar positions
    hsv_frame = cv2.cvtColor(current_frame_copy, cv2.COLOR_BGR2HSV)
    
    # Define lower and upper bounds for the blue color
    if(id_frame==2 ):
        
        EDGE_HSV_BANDS = [
            # (lower-H, lower-S, lower-V) , (upper-H, upper-S, upper-V)
            ((94,  45,   0),  (111, 153, 201)),   # bottom band
            ((0,    0,   0),  (255, 255,  85)),   # top band
        ]

        # ── build the mask by OR-ing each band’s inRange result ───────────────────
        mask_edges = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)  # start with all-black
        for lower, upper in EDGE_HSV_BANDS:
            mask_edges |= cv2.inRange(hsv_frame, np.array(lower), np.array(upper))

        # (optional) apply morphology once at the end
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_edges = cv2.morphologyEx(mask_edges, cv2.MORPH_OPEN,  kernel)
        mask_edges = cv2.morphologyEx(mask_edges, cv2.MORPH_CLOSE, kernel)
        edges_min_area = 35
        edges_max_area = 420
    else:
        # ── single edge-detection HSV band for id_frame == 1 ───────────────────────
        EDGE_HSV_BAND = ((80, 90, 187),   # lower-H, S, V
                        (139, 189, 255)) # upper-H, S, V
        # build mask
        lower, upper = map(np.array, EDGE_HSV_BAND)
        mask_edges = cv2.inRange(hsv_frame, lower, upper)

        # (optional) morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_edges = cv2.morphologyEx(mask_edges, cv2.MORPH_OPEN,  kernel)
        mask_edges = cv2.morphologyEx(mask_edges, cv2.MORPH_CLOSE, kernel)
        edges_min_area = 170
        edges_max_area = 3000

    # output_size = (800, 650) if id_frame==1 else (1050, 400)  # Replace with desiwhite output dimensions
    # output_size = (460  ,300*3 ) if id_frame==2 else (300*3,490  )  # Replace with desiwhite output dimensions
    output_size = (460  ,300*3 ) if id_frame==2 else (460  ,300*3 )  # Replace with desiwhite output dimensions

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_edges = cv2.morphologyEx(mask_edges, cv2.MORPH_OPEN, kernel)
    mask_edges = cv2.morphologyEx(mask_edges, cv2.MORPH_CLOSE, kernel)
    contours_edges, _ = cv2.findContours(mask_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    edges=[]
    H, W = frame.shape[:2]
    bord_perc_h=.05
    bord_perc_w=.01
    max_w=50
    max_h=50
    for cnt in contours_edges:  
        x, y, w, h = cv2.boundingRect(cnt)
        if w<max_w and h<max_h and y < (int)(H*(1-bord_perc_h)) and x< (int)(W*1-bord_perc_w) and  y > (int)(H*bord_perc_h) and x> (int)(W*bord_perc_w) and cv2.contourArea(cnt) > edges_min_area and cv2.contourArea(cnt) < edges_max_area:
            center = (x + w // 2, y + h // 2)  # Calculate the center of the mark
            edges.append(center)
            cv2.circle(current_frame_copy, center, radius=7, color=(0, 12, 145 ), thickness=-1)  # Draw a filled circle

    # Perform the transformation
    warped=None
    try:
        warped = warp_perspective(frame, order_points(edges), output_size)
        # cv2.circle(warped, center, radius=50, color=(0, 255, 255), thickness=-1)  # Draw a filled circle
        # if (warped is not None):
        #     cv2.imshow("warped", warped)
    except:
        warped=current_frame_copy

    # current_frame_copy=warped
    # Convert the current frame to HSV

    hsv_frame = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

    mask_black_dot=[]
    mask_white_dot=[]

    black_min_area = 50
    black_max_area = 900

    if(id_frame==2):
         
        white_min_area = 50
        white_max_area = 470
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # white-dot mask
        lw  = np.array(DOT_CFG_2["white"]["lower"])
        uw  = np.array(DOT_CFG_2["white"]["upper"])
        mask_white_dot = cv2.inRange(hsv_frame, lw, uw)
        mask_white_dot = cv2.morphologyEx(mask_white_dot, cv2.MORPH_OPEN,  kernel)
        mask_white_dot = cv2.morphologyEx(mask_white_dot, cv2.MORPH_CLOSE, kernel)

        # black-dot mask
        lb  = np.array(DOT_CFG_2["black"]["lower"])
        ub  = np.array(DOT_CFG_2["black"]["upper"])
        mask_black_dot = cv2.inRange(hsv_frame, lb, ub)
        mask_black_dot = cv2.morphologyEx(mask_black_dot, cv2.MORPH_OPEN,  kernel)
        mask_black_dot = cv2.morphologyEx(mask_black_dot, cv2.MORPH_CLOSE, kernel)
        
    else: 
        white_min_area = 40
        white_max_area = 450
        # -------------------------------------------------------------
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # ---- white dot mask ----
        lower_white = np.array(DOT_CFG["white"]["lower"])
        upper_white = np.array(DOT_CFG["white"]["upper"])
        mask_white_dot = cv2.inRange(hsv_frame, lower_white, upper_white)
        mask_white_dot = cv2.morphologyEx(mask_white_dot, cv2.MORPH_OPEN,  kernel)
        mask_white_dot = cv2.morphologyEx(mask_white_dot, cv2.MORPH_CLOSE, kernel)

        # ---- black dot mask ----
        lower_black = np.array(DOT_CFG["black"]["lower"])
        upper_black = np.array(DOT_CFG["black"]["upper"])
        mask_black_dot = cv2.inRange(hsv_frame, lower_black, upper_black)
        mask_black_dot = cv2.morphologyEx(mask_black_dot, cv2.MORPH_OPEN,  kernel)
        mask_black_dot = cv2.morphologyEx(mask_black_dot, cv2.MORPH_CLOSE, kernel)    

    # Find contours in the mask
    contours_white_dot, _ = cv2.findContours(mask_white_dot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_black_dot, _ = cv2.findContours(mask_black_dot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rect_points = np.array(edges, dtype=np.int32)
    rect_points = rect_points.reshape((-1, 1, 2))  # Reshape for use with cv2 functions
    #order points

    # cv2.polylines(current_frame_copy, [rect_points], isClosed=True, color=(255, 255, 0), thickness=3)
    rect_points = cv2.convexHull(rect_points, clockwise=True)
    cv2.polylines(current_frame_copy, [rect_points], isClosed=True, color=(255, 0, 0), thickness=2)
  
    # Filter contours based on area and draw circles around detected corners
    
    white_dot=[]
    for cnt in contours_white_dot:
        if cv2.contourArea(cnt) > white_min_area and cv2.contourArea(cnt) < white_max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            center = (x + w // 2, y + h // 2)  # Calculate the center of the mark
            cv2.circle(warped, center, radius=12, color=(12, 20, 190), thickness=-1)  # Draw a filled circle
            if(len(last_white_dots[id_frame-1])==0):
                last_white_dots[id_frame-1]=center
                white_dot.append(center)
                break
            
            if(len(last_white_dots[id_frame-1])>0):
                if ((last_white_dots[id_frame-1][0]-center[0])**2 + (last_white_dots[id_frame-1][1]-center[1])**2)**0.5 < 100: 
                    last_white_dots[id_frame-1]=center
                    white_dot.append(center) 
                    break 
    black_dot=[]

    for cnt in contours_black_dot:
        if cv2.contourArea(cnt) > black_min_area and cv2.contourArea(cnt) < black_max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            center = (x + w // 2, y + h // 2)  # Calculate the center of the mark
            cv2.circle(warped, center, radius=12, color=(12, 120, 1), thickness=-1)  # Draw a filled circle
            if(len(last_black_dots[id_frame-1])==0):
                last_black_dots[id_frame-1]=center
                black_dot.append(center)
                break
            
            if(len(last_black_dots[id_frame-1])>0):
                if ((last_black_dots[id_frame-1][0]-center[0])**2 + (last_black_dots[id_frame-1][1]-center[1])**2)**0.5 < 100: 
                    last_black_dots[id_frame-1]=center
                    black_dot.append(center) 
                    break
            
    return (edges,white_dot,black_dot),( current_frame_copy,warped,mask_edges,mask_white_dot,mask_black_dot)

if __name__ == "__main__":
    last_white_dots = [[], []]
    last_black_dots = [[], []]

    video_path_A = "100.mp4"
    video_path_B = "200.mp4"

    cap_A = cv2.VideoCapture(video_path_A)
    cap_B = cv2.VideoCapture(video_path_B)

    # To Sync the two videos, Set initial frames (example: 5+18 for A, 5+0 for B)
    cap_A.set(cv2.CAP_PROP_POS_FRAMES, 23)
    cap_B.set(cv2.CAP_PROP_POS_FRAMES, 5)

    if not cap_A.isOpened() or not cap_B.isOpened():
        print("Error: Could not open video(s).")
        exit()

    paused = False
    current_image_A = None
    current_image_B = None

    total_PerspectiveTransform_time = []
    total_time = []
    total_warpPerspective_time = []

    while True:

        if not paused:
            ret_A, frame_A = cap_A.read()
            ret_B, frame_B = cap_B.read()
            if not ret_A or not ret_B:
                print("End of one or both videos.")
                break
            
            print("################# ")
            frame_A = cv2.resize(frame_A, (720, 1280))
            frame_B = cv2.resize(frame_B, (478, 850))
            current_image_A = frame_A.copy()
            current_image_B = frame_B.copy()

        if current_image_A is not None and current_image_B is not None: 
            # --- Process Frame A ---
            _, white_dot_a, black_dot_a, imgs_a = None, [], [], None
            try:
                (edges, white_dot_a, black_dot_a), (frame_copy_a, warped_a, *_ ) = extract_edges_and_anchors(current_image_A, 1)
                frame_disp_a = cv2.resize(frame_copy_a, (int(frame_copy_a.shape[1] * 0.6), int(frame_copy_a.shape[0] * 0.6)))
                cv2.imshow("Frame A", frame_disp_a)
            except Exception as e:
                print(f"Error processing frame A: {e}")

            # --- Process Frame B ---
            _, white_dot_b, black_dot_b, imgs_b = None, [], [], None
            try:
                (edges, white_dot_b, black_dot_b), (frame_copy_b, warped_b, *_ ) = extract_edges_and_anchors(current_image_B, 2)
                cv2.imshow("Frame B", frame_copy_b)
            except Exception as e:
                print(f"Error processing frame B: {e}")

            # --- Homographic Stitch ---
            try:
                start = cv2.getTickCount()
                homographic_stitch_image(warped_a, warped_b, white_dot_a, white_dot_b, black_dot_a, black_dot_b)
                end = cv2.getTickCount()
                total_time.append((end - start) / cv2.getTickFrequency())
                avg_homo = sum(total_time) / len(total_time)
                print(f"Homography avg time: {avg_homo:.6f} s")
            except Exception as e:
                print(f"Stitch (homography) error: {e}")

            # --- Manual Geometric Stitch ---
            try:
                if white_dot_a and white_dot_b and black_dot_a and black_dot_b:
                    start = cv2.getTickCount()
                    stitch_images_manual(
                        warped_a, warped_b,
                        white_dot_a[-1], white_dot_b[-1],
                        black_dot_a[-1], black_dot_b[-1]
                    )
                    end = cv2.getTickCount()
                    total_PerspectiveTransform_time.append((end - start) / cv2.getTickFrequency())
                    avg_manual = sum(total_PerspectiveTransform_time) / len(total_PerspectiveTransform_time)
                    print(f"Geometric avg time: {avg_manual:.6f} s")
                    print(f"Ratio geometric/homography: {avg_manual / avg_homo:.3f}")
            except Exception as e:
                print(f"Stitch (manual) error: {e}")

        key = cv2.waitKey(30) & 0xFF
        if key == ord('s'):
            paused = not paused
        elif key == ord('q'):
            break

    cap_A.release()
    cap_B.release()
    cv2.destroyAllWindows()
