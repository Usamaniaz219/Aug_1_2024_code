

import numpy as np
import easyocr
import cv2
import math
import os
import time
import re

bounding_boxes = []

def load_image(image_path):
  
    return cv2.imread(image_path)

def resize_image_to_multiple(image, tile_width, tile_height):
    # Resize the image to be a multiple of tile_width and tile_height
    new_width = math.ceil(image.shape[1] / tile_width) * tile_width
    new_height = math.ceil(image.shape[0] / tile_height) * tile_height
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return resized_image

def calculate_num_rows_and_cols(image, tile_width, tile_height):
    # Calculate the number of rows and columns
    num_rows = math.ceil(image.shape[0] / tile_height)
    num_cols = math.ceil(image.shape[1] / tile_width)
    return num_rows, num_cols

def extract_tile(image, start_x, start_y, tile_width, tile_height):
    # Extract the tile from the image
    end_x = min(start_x + tile_width, image.shape[1])
    end_y = min(start_y + tile_height, image.shape[0])
    return image[start_y:end_y, start_x:end_x]

def detect_text_in_tile(image, tile_width, tile_height, reader):
    # Initialize a list to store the bounding box coordinates
    bounding_boxes = []
    output_image = np.copy(image)
    # allowlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

    # Iterate over each row
    num_rows, num_cols = calculate_num_rows_and_cols(image, tile_width, tile_height)
    # print("Number of Rows and Number of Columns",num_rows,num_cols)
    for r in range(num_rows):
        # Iterate over each column
        for c in range(num_cols):
            # Calculate the starting coordinates of the tile
            start_x = c * tile_width
            start_y = r * tile_height

            # Extract the tile from the image
            tile = extract_tile(image, start_x, start_y, tile_width, tile_height)
            # print("Tile methods and attribute:",dir(tile))

          
            result = reader.readtext(tile)
            # print("easy ocr results:",result)

            # Check if any bounding boxes were returned
            if len(result) > 0:

                # Extract the bounding box coordinates and text from the result
                bounding_boxes_tile = [bbox for bbox, _, _ in result]
                # print("Bounding Box Tile",bounding_boxes_tile)

                # Map the bounding box coordinates back to the original image coordinates
                for bbox in bounding_boxes_tile:
                    # print("bbox type",type(bbox))
                    try:
                        [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] = bbox
                    except ValueError:
                        continue

                    # Adjust bounding box coordinates to fit the original image
                    # print("x1",x1)
                    x1 += start_x
                    y1 += start_y
                    x2 += start_x
                    y2 += start_y
                    x3 += start_x
                    y3 += start_y
                    x4 += start_x
                    y4 += start_y

                    mapped_bbox = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                    # print("mapped bbox",type(mapped_bbox))
                    mapped_bbox = np.array(mapped_bbox, dtype=np.int32)
                    mapped_bbox = mapped_bbox.reshape((-1, 1, 2))
                    bounding_boxes.append(mapped_bbox)
                    output_image = cv2.polylines(output_image, [mapped_bbox], isClosed=True, color=(0, 0, 255), thickness=2)

    return bounding_boxes, output_image

def retain_non_blank_bboxes(bboxes, source_mask_image):
    # Initialize the height and width based on the source mask image
    height, width = source_mask_image.shape[:2]

    # Create a blank (black) image
    blank_image = np.zeros((height, width), dtype=np.uint8)

    # List to retain non-blank bounding boxes
    retained_bboxes = []

    for bbox in bboxes:
        # print("bbox",bbox[0][0][0],bbox[0][0][1])
        # Create a copy of the blank image for drawing
        image_with_bbox = blank_image.copy()

        # Draw the bounding box on the image
        pt1 = (bbox[0][0][0], bbox[0][0][1])
        pt2 = (bbox[1][0][0], bbox[1][0][1])
        pt3 = (bbox[2][0][0], bbox[2][0][1])
        pt4 = (bbox[3][0][0], bbox[3][0][1])

        pts = np.array([pt1, pt2, pt3, pt4], np.int32)
        pts = pts.reshape((-1, 1, 2))

        cv2.fillPoly(image_with_bbox, [pts], (255))
        

        result_image = cv2.bitwise_and(image_with_bbox, source_mask_image) # Perform logical AND operation with the source mask image
       
        if np.any(result_image):  # Check if the result image is blank
            retained_bboxes.append(bbox)

    return retained_bboxes


intersected_bbox = []
threshold = 0.1

def retain_intersected_bbox(retained_bboxes,source_mask_image):
    height,width = source_mask_image.shape[:2]
    # print("Source Mask Image shape",source_mask_image)
    blank_image = np.zeros((height,width),dtype=np.uint8)
    if retained_bboxes==0:
        return intersected_bbox.clear()
        
    else:
        for i,bbox in enumerate(retained_bboxes):
            blank_image_with_bbox1 = blank_image.copy()
            # print("blank_image_shape",blank_image_with_bbox1.shape)
            pt1 = (bbox[0][0][0], bbox[0][0][1])
            pt2 = (bbox[1][0][0], bbox[1][0][1])
            pt3 = (bbox[2][0][0], bbox[2][0][1])
            pt4 = (bbox[3][0][0], bbox[3][0][1])
            pts = np.array([pt1,pt2,pt3,pt4],np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.fillPoly(blank_image_with_bbox1,[pts],(255))
            # print("i",i)
            
            # cv2.imwrite("source.jpg",source_mask_image)
            intersections = cv2.bitwise_and(blank_image_with_bbox1,source_mask_image)
            # print("intersection_type:",intersections)
            # cv2.imwrite("intersected_bbox.jpg",blank_image_with_bbox1)
        


            intersection_area = np.sum(intersections)
            # print("intersection area",intersection_area)
            bbox_mask_intersection_area = np.sum(blank_image_with_bbox1)
            # print("bbox_mask_intersection_area",bbox_mask_intersection_area)
            if bbox_mask_intersection_area==0:
                return 0
            intersection_percentage = intersection_area/bbox_mask_intersection_area
            # print("intersection percentage",intersection_percentage)
            if intersection_percentage >=threshold:
                intersected_bbox.append(bbox)
                # print("appended")

    return intersected_bbox


def draw_retained_bboxes_on_mask(intersected_bboxes, source_mask_image):
    # Create a copy of the source mask image
    mask_with_bboxes = source_mask_image.copy()

    height,width = source_mask_image.shape[:2]
    # print("Source Mask Image shape",source_mask_image)
    blank_image_2 = np.zeros((height,width),dtype=np.uint8)


    for bbox in intersected_bboxes:
        # print("bbox",bbox[0])
        # blank_image_2_copy = blank_image_2.copy()
        pt1 = (bbox[0][0][0], bbox[0][0][1])
        pt2 = (bbox[1][0][0], bbox[1][0][1])
        pt3 = (bbox[2][0][0], bbox[2][0][1])
        pt4 = (bbox[3][0][0], bbox[3][0][1])


        pts = np.array([pt1, pt2, pt3, pt4], np.int32)
        pts = pts.reshape((-1, 1, 2))

        cv2.fillPoly(mask_with_bboxes, [pts], (255))
        # cv2.fillPoly(blank_image_2, [pts], (255))

    # cv2.imwrite("Intersected_bounding_boxes_blank_mask.jpg",blank_image_2)
    return mask_with_bboxes
  


def process_image(source_image_path, source_mask_path, output_dir):
    mask_image_name = os.path.splitext(os.path.basename(source_image_path))[0]
    print(f"Processing mask image: {mask_image_name}")

    source_image = cv2.imread(source_image_path, cv2.IMREAD_GRAYSCALE)
    # cv2.imwrite("mask-temp.jpg",source_image)
    if source_image is None:
        print(f"Error reading mask image: {source_image_path}")
        return None
    reader = easyocr.Reader(['en'], gpu=True)
    bounding_boxes, output_image= detect_text_in_tile(source_image, tile_width, tile_height, reader)
    print("length of bounding boxes",len(bounding_boxes))

    source_mask_image = cv2.imread(source_mask_path, cv2.IMREAD_GRAYSCALE)
    if source_mask_image is None:
        print(f"Error reading bounding box image: {source_mask_path}")
        return None
    
    retained_bboxes = retain_non_blank_bboxes(bounding_boxes, source_mask_image)
    print("retained bounding boxes",len(retained_bboxes))
    intersected_bboxes = retain_intersected_bbox(retained_bboxes,source_mask_image)
    print("intersected bounding boxes",len(intersected_bboxes))
    
    final_mask_image = draw_retained_bboxes_on_mask(intersected_bboxes, source_mask_image)
    intersected_bbox = intersected_bboxes.clear()
    # cv2.imwrite("final_mask_image.jpg",final_mask_image)

    return final_mask_image
    
original_file_list = []
mask_dirs = []


def process_images(input_dir, output_dir, bounding_box_dir):
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()
    file_count = 0
    
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                original_file_list.append(filename)
                file_count += 1
                image_path = os.path.join(input_dir, filename)
                mask_image_name = os.path.splitext(os.path.basename(image_path))[0]
                print("mask image name :",mask_image_name)
            
                for root,dirs, files in os.walk(bounding_box_dir):              
                    # all_masks = os.listdir(bounding_box_dir)
                    for dir in dirs:
                        if dir==mask_image_name:
                            dir1 = os.path.join(root,dir)
                            mask_dirs.append(dir1)
                            
                            all_masks = os.listdir(dir1)
                            masks_renamed = [mask.replace(".jpg","").replace(".png","") for mask in all_masks]

                            for renamed_mask in masks_renamed:
                                mask_path = f"{bounding_box_dir}/{dir}/{renamed_mask}.jpg"
                                output_ = process_image(image_path, mask_path, output_dir)
                                if output_ is None:
                                    continue

                                output_subdir = os.path.join(output_dir, os.path.basename(os.path.dirname(image_path)))
                                output_subdir = f"{output_subdir}_{renamed_mask}_intersection_of_0.1"
                                os.makedirs(output_subdir, exist_ok=True)
                                
                                output_file_path = os.path.join(output_subdir, f"{renamed_mask}_output_mask.jpg")
                                cv2.imwrite(output_file_path,output_)
                                
                                print(f"Processed {filename} in {time.time() - start_time:.2f} seconds")
                        continue
                    break    
                

if __name__ == "__main__":

    tile_width = 1024
    tile_height = 1024

    input_directory = '/home/usama/Converted_1_jpg_from_tiff_july3_2024/'
    bounding_box_dir = '/home/usama/test_samples_aug_2024/'
    output_directory = '/home/usama/test_dir_2/'
    
    process_images(input_directory, output_directory, bounding_box_dir)



























































