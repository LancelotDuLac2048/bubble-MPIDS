# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# Import required modules 
import cv2 as cv
import numpy as np 
import os

# %%
from ImageAnotater import ImageAnnotator
import ipywidgets as widgets


# %%
# %matplotlib ipympl
# we want interactive plots for this 
# that means using ipympl
# to make the figures apear
# we need to use 
# f, ax = plt.subplos()

# %%
from matplotlib import pyplot as plt

# %%
# Define the dimensions of checkerboard 
# this is edgeds in x and y NOT sqares
CHECKERBOARD = (9, 6) 

# %%
# stop the iteration when specified 
# accuracy, epsilon, is reached or 
# specified number of iterations are completed. 
criteria = (cv.TERM_CRITERIA_EPS + 
            cv.TERM_CRITERIA_MAX_ITER, 30, 0.1) 

# %%
filename = "test_data/extracted_15.jpg"
image = cv.imread(filename) 

# %%
im_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) 

# %%
f, ax = plt.subplots()
ax.imshow(im_gray)

# %% [markdown]
# Use the tool below to select a bounding box for the grid.  
# If `image_files` contains only one image, clicking it will close the interactive figure

# %%
# %matplotlib widget
image_files = ["test_data/extracted_15.jpg"]
annotator = ImageAnnotator(image_files)

button = widgets.Button(description="Next Image")
button.on_click(lambda b: annotator.next_image())
annotator.button = button  # Pass the button to the annotator
display(button)

# %%
print(annotator.bounding_boxes[0])

# %%
#bounding box of the calibration pattern, moved the pxs a bit for better results
x=655
y=60
x2=805
y2=180

# %%
# create a numpy slice object that allow us to index into the image mor readable
# opencv images are stored as y,x
roi = np.s_[y:y2,x:x2]

# %% [markdown]
# # Trying to use OpenVC

# %%
# select threshold from the roid
f,ax = plt.subplots()
ax.imshow(im_gray[roi])

# %%
# The corners in the thresholded image need to touch for the algorithm to work
# all below 95 -> black, all abovve white
thresh,bw_im = cv.threshold(im_gray,95, 255, cv.THRESH_BINARY)

# %%
f,ax = plt.subplots()
ax.imshow(bw_im[roi])

# %%
# Find the chess board corners 
# If desired number of corners are 
# found in the image then ret = true 
ret, corners = cv.findChessboardCorners( 
                bw_im[roi], (9,6),  
                cv.CALIB_CB_ADAPTIVE_THRESH  
                + cv.CALIB_CB_FAST_CHECK + 
                cv.CALIB_CB_NORMALIZE_IMAGE) 

# %%
ret

# %%
# We need 54 corners ( 6*9) for things to work
assert corners.shape[0] == 54


# %%
rois_copy = np.copy(bw_im[roi])
image = cv.drawChessboardCorners(rois_copy,  
                                  CHECKERBOARD,  
                                  corners, ret) 

# %%
f,ax = plt.subplots()
ax.imshow(rois_copy)

# %%

# %%
# stop the iteration when specified 
# accuracy, epsilon, is reached or 
# specified number of iterations are completed. 
criteria = (cv.TERM_CRITERIA_EPS + 
            cv.TERM_CRITERIA_COUNT, 30, 0.001) 

# %%
help(cv.cornerSubPix)

# %%
# If desired number of corners can be detected then, 
# refine the pixel coordinates and display 
# them on the images of checker board 


# Refining pixel coordinates a
# for given 2d points. 
rois_copy2 = np.copy(im_gray[roi])
corners2 = cv.cornerSubPix( 
    rois_copy2, corners, (5, 5), (-1, -1), criteria) 

# Draw and display the corners 
image = cv.drawChessboardCorners(rois_copy2,  
                                  CHECKERBOARD,  
                                  corners2, ret) 

f,ax = plt.subplots()

ax.imshow(rois_copy2)

# For our image gives this 

# %% [markdown]
# # Using an alternative implementation
# see [here](https://github.com/lambdaloop/checkerboard)

# %% [markdown]
# !pip install checkerboard

# %%
from checkerboard import detect_checkerboard

# %%
rois_copy3 = np.copy(im_gray[roi])

# %%
corners3, score = detect_checkerboard(rois_copy3,(9,6),winsize=5)

# %%
score

# %%
f,ax= plt.subplots()
image = cv.drawChessboardCorners(rois_copy3,  
                                  CHECKERBOARD,  
                                  corners3.astype(np.float32), True) 
ax.imshow(rois_copy3)

# %% [markdown]
# We cropped the image to the bounding box of the board, but we want callibration in coordinates of the whole image. So we need to add the x and y offsets

# %%
corners3[0]

# %%
corners_in_image = corners3 + np.array([x,y])

# %% [markdown]
# Always check!

# %%
corners_in_image.shape

# %%
f,ax = plt.subplots()
ax.imshow(im_gray,cmap='grey')
ax.scatter(corners_in_image[:,0,0],corners_in_image[:,0,1],s=2,c='r')

# %% [markdown]
#

# %% [markdown]
# # camera calibration
#
# To calibrate a camera we need 2d and 3d points that correspond to each other.
# The 2d points we get from the pattern detection.
#
# The 3d points we just `define`

# %%
#  3D points real world coordinates, theses are the Checker Board Corners 
objectp3d = np.zeros(( CHECKERBOARD[0]  
                      * CHECKERBOARD[1],  
                      3), np.float32) 

# %%
objectp3d.shape

# %%
for x in range(CHECKERBOARD[0]):
    for y in range(CHECKERBOARD[1]):
        objectp3d[CHECKERBOARD[1]*x+y,:2] = np.array([x,y])

# %%
objectp3d

# %%
three_d_points = [objectp3d]
two_d_points = [corners_in_image.astype(np.float32)]

# %% [markdown]
# # Caveat
#
# Be aware that `cornerSubPix` and `detect_checkerboard` return the points in different order. Since we want to match them to 3d points we need to be carefill

# %% [markdown]
# ## detect_checkerboard
#
# y direction first

# %%
f,ax = plt.subplots()
ax.scatter(corners_in_image[:,0,0],corners_in_image[:,0,1])
for i,xy in enumerate(corners_in_image[:,0]):
    ax.text(xy[0],xy[1],str(i))

# %% [markdown]
# ## cornerSubPix
#
# x direction first

# %%
f,ax = plt.subplots()
ax.scatter(corners2[:,0,0],corners2[:,0,1])
for i,xy in enumerate(corners2[:,0]):
    ax.text(xy[0],xy[1],str(i))

# %%
ret, matrix, distortion, r_vecs, t_vecs = cv.calibrateCamera( 
    three_d_points, 
    two_d_points,
    im_gray.shape[::-1], # size of the image, since cs graphics its reversed from numpy, just a convention...
    None,
    None
    ) 

# %%
matrix

# %%
reprojected_points, jacobian = cv.projectPoints(objectp3d,r_vecs[0], t_vecs[0],matrix,distortion)

# %%
f,ax = plt.subplots()
ax.scatter(reprojected_points[:,0,0],reprojected_points[:,0,1])
ax.scatter(corners_in_image[:,0,0],corners_in_image[:,0,1],s=2,c='r')
