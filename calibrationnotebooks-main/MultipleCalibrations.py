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
import numpy.typing as npt
import os
from checkerboard import detect_checkerboard

# %%
from dataclasses import dataclass

# %%
from ImageAnotater import ImageAnnotator
import ipywidgets as widgets


# %%
from matplotlib import pyplot as plt


# %%
@dataclass
class CameraCalibtationResult():
    ret : bool
    intrinsic_matrix : npt.NDArray
    distortion : npt.NDArray
    r_vecs : npt.NDArray
    t_vecs : npt.NDArray


# %%
# %matplotlib ipympl
image_files = ["test_data/Angle1.jpg",
               "test_data/Angle2.jpg",
               "test_data/Angle3.jpg",
               "test_data/Angle4.jpg"]

annotator = ImageAnnotator(image_files)

button = widgets.Button(description="Next Image")
button.on_click(lambda b: annotator.next_image())
annotator.button = button  # Pass the button to the annotator
display(button)

# %%
for im,bbox in zip(image_files,annotator.bounding_boxes):
    print(im,"\n")
    print(bbox,"\n\n")

# %%
images = [cv.imread(f,cv.COLOR_BGR2GRAY) for f in image_files]

# %%
rois1 = np.s_[60:180,650:810]

rois2 = np.s_[920:1200,1680:2000]

rois3 = np.s_[700:940,890:1175]

rois4 = np.s_[590:830,390:620]

rois = [rois1,rois2,rois3,rois4]

# %%
f, axs = plt.subplots(2,2)
for roi, im, ax  in zip(rois,images, axs.flatten()):
    ax.imshow(im[roi])


# %%
plt.close("all")

# %%
# Define the dimensions of checkerboard 
# this is edgeds in x and y NOT sqares
CHECKERBOARD = (9, 6) 

# %%
corners_list = []
scores = []
windows = [5,10,8,10]
for roi, im,win  in zip(rois,images,windows):
    corners, score = detect_checkerboard(im[roi],CHECKERBOARD,winsize=win)
    corners_list.append(corners)
    scores.append(score)

# %%
scores

# %%
corners.shape

# %%
f, axs = plt.subplots(2,2)
for roi, im,c, ax  in zip(rois,images,corners_list, axs.flatten()):
    corners = c #+ np.array([roi[0].start,roi[1].start])
    ax.imshow(im[roi])
    ax.scatter(corners[:,0,0],corners[:,0,1],s=1,c='r')


# %%
f, axs = plt.subplots(2,2)
original_corners = []
for roi, im,c, ax  in zip(rois,images,corners_list, axs.flatten()):
    corners = c + np.array([roi[1].start,roi[0].start])
    original_corners.append(corners)
    ax.imshow(im)
    ax.scatter(corners[:,0,0],corners[:,0,1],s=0.1,c='r')

# %%
#  3D points real world coordinates, theses are the Checker Board Corners 
objectp3d = np.zeros(( CHECKERBOARD[0]  
                      * CHECKERBOARD[1],  
                      3), np.float32) 

# %%
for x in range(CHECKERBOARD[0]):
    for y in range(CHECKERBOARD[1]):
        objectp3d[CHECKERBOARD[1]*x+y,:2] = np.array([x,y])

# %%
three_d_points = [objectp3d]
camera_calibrations = []
for c,im in zip(original_corners,images):
    two_d_points = [c.astype(np.float32) ]
    ret, matrix, distortion, r_vecs, t_vecs = cv.calibrateCamera( 
                                    three_d_points, 
                                    two_d_points,
                                    im.shape[:2][::-1], # size of the image, since cs graphics its reversed from numpy, just a convention...
                                    None,
                                    None
                                    )
    calib = CameraCalibtationResult(ret = ret,
                                    intrinsic_matrix=matrix,
                                    distortion=distortion,
                                    r_vecs=r_vecs[0],
                                    t_vecs=t_vecs[0])
    camera_calibrations.append(calib)

# %%
f,axs = plt.subplots(2,2)
for calib,corners,im,ax in zip(camera_calibrations,original_corners,images,axs.flatten()):
    reprojected_points, jacobian = cv.projectPoints(objectp3d,calib.r_vecs, calib.t_vecs,calib.intrinsic_matrix,calib.distortion)
    ax.imshow(im)
    ax.scatter(reprojected_points[:,0,0],reprojected_points[:,0,1],s=0.2,c='b')
    ax.scatter(corners[:,0,0],corners[:,0,1],s=0.1,c='r')

# %% [markdown]
# # What now could be further steps here?
# In CV:
# - undistort images using [undistort](https://docs.opencv.org/3.1.0/da/d54/group__imgproc__transform.html#ga69f2545a8b62a6b0fc2ee060dc30559d)
# - Use [stereocalibrate](https://docs.opencv.org/3.0-beta/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#stereocalibrate) to calibrate the cameras pairwise
# - rectify images to make matching easiert using [stereoRectify](https://docs.opencv.org/3.1.0/d9/d0c/group__calib3d.html#ga617b1685d4059c6040827800e72ad2b6)
# - match points in the rectified image using suitable tools, make use of the fact that epi polar lines are paralel in the images now. The epi lines can be found using [computeCorrespondEpilines](https://docs.opencv.org/3.1.0/d9/d0c/group__calib3d.html#ga19e3401c94c44b47c229be6e51d158b7). This
# - use the projection matrices returned by retify to project back to 3d using [triangulatepoints](https://docs.opencv.org/3.0-beta/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#triangulatepoints)

# %%

