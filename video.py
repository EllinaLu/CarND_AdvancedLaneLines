import argparse
import imageio
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
import pylab as pylab
import os

#imageio.plugins.ffmpeg.download()

from moviepy.editor import VideoFileClip
from moviepy.editor import ImageSequenceClip

#initialize the following global variables
mtx = []
dist = []
perspMat = []
perspMat_inv = []
imgShape = (1280, 720)
frameCount = 0
#max. pixel difference in first x position of line between frames
lineBaseXDiff_Threshold = 100

# window settings
window_width = 50 
window_height = 40 # Break image into 18 vertical layers since image height is 720
margin = 100 # How much to slide left and right for searching

ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meteres per pixel in x dimension

ploty = np.linspace(0, imgShape[0], 101 )

# load pickled distortion matrix
with open('camera_coefficients.p', mode='rb') as f:
    dist_pickle = pickle.load(f)
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"] 

#calculate perspective matrix from input src and dst points
def calculatePerspectiveMatrix(src, dst):
    m = cv2.getPerspectiveTransform(src, dst)
    inv = cv2.getPerspectiveTransform(dst, src)
    return m, inv

#hardcode the perspective src/dst points for now?
#start from bottom-left
#src = np.float32([[275, 670], [599, 451], [681, 451], [1048, 670]])
#src = np.float32([[242, 685], [599, 451], [681, 451], [1075, 685]])
src = np.float32([[242, 685], [599, 451], [681, 451], [1075, 685]])
dst = np.float32([[(imgShape[0] / 4), imgShape[1]],
    [(imgShape[0] / 4), 0],
    [(imgShape[0] * 3 / 4), 0],
    [(imgShape[0] * 3 / 4), imgShape[1]]])
#calculate perspective matrix
perspMat, perspMat_inv = calculatePerspectiveMatrix(src, dst)

#########Start of functions and classes ######

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        #1st x position of the line in birdview image
        self.baseXCenter = -1
        
    def updateBasePosition(self, newCenter):
        if self.baseXCenter != -1:
            if np.absolute(self.baseXCenter-newCenter) < lineBaseXDiff_Threshold:
                self.baseXCenter = newCenter
        else:
            self.baseXCenter = newCenter
            
        return self.baseXCenter       

#undistort input img and return result
#store the 2 images to file if needed
def undistort(img, mtx, dist, isStoreFile=False, fileName=''):
    fig = plt.figure()
    #undistort        
    dst = cv2.undistort(img, mtx, dist)
    
    if isStoreFile == True:
        # Plot the result
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        f.tight_layout()
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=30)
        ax2.imshow(dst)
        ax2.set_title('Undistorted', fontsize=30)
        pylab.savefig(fileName)
        
    plt.close(fig)
    return dst

#calculate perspective transform matrix and
# perform the perspective transform on the input image
def DoPerspectiveTransform(img, m):
    warped = cv2.warpPerspective(img, m, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    return warped

# Define a function that applies Sobel x or y, 
# then takes an absolute value and applies a threshold.
# Note: calling your function with orient='x', thresh_min=5, thresh_max=100
# should produce output like the example image shown above this quiz.
def abs_sobel_thresh(img, orient='x', thresh=(0, 255), sobel_kernel=3):
    isX = 0
    isY = 0    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        isX = 1
    if orient == 'y':
        isY = 1
    sobel = cv2.Sobel(gray, cv2.CV_64F, isX, isY, ksize=sobel_kernel)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled)
    binary_output[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
    return binary_output

# Define a function that applies Sobel x and y, 
# then computes the magnitude of the gradient
# and applies a threshold
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sX = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sY = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude 
    sMag = np.sqrt(np.square(sX) + np.square(sY))
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled = np.uint8(255*sMag/np.max(sMag))
    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled)
    binary_output[(scaled >= mag_thresh[0]) & (scaled <= mag_thresh[1])] = 1
    return binary_output

# Define a function that applies Sobel x and y, 
# then computes the direction of the gradient
# and applies a threshold.
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sX = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sY = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    absX = np.absolute(sX)
    absY = np.absolute(sY)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    dir = np.arctan2(absY, absX)
    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(dir)
    binary_output[(dir >= thresh[0]) & (dir <= thresh[1])] = 1
    return binary_output

# Define a function that thresholds the S-channel of HLS
# Use exclusive lower bound (>) and inclusive upper (<=)
def s_thresh(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    binary_output = np.zeros_like(hls[:,:,2])
    s = hls[:,:,2]
    binary_output[(s > thresh[0]) & (s <= thresh[1])] = 1
    return binary_output

#do lane detection
#output binary thresholded image
#function to encapsulate lane detection pipeline
def detect_lane_binary(undistorted, isSave=False):
    ksize = 3
    gradx = abs_sobel_thresh(undistorted, orient='x', sobel_kernel=ksize, thresh=(20, 150))
    grady = abs_sobel_thresh(undistorted, orient='y', sobel_kernel=ksize, thresh=(20, 150))
    mag_binary = mag_thresh(undistorted, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_threshold(undistorted, sobel_kernel=ksize+4, thresh=(0.7, 1.3))
    hls_binary = s_thresh(undistorted, thresh=(175, 255))
    
    combined = np.zeros_like(dir_binary)
    #combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (hls_binary == 1) ] = 1
    combined[((gradx == 1) & (grady == 1)) | (hls_binary == 1) ] = 1
    #combined[(gradx == 1) | (hls_binary == 1) ] = 1
    
    if isSave == True:
        # Plot the result
        #f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(16, 10))
        f.tight_layout()
        ax1.imshow(undistorted)
        ax1.set_title('Original Image', fontsize=20)
        ax2.imshow(hls_binary, cmap='gray')
        ax2.set_title('HLS thresholded.', fontsize=20)
        ax3.imshow(mag_binary, cmap='gray')
        ax3.set_title('Gradient Mag', fontsize=20)
        ax4.imshow(gradx, cmap='gray')
        ax4.set_title('Gradient X', fontsize=20)
        ax5.imshow(grady, cmap='gray')
        ax5.set_title('Gradient Y', fontsize=20)
        ax6.imshow(combined, cmap='gray')
        ax6.set_title('Combined', fontsize=20)
        pylab.savefig('output_images/{0}_thresh_interm.png'.format(os.path.basename(p)))
        plt.close(f)
    
    return combined

def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def find_window_centroids(image, window_width, window_height, margin, leftL, rightL):
    
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    window[int(window_width/4):int(3*window_width/4)] = 1.7
    window[0:int(window_width/4)] = 0.3
    window[int(3*window_width/4):window_width-1] = 0.3
    
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)
    
    l_center = leftL.updateBasePosition(l_center)
    r_center = rightL.updateBasePosition(r_center)
    
    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))
    
    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(image.shape[0]/window_height)):
	    # convolve the window into the vertical slice of the image
	    image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
	    conv_signal = np.convolve(window, image_layer)
	    # Find the best left centroid by using past left center as a reference
	    # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
	    offset = window_width/2
	    l_min_index = int(max(l_center+offset-margin,0))
	    l_max_index = int(min(l_center+offset+margin,image.shape[1]))
	    new_l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
	    if np.absolute(new_l_center - l_center) < window_width*1.5 :
	        l_center = new_l_center
	    #print  ('[', level, '] l_center= ', l_center, 'min/max= ', l_min_index, ', ', l_max_index)
	    
	    # Find the best right centroid by using past right center as a reference
	    r_min_index = int(max(r_center+offset-margin,0))
	    r_max_index = int(min(r_center+offset+margin,image.shape[1]))
	    new_r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
	    if np.absolute(new_r_center - r_center) < window_width*1.5 :
	        r_center = new_r_center
	    # Add what we found for that layer
	    window_centroids.append((l_center,r_center))

    return window_centroids

def calculateCurvature(fit, ploty):    
    y_eval = np.max(ploty)
    curve = ((1 + (2*fit[0]*y_eval + fit[1])**2)**1.5) / np.absolute(2*fit[0])
    return curve

#Process each image in the input video clip
def processImage(image):
    global frameCount
    frameCount = frameCount + 1
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    undistorted = undistort(image, mtx, dist, False)
    combined = detect_lane_binary(undistorted, False)
    birdView = DoPerspectiveTransform(combined, perspMat) 
    centroids = find_window_centroids(birdView, window_width, window_height, margin, left_lane, right_lane)       

    left_xvals = [i[0] for i in centroids]
    yvals = [imgShape[1]-(i-1)*window_height+0.5*window_height for i in range(len(centroids))]
    left_pixel_fit = np.polyfit(yvals, left_xvals, 2)
    left_fit = np.polyfit([i * ym_per_pix for i in yvals], [i * xm_per_pix for i in left_xvals], 2)
    
    right_xvals = [i[1] for i in centroids]
    right_pixel_fit = np.polyfit(yvals, right_xvals, 2)
    right_fit = np.polyfit([i * ym_per_pix for i in yvals], [i * xm_per_pix for i in right_xvals], 2)
    #print ('left_fit=', left_fit, 'right_fit=', right_fit)

    #calculate curvature
    left_curvature = calculateCurvature(left_fit, yvals)
    right_curvature = calculateCurvature(right_fit, yvals)
    
    #draw the detected lane onto original road image
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(birdView).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # If we found any window centers
    if len(centroids) > 0:
        # Points used to draw all the left and right windows
        l_points = np.zeros_like(birdView)
        r_points = np.zeros_like(birdView)
    
        # Go through each level and draw the windows 	
        for level in range(0,len(centroids)):
            # Window_mask is a function to draw window areas
    	    l_mask = window_mask(window_width,window_height,birdView,centroids[level][0],level)
    	    r_mask = window_mask(window_width,window_height,birdView,centroids[level][1],level)
    	    # Add graphic points from window mask here to total pixels found 
    	    l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
    	    r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255
    
        # Draw the results
        template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
        zero_channel = np.zeros_like(template) # create a zero color channel
        template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
        birdView = birdView * 255
        warpage = np.array(cv2.merge((birdView,birdView,birdView)),np.uint8) # making the original road pixels 3 color channels
        bird_with_window = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results
        
        #Draw the fitted polynomial curve
        left_fitx = left_pixel_fit[0]*ploty**2 + left_pixel_fit[1]*ploty + left_pixel_fit[2]
        right_fitx = right_pixel_fit[0]*ploty**2 + right_pixel_fit[1]*ploty + right_pixel_fit[2]
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, perspMat_inv, (imgShape[0], imgShape[1]))         
    # Combine the result with the original image
    result = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)
    
    #calculate distance to center
    veh_pos = result.shape[1]/2
    lane_middle = (left_fitx[-1] + right_fitx[-1])//2
    off_center = (veh_pos - lane_middle)*xm_per_pix
    #print ('Distance off center=', np.absolute(off_center))
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    str1 = str('Distance from center: {0:.2f} m').format(off_center)
    str2 = str('Left curvature: '+str(round(left_curvature, 2))+' m')
    str3 = str('Right curvature: '+str(round(right_curvature, 2))+' m')
    cv2.putText(result, str1, (50, 50), font, 1, (255, 255, 255), 2)
    cv2.putText(result, str2, (50, 80), font, 1, (255, 255, 255), 2)
    cv2.putText(result, str3, (50, 110), font, 1, (255, 255, 255), 2)
    
    if off_center > 1.0:
        f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(16, 10))
        f.tight_layout()
        ax1.imshow(undistorted)
        ax1.set_title('Original Image', fontsize=20)
        ax2.imshow(combined, cmap='gray')
        ax2.set_title('Combined.', fontsize=20)
        ax3.imshow(birdView, cmap='gray')
        ax3.set_title('Bird view', fontsize=20)
        ax4.imshow(bird_with_window, cmap='gray')
        ax4.set_title('Window fitting', fontsize=20) 
        ax5.imshow(result, cmap='gray')
        ax5.set_title('Result', fontsize=20)
        pylab.savefig('video_images/{0}.png'.format(frameCount))
        plt.close(f)
    
    return result

right_lane = Line()
left_lane = Line()

def main():    
    parser = argparse.ArgumentParser(description='Detect lanes in video')
    parser.add_argument(
        'inputFile',
        type=str,
        default='',
        help='Path to input video file.'
    )
    parser.add_argument(
        'outputFile',
        type=str,
        default='',
        help='Filename of output video')
    args = parser.parse_args()
     
    clip = VideoFileClip(args.inputFile)
    new_clip = clip.fl_image( processImage )
    new_clip.write_videofile(args.outputFile, audio=False)    


if __name__ == '__main__':
    main()