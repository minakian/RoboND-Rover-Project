[//]: # (Image References)
[image_0]: ./misc/rover_image.jpg
[image_1]: ./rock.png
[image_2]: ./transform.png
# Search and Sample Return Project
![alt text][image_0]

This project is modeled after the [NASA sample return challenge](https://www.nasa.gov/directorates/spacetech/centennial_challenges/sample_return_robot/index.html) and it will give you first hand experience with the three essential elements of robotics, which are perception, decision making and actuation.  You will carry out this project in a simulator environment built with the Unity game engine.  

## Writeup for the Rover Project
### Objective
The main focus of this project was to develop code for the rover that will map and autonomously navigate the terrain of the simulator.

### Modified / Added Functions
For the most part I followed along with the explanation video provided in class. The first modification was to the `perspective_transform()` function.
```
def perspect_transform(img, src, dst):

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image

    mask = np.ones_like(img[:,:,0])
    mask = cv2.warpPerspective(mask, M, (img.shape[1], img.shape[0]))
    ran = int(mask.shape[0]/2)
    for i in range(40):
        for j in range(mask.shape[1]):
            mask[i,j]=0


    return warped, mask
```
![alt text][image_1]
As done in the demonstration, I added the mask feature, but with a slight change. I change the top 80 pixels to 0. The further from the camera the landmark is, the less reliable the information, and there is always a cliff in view. This would in turn designate navigable terrain that is out of view of the camera as unnavigable. This alleviates some of this error.

The next addition was adding a function for rock detection.
```
def find_rocks(img, levels=(110, 110, 50)):
    rockpix = ((img[:,:,0] > levels[0]) \
              & (img[:,:,1] > levels[1]) \
              & (img[:,:,2] < levels[2]))

    color_select = np.zeros_like(img[:,:,0])
    color_select[rockpix] = 1

    return color_select
```
![alt text][image_2]
This follows with the video in using the thresholding technique but with color thresholds that are more applicable to the color if the golden rocks.

### process_image() Completion
Again, for the most part, I followed along with the demonstration video.
I followed the steps given as comments in the code.
1. Define source and destination points for perspective transform
```
  dst_size = 5
  bottom_offset = 6
  image = Rover.img
  source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
  destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                    [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                    [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                    [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                    ])
```
2. Apply perspective transform
```
  warped, mask = perspect_transform(image, source, destination)
```
3. Apply color threshold to identify navigable terrain/obstacles/rock samples
```
  threshed = color_thresh(warped)
  obs_map = np.absolute(np.float32(threshed) - 1) * mask
```
4. Update Rover.vision_image (this will be displayed on left side of screen)
```
  Rover.vision_image[:,:,2] = threshed * 255
  Rover.vision_image[:,:,0] = obs_map * 255
```
5. Convert map image pixel values to rover-centric coords
```
  xpix, ypix = rover_coords(threshed * mask)
```
6. Convert rover-centric pixel values to world coordinates
```
  world_size = Rover.worldmap.shape[0]
  scale = 2 * dst_size
  xpos = Rover.pos[0]
  ypos = Rover.pos[1]
  yaw = Rover.yaw
  x_world, y_world = pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale)

  obsxpix, obsypix = rover_coords(obs_map)
  obs_x_world, obs_y_world = pix_to_world(obsxpix, obsypix, xpos, ypos, yaw, world_size, scale)
```
7. Update Rover worldmap (to be displayed on right side of screen)
```
  Rover.worldmap[y_world, x_world, 2] += 10
  Rover.worldmap[obs_y_world, obs_x_world, 0] += 1
```
8. Convert rover-centric pixel positions to polar coordinates
```
  dist, angles = to_polar_coords(xpix, ypix)
  Rover.nav_angles = angles
```
9. Finding Rocks
```
  rock_map = find_rocks(warped, levels=(110, 110, 50))
  if rock_map.any():
      rock_x, rock_y = rover_coords(rock_map)

      rock_x_world, rock_y_world = pix_to_world(rock_x, rock_y, xpos, ypos, yaw, world_size, scale)
      rock_dist, rock_ang = to_polar_coords(rock_x, rock_y)
      rock_idx = np.argmin(rock_dist)
      rock_xcen = rock_x_world[rock_idx]
      rock_ycen = rock_y_world[rock_idx]

      Rover.worldmap[rock_ycen, rock_xcen, 1] = 255
      Rover.vision_image[:,:,1] = rock_map * 255
  else:
      Rover.vision_image[:,:,1] = 0
```
A video of this can be found in `./output` that was built from my own training data collection.

## Simulator information
Screen Res = 1280x960
Graphics Quality = Fantastic

## The Simulator
The first step is to download the simulator build that's appropriate for your operating system.  Here are the links for [Linux](https://s3-us-west-1.amazonaws.com/udacity-robotics/Rover+Unity+Sims/Linux_Roversim.zip), [Mac](	https://s3-us-west-1.amazonaws.com/udacity-robotics/Rover+Unity+Sims/Mac_Roversim.zip), or [Windows](https://s3-us-west-1.amazonaws.com/udacity-robotics/Rover+Unity+Sims/Windows_Roversim.zip).  

You can test out the simulator by opening it up and choosing "Training Mode".  Use the mouse or keyboard to navigate around the environment and see how it looks.

## Dependencies
You'll need Python 3 and Jupyter Notebooks installed to do this project.  The best way to get setup with these if you are not already is to use Anaconda following along with the [RoboND-Python-Starterkit](https://github.com/ryan-keenan/RoboND-Python-Starterkit).


Here is a great link for learning more about [Anaconda and Jupyter Notebooks](https://classroom.udacity.com/courses/ud1111)

## Recording Data
I've saved some test data for you in the folder called `test_dataset`.  In that folder you'll find a csv file with the output data for steering, throttle position etc. and the pathnames to the images recorded in each run.  I've also saved a few images in the folder called `calibration_images` to do some of the initial calibration steps with.  

The first step of this project is to record data on your own.  To do this, you should first create a new folder to store the image data in.  Then launch the simulator and choose "Training Mode" then hit "r".  Navigate to the directory you want to store data in, select it, and then drive around collecting data.  Hit "r" again to stop data collection.

## Data Analysis
Included in the IPython notebook called `Rover_Project_Test_Notebook.ipynb` are the functions from the lesson for performing the various steps of this project.  The notebook should function as is without need for modification at this point.  To see what's in the notebook and execute the code there, start the jupyter notebook server at the command line like this:

```sh
jupyter notebook
```

This command will bring up a browser window in the current directory where you can navigate to wherever `Rover_Project_Test_Notebook.ipynb` is and select it.  Run the cells in the notebook from top to bottom to see the various data analysis steps.  

The last two cells in the notebook are for running the analysis on a folder of test images to create a map of the simulator environment and write the output to a video.  These cells should run as-is and save a video called `test_mapping.mp4` to the `output` folder.  This should give you an idea of how to go about modifying the `process_image()` function to perform mapping on your data.  

## Navigating Autonomously
The file called `drive_rover.py` is what you will use to navigate the environment in autonomous mode.  This script calls functions from within `perception.py` and `decision.py`.  The functions defined in the IPython notebook are all included in`perception.py` and it's your job to fill in the function called `perception_step()` with the appropriate processing steps and update the rover map. `decision.py` includes another function called `decision_step()`, which includes an example of a conditional statement you could use to navigate autonomously.  Here you should implement other conditionals to make driving decisions based on the rover's state and the results of the `perception_step()` analysis.

`drive_rover.py` should work as is if you have all the required Python packages installed. Call it at the command line like this:

```sh
python drive_rover.py
```  

Then launch the simulator and choose "Autonomous Mode".  The rover should drive itself now!  It doesn't drive that well yet, but it's your job to make it better!  

**Note: running the simulator with different choices of resolution and graphics quality may produce different results!  Make a note of your simulator settings in your writeup when you submit the project.**

### Project Walkthrough
If you're struggling to get started on this project, or just want some help getting your code up to the minimum standards for a passing submission, we've recorded a walkthrough of the basic implementation for you but **spoiler alert: this [Project Walkthrough Video](https://www.youtube.com/watch?v=oJA6QHDPdQw) contains a basic solution to the project!**.
