**Repo Usage**
_chess_board_calibration.py_
Running this script will populate files in the data folder of the repo containing the camera matrix and distortion coefficients. To calibrate new camera sensors, take at least 8 images of a measured chess board and copy the format at the bottom of the script to output new calibration files. 

_wheel_ranger.py_
To run this script, use the following command format in the terminal "python wheel_ranger.py <path> [cam: 0.5x / 1x]". An example of running the scipt looks like "python wheel_ranger.py data/1x_burst_2 1x" The usage of "0.5x" and "1x" tells the script which calibration to apply. Upon completion, a new folder will be generated in the results folder where a series of 2x2 images will populate showing the output at various stages in the script for tuning purposes. 

**Personal Takeaways from this project**
1. I applied camera calibration techniques learned in a prior course to a practical application
2. Applied fundamental feature detection techniques learned in the computer vision course such as gaussian blurring, canny edge detection, corner detection, and 3d geometric principles from the pinhole model
3. Sought feature detection methods beyond course topics like edge dilation, contour splitting, and various prebuilt ellipse fitting functions; evaluating the trade-offs of each method
4. I learned to more closely note the terminal output of git commands as I did not record many of my pushes due to file sizes being too large. I experienced the difficulties with trying to push large data folders of images to git
