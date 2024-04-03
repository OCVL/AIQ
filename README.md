# AIQ
Automatic Image Quality Photoreceptor Based AO Images

The enclosed software was developed for determining the image quality of photoreceptor based AO images. Please cite the following paper when using the software in your publications:
B.D. Brennan, H. Heitkotter, J. Carroll, S. Tarima, and R. F. Cooper, "Quantifying image quality in AOSLO images of photoreceptors", Biomedical Optics Express, Vol. TBD(TBD), pp. TBD, 2024.

 The code for this software was written and tested using Python 3.10 using the PyCharm IDE.

 ### Main function of AIQ.py Script:
  * Allows for each image to be selected from a file explorer window
  * Allows for the name of the file where the SNR and the pull image path are saved
  * Determines the SNR of all the images selected

## How to Operate:
### To get started, we recommend opening up the "AIQ" Folder in the python editor of choice. 
Then open the python file called, "AIQ.py" and run the program. Following the user is prompted to select all images to be used in the pop-up file explorer. **Presently, there is no constraint on the naming convention of the images to be used.**

Following image selection, the output file will be saved to the same directory of the code. This output file contains the full path and name of the image and the SNR value determined by the algorithm. 

## Python 3.10 Dependencies:
### The following libraries are needed for the algorithm to run:
  * scipy
  * From skimage.transform the warp_polar function
  * numpy
  * cv2
  * From tkinter the fieldialog function
