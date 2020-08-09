## Real Time Face Detector

### Harvard Course CSCI-S89 Introduction to Deep Learning

#### A Project by Ativeer Patni

Date: 08/09/2020

##### Data for this project can be found here https://drive.google.com/file/d/1nl2Zjo-MiQ40PnxvyeIJWy6xSPQXe8Wk/view?usp=sharing

##### Link to Kaggle Competition: https://www.kaggle.com/deadskull7/fer2013

##### Steps for Implementation:

Following steps are only if you want a demo of it, for training please download different script inside the Training_Folder (steps mentioned ahead)

1. Clone this repository.
2. Make sure you have the right requirements. (Better to install it before running it) (use the requirements.txt file above)
3. On command line, go to the folder where this is cloned.
4. Enter the command python model_load.py -m <model_name> -n <name_of_the_output_video_file>
5. Options for model_name:
	a. VGG for VGG16 model
	b. CNN for Custom made CNN
	c. Mobile for MobileNetV2
6. Name of the Video file is optional.
7. Press Q to Exit at anytime you want

If you want to run the training file: 
1. Please download the script ativeer_patni_harvard_project.ipynb and load it in Google Script (Or if you have enough GPU on your machine then skip colab)
2. Make sure you give enough credentials to run this script and mount the file properly (Skip this step if you are training on your local machine)
3. If you are running on your local Machine, please change the path of fer2013.zip/csv file to the path in your local machine.
4. Run the script.
5. Make sure you change the path where you want to download the trained model

