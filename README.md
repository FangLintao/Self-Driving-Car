# Self-Driving-Car
![image](https://github.com/FangLintao/Self-Driving-Car/blob/master/images/introduction.png)  
this project is to design neural network models to detect images'features from real world, so that autonomous cars are able to drive by themselves based on these detected features
# Data
training data: ![Ch2_002](https://github.com/udacity/self-driving-car/blob/master/datasets/CH2/Ch2_002.tar.gz.torrent)  
testing data: ![Ch2_001](https://github.com/udacity/self-driving-car/blob/master/datasets/CH2/Ch2_001.tar.gz.torrent)
### Brief Introduction
* Contain:Inside these two datasets, mian informations include steering angles, speed and torque from left, center and right cameras  
* Size: images inside datasets are 640*320  
### Reading Tool
Because the format of these two files are in rosbag, so it is able to export images from rosbag by check ![udacity-driving-reader tool](https://github.com/rwightman/udacity-driving-reader) from Mr.Ross Wightman 
# Preprocessing
### DataLoading
For 3DCNN+LSTM and TransferLearning models, we load images in different sizes. In ![Dataloading code](https://github.com/FangLintao/Self-Driving-Car/tree/master/DataLoading), two files can be found.

    ConsecutiveBatchSampler.py & UdacityDataset.py

#### 3DCNN+LSTM
* Loading Size: Batch_size * sequence_length * channels * height * width  
* Feeding Size: images in Batch_size * sequence_length * channels * 320 * 120   

      Note : This is reasonable to consider that if we predict images from certain sequency, then prediction angles could be analyzed based on former informations, which ensure strong support to improve precision on steering angle. This can be achieved by 3DCNN and LSTM structures.  

#### TransferLearning
* Loading Size: Batch_size * channels * height * width  
* Feeding Size: images in Batch_size * channels * 224 * 224   

      Note : TransferLearning excludes LSTM but includes 2DCNN, which means feeding each image into model would be good choice.

# Models
For ![3DCNN+LSTM and TransferLearning models](https://github.com/FangLintao/Self-Driving-Car/tree/master/model), two files can be found.

    Convolution3D.py & TransferLearning.py

### 3DCNN+LSTM
![image](https://github.com/FangLintao/Self-Driving-Car/blob/master/images/3DCNN%2BLSTM%20model.png)  
#### Brief Introduction
As above shows, insides 3D convolution layers, residual connection layers are added in order to tackle graident vanishing situation; When going through LSTM, memory property is to withdraw information from former images and output integrated infromation to next linear connection layers
### TransferLearning
![image](https://github.com/FangLintao/Self-Driving-Car/blob/master/images/TL%20model.png)  
Reference: ![Self-Driving Car Steering Angle Prediction Based on Image Recognition](https://arxiv.org/abs/1912.05440), Shuyang Du, Haoli Guo, Andrew Simpson, arXiv:1912.05440v1[sc.CV] 11.Dec.2019, page 4, "Figure 3. Architecture used for transfer learning model"  
#### Brief Introduction
As above shows, we use pretrain network of former 45 layers in ResNet50 to process images and then feed them into linear connection layers
# Results
below is loss values in two models in different stages  

![image](https://github.com/FangLintao/Self-Driving-Car/blob/master/images/result.png)  

For visualze outputs from our models, you can go to ![Visualization](https://github.com/FangLintao/Self-Driving-Car/tree/master/Visualization)

    Visualization.ipynb

### 3DCNN+LSTM
* below is attention map；  
![image](https://github.com/FangLintao/Self-Driving-Car/blob/master/images/3DCNN.png)  
* below is kernel images:  
![image](https://github.com/FangLintao/Self-Driving-Car/blob/master/images/CNN3D%2BLSTM%20-%20kernel.png)  
By using this 3DCNN+LSTM model, you should get this result
### TransferLearning
* below is attention map；  
![image](https://github.com/FangLintao/Self-Driving-Car/blob/master/images/TL.png)  
* below is kernel images:  
![image](https://github.com/FangLintao/Self-Driving-Car/blob/master/images/TransferLearning%20-%20kernel.png)  
By using this TransferLearning model, you should get this result  
### Result video
For the video, you can check ![attention map video](https://github.com/FangLintao/Self-Driving-Car/blob/master/images/attention%20video.mp4)
# Pointers and Acknowledgements
* Some of the model architectures are based on ![Self-Driving Car Steering Angle Prediction Based on Image Recognition](https://arxiv.org/abs/1912.05440).
* ![rwightman's docker tool](https://github.com/rwightman/udacity-driving-reader) was used to convert the round 2 data from ROSBAG to JPG.
* ![Pytorch](https://pytorch.org/) was used to build neural network models.









