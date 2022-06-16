# Comp702-Project.-


COMP702 – Image Processing and Computer Vision
 
Abstract:
This project aims to identify the features of a given and old or new South African bank note (R10, R20, R50, R100, R200) referencing a collection of notes. It takes into consideration the sides (back and front), scale, and rotation of the note to identify a similarity between the given note and reference notes. This is done through preprocessing our given dataset of bank notes, enhancing, segmenting, performing feature extraction, and classifying the bank note. The proposed system will be able to work on old and new South African banknotes. Since the South African banknotes are easily classifiable based on the color and the texture, hence then we propose a system that will use GLCM to extract the texture of the banknote and the color of the image will be selected based on the RGB spaces. Where when it comes to classifying each note based on the denomination will be using SVM and KNN. The performance of these classifier was evaluated in terms of training performance and classification accuracies.

Introduction:
In the past decade, there has been growth in technology, which resulted in the development of modern banking services such as Automated Teller Machines, banknote-counting machines, and vending machines [1]. These modern banking services are expected to handle different banking notes and to be able to recognize their monetary value based on the distinguishing features each has. This process has been problematic since the advancement of technologies has introduced types of machinery that can create counterfeit banking notes. The problem resulting from the growth of counterfeit banknotes has attracted researchers to investigate new reliable methods that can recognize banknotes efficiently. Several methods that have been developed are either using Deep Learning (DL) and image processing techniques since they tend to be accurate and robust and are vital features in such systems [2].
In this system, we propose a banknote recognition system that will use the South African Rand (ZAR) banknote currencies and classify each according to each distinguishing feature. The primary technique that will be considered in achieving the task is image processing. This technique will focus on extracting distinguishable features of a banknote to recognize and classify it accordingly, whether it is a R10, R20, R50, R100 or R200. Extracting only useful features plays a significant role in reducing input data needed for the banknote classification while focusing on regions with high degrees of discrimination, hence making the process faster and yet reliable. The system of banknote recognition is expected to be robust as such; it should be insensitive to different directions the banknote is being used, i.e., can identify banknotes from each side and each direction. 
South Africa has recently changed the banknote from the year 2018, and they are still old banknotes which are currently in circulation. Hence, this system is designed in such a way that it should recognize the old banknotes and new ones based on the discriminative region that each banknote exhibits and based on similarly mapping. It should also be noted that this system is mainly based on banknote recognition and classification; it has not been designed to consider counterfeits notes; hence it may not be able to distinguish them from genuine banknotes. However, this system can be extended by adding other features captured using tools such as infrared or ultraviolet spectra [3].

Related Works:

Fake currency detection is another form of bank note recognition where you are comparing a counterfeit note to a real note. You would need to identify similarities between them to verify a given note. India is plagued by fake currency and black money, so a fake currency detector was developed by Tushar Agasti, Gajanan Burand, Patik Wade, and P. Chitra [4]. They realized that there were many machines that accepted money and dispensed consumer items, like candy and soft drinks, that may be inaccurate in detecting fake currency. These machines would need to be able to analyze the given note for invisible and visible features to verify its authenticity. The invisible features are seen with the use of an ultraviolet (UV) light [4]. Matlab is used for computational work and analysis as feature extraction of images can be difficult [4]. The methodology used to achieve this is scan the image under a UV light, greyscale the RGB image, perform edge detection, crop main characteristics and segment them, extract characteristics after segmentation, calculate the intensity of each feature, and verify if the condition is met [4]. These characteristics include, but are not limited to security thread, serial number, latent image, watermark, identification mark [4]. 
An alternative approach would be to extract the colour and texture features to form feature vectors where the test samples are classified using k-Nearest Neighbour based on those features [5]. These images are then fed into a Convolutional Neural Network for classification [5].  [6] utilised a 99 per cent accurate hierarchical technique for high-speed classification of US banknotes. Several discrete points from the entire image are chosen, and the average of the pixel at each location and its surrounding pixels is used to calculate the observed value for each point [4]. They took 32,850 samples from 12 different types of US currencies. The distance between the template vectors and the feature vectors from the observation points is measured and used to classify the banknote. Low-dimensionality vectors are used to achieve high-speed processing, lowering computational expenses. 

 
Methods and Techniques:

Image processing.
The image to be proposed must be put in a format appropriate for digital computing. It includes transformation of image from one format into other. It also involves cropping, binarization and noise removal using different features.
For this project four techniques have been compared and contrasted, namely Global Thresholding, Adaptive Thresholding, Otsu’s Algorithm and Canny Edge Detection.
For Global Thresholding, we applied five techniques, namely Binary, Binary Inverse, Truncated, Tozero and Tozero Inverse. All of which had a threshold value of 127. The results of this technique can be shown in image 1. 
Image 1: Results of various Global Thresholding techniques. 
For Adaptive Thresholding, we applied three techniques: Binary Thresholding, Adaptive Mean Thresholding and Adaptive Gaussian Thresholding. Before these can take place, the image was first smoothed using a 5x5 median filter. The results of these techniques can be shown in image 2.
Image 2: Results of the Adaptive Thresholding techniques. 
Using Otsu’s Algorithm, we applied three techniques: normal Global Thresholding, Global Thresholding with Otsu’s Algorithm and Otsu’s Algorithm with Gaussian filtering. The results from this are shown in image 3.
Image 3: Results of the various techniques applied with Otsu’s Algorithm and its histogram data.

The final image segmentation technique used was the Canny Edge detection technique. Our method had a minimum threshold of 100 and a maximum of 200. The results of this can be shown in image 4.
Image 4: Results of using the Canny Edge detection technique. 
From the results shown in the above experiments, the Adaptive Mean Thresholding technique showed promise and was used for image segmentation in this project.

Feature Extraction.

In this paper, first order statistics and second order statistics or Gray Level Co-occurrence Matrix (GLCM) are formulated to obtain statistical texture features. Five texture features are extracted from the second other statistics and using the GLCM, namely: Homogeneity, Correlation, Contrast, Dissimilarity and Energy. Where the other feature extraction involves extraction of the image colour of the banknote through the usage of RGB value which is the simple component that specifies the composition of red, green and blue pixel composition.

1.	Grey-Level Co-Occurrence Matrix and Haralick Features.

The features were first proposed by R.M. Haralick, the co-occurrence matrix illustration of texture features explores the grey level spatial dependence of texture[]. Where the co-occurrence matrix is characterized for an image by the method of partitioning of co-occurring ideals at a given offset. In order to calculate these, we have to use a library known as mahotas and that of skimage. These enabled us to be able to calculate the five Haralick features.  Where [] mentioned that selection of a set of appropriate input feature variables is an important issue in the building of a classifier and mentioning the importance of keeping the number of features as low as possible to increase satisfactory predictive performance.


The first was that of Homegenity, which gave the value that measures the closeness of the distribution of elements in the GLCM to the GLCM diagonal. Its range is [0 1]. 
 
Second it was correlation Coefficient : A co-measure that concludes the degree to which two variable's activities is associated. Its range is [-1 1]. Where  “-1” is a negative correlation value, and +1 is a positive correlation value.
 
Third it was energy : Which measures the homogeneousness of the image and can be calculated from the normalized gray level co-occurrence matrix. It is a suitable measure when it comes to detection of disorder in texture Image.
 
Followed by Contrast : Contrast is a measure of local level variations  and texture of shadow of depths which takes high values for image of high contrast.
 
Lastly feature is that of Dissimilarity: This feature is used to measure the distance between pairs of pixels in the region of interest.
 

While there are about 14 Haralick Features we chose these where five, where [] mentioned that these feature when are used together they provide high discriminative power to distinguish two different kind of images and demonstrated it mathematical. 


2.	Colour as a feature. 

Colour of the banknote is one of the most notable features that distinguishes one denomination. Since each banknote denomination is South Africa is having it distinct colour it made it a good feature when it comes to classifying the banknotes.  Where the main advantages of using the colour as features is the chromaticity (quality) of the colours does not change before mistreated banknotes; and since resizing the banknotes does not change its colour composition. Hence, in this paper we extract the colour features, under the RGB space. 
Here the colours are extracted from the original Image to find the composition of red, green, and green pixels. We then construct a feature vector fore each and every image to make up a training dataset that can be used through KNN classifier. 


Classification 

In facilitating these method we investigated two methods that of KNN and SVM since they tends to work very well with data set that is not so huge and require no to less training in most of the times.

K-Nearest Neighbours (KNN).

This method was incorporated in such that it used the feature vectors we used to classify the new images that were not used for training. They function by computing the distance from the unlabelled data to every training data point and selects the best k neighbours with the shortest distance [15]. In this experiment we used the colours feature vector and the five Haralick features to create labelled vector through the features and the name of the dataset we too the first integer part to make up the label for each banknote. Then we used the fit from sklearn library to fit the test and train dataset in such the KNN will be able to predict the banknote based on the lowest distance using the Euclidean distance from the labelled data and that which is unlabelled. Through changing the values of k we achieved different results but we chose that of 3 and which had a better accuracy.

Support Vector Machine (SVM).

The aim of SVM is to efficiently deal with a two-class classification problem. Given two patterns, the basic idea of SVM is to construct a good separating hyperplane as the decision surface in such a way that the margin of separation between these two patterns is maximized. In this case we used the first order statistics from Haralick features using the mahotas which enables us too find the first order statistics features then we used the mean of each to feature, to make a feature vector for each image. In this case we used the sklearn library to create a linear classifier and fit on the features as the input data while extracting the number of each image and keep it as labelled for training the model.

Dataset, training, and testing.
In this case we did not have massive dataset hence we ended up settling for classic feature-based classification methods such as that of KNN and SVM. The data given was that of images and it was not so labelled hence to facilitate that we had to use the first integer of each banknote name and use it as the label to label the data for training and evaluation.  


Results and Discussion

In this system that we proposed we used the method of KNN and SVM, where despite the normal outcomes where the SVM tends to perform very well in classifying the data in case it had an accuracy of around 70% and while the KNN had an average accuracy of around 90% when considering the using the confusion metrics and the classification report. It should be note since out dataset was about 50 banknotes which were also divided in 85% and 15% for training the dataset was so small hence the training was very low as for SVM since it mostly needs an adequate dataset to be able to extract features and present a better hyperplane to separate the banknotes and noted that the denomination is made up of 5 classes. While we also suspected the usage of linear SVM had the most impact when it comes to lower accuracy of SVM. Hence in discussion and evaluation will be based more on results of KNN.
The KNN weighted average for precision was 94.00, 91% and 90.0 for recall and f1-score respectively. While in the result from the classification reports showed below show that classification of the R20 was had much impact where having 67% of precision hence affecting the impact to the overall performance while it can be seen that other denomination were able to get even 100% terms of accuracy in different categories. Where checking at the F1- score are final deciders since, in their equation, they involve both recall and precision, telling us a percentage of positive prediction that accounts for all classes' performance, which is better since our class was imbalanced. Hence this had a good accuracy.


Conclusion.

The system that we proposed to be recognised different South Africa banknotes namely: R10, R20, R50, R100 and R100 was able to achieve an accuracy of 90% plus percent using KNN and got around 70% when using the SVM. With hope this can even get more better if we cut crop the images and focus on certain regions instead of using almost the whole banknote and while also seeing the room of growth if we can have more banknotes for training the model. Also, it was also suspected in terms of SVM lower accuracy that it can be increase by using another approach instead of linear SVM. Where the extension in this could be to have other features incorporated to even assist in fake detection of banknotes.




References:
1.	Lee, J.W., Hong, H.G., Kim, K.W. and Park, K.R., 2017. A survey on banknote recognition methods by various sensors. Sensors, 17(2), p.313.
2.	Hassanpour, H., Yasui, A. and Ardeshir, G., 2007, February. Feature extraction for paper currency recognition. In 2007 9th International Symposium on Signal Processing and Its Applications (pp. 1-4). IEEE.
3.	Sharma, B. and Kaur, A., 2012. Recognition of Indian paper currency based on LBP. International Journal of Computer Applications, 59(1).
4.	Agasti, T., Burand, G., Wade, P. and Chitra, P., 2017. Fake currency detection using image processing.
5.	Chowdhury, U., Jana, S. and Parekh, R., 2020. Automated System for Indian Banknote Recognition using Image Processing and Deep Learning. [online] IEEE Explore. Available at: <https://ieeexplore.ieee.org/abstract/document/9132850> [Accessed 7 June 2022].
6.	Jaitley, U., Why Data Normalization is necessary for Machine Learning models [online]
Available at: https://medium.com/@urvashilluniya/why-data-normalization-is-necessary-for-machine-learning-models-681b65a05029. [Accessed 16 June 2022]
7.	Haralick R. M., Shanmugam K., Disntein I., 1973. Textural Features for Image Classification 


