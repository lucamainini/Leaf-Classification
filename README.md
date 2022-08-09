# Leaf-Classification
Classification of leaves

## 1. Introduction
The problem we had to face is the classification of leaves images. Our dataset is composed by 17.728 images divided into 14 classes: Apple, Blue- berry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry and Tomato. Most of the images have dimensions 256 x 256 and have been isolated on a black background except for three outliers with different shapes; we resized them either before feeding them to the network or in the network itself. We first noticed the unbalanceness of our classes, as shown in *Figure 1*. To address this issue, we experimented two different approaches: class weights and dataset balancing via upsampling. We first built a simple model to check what we could achieve with very few levels, then we experimented the transfer learning technique with some of the most important models in image recognition.

## 2. Data Augmentation
Since some classes contain a limited number of data, we have decided to use the data augmentation technique. We initially tried to figure out how to modify the images, particularly how to set the parameters of ImageDataGenerator. Displaying generated images, we could broadly understand the range of these hyperparameters: for instance, a brightness parameter superior to 2 generates meaningless images.
We then trained our model on images generated with different parameters saving for each combination of values the model and the score obtained on the validation. Then, we would have liked to test these models on the CodaLab set and then choose the best parameters with respect to the obtained value [`Data_Augmentation_Tuning.ipynb`]. Unfortunately, we had a too limited number of attempts, so we leave this tuning procedure as improvements to be done in the future.
We also tried implementing a custom augmentation to simulate a more realistic environment as shown in Figure 2 [`Custom_Augmentation.ipynb`] but this proved unsuccessful.

## 3. First Model

We first consider a first simple model, constituted by 3 increasing-depth blocks of 2 convolutional and activation layers alternated with layers of Max Pooling and, at the top, a fully connected network with one hidden layer. To reduce overfitting, we have also added 2 dropout layers. The input has been normalized by dividing by 255. The model is represented in Figure 3 and summarized in Table 1.
Initially, we had extracted a small part of the data to use it as a test set. However, the accuracy on this test set was very different from that obtained on the competition test. Then, we have decided to use all the available dataset to carry out training and validation and to test the model through the challenge platform. The performance of this model was not so low (score: 0.64), but extremely improvable especially for the classes Potato, Apple, Cherry and Corn.

## 4. Two approaches to face classes unbalanceness
At this stage of the project, we were faced with the fact that the classes in our training dataset were not balanced.

### 4.1 Class Weights

In order to solve the problem, we have decided to modify the cost function to minimize, giving more weight to smaller classes such that the final loss function is a weighted average of each class loss function:

$$
\begin{align*}
    L &= \sum_{i=1}^{14} w_i\cdot L_i \\
    w_i &= \frac{N}{14\cdot n_i}  
\end{align*}
$$
