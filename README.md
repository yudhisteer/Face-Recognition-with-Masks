# Automating Attendance System using Face-Recognition-with-Masks

## Abstract

## Action Plan

## Phase 1: Face Recognition

### 1.1 Face Verification vs Face Recognition

### Verification
Face verification is quite simple. We can take our FaceID for example which we use to unlock our phone or at the airport when scanning our passport and verifying if it is really us. So it works in 2 steps:

- **We input the image or the ID of a person.**
- **We output whether the image ir ID is that of the claimed person.**

![image](https://user-images.githubusercontent.com/59663734/142724385-df2203a3-5f8c-435f-8ea6-7f9cc690861a.png)

It is a ```1:1``` problem. We expect to have a high accuracy of the face verification system - ```>99%``` - so that it can further be used into the face recognition system.

### Face Recognition
Face recognition is much harder than face verification. It is used mainly for attendance system in offices, or when facebook automatically tags your friend. The process is as such:

- **We have a database of K persons.**
- **We take the image of a person.**
- **We output the ID if the image is any of the K persons or if it is not.**

![image](https://user-images.githubusercontent.com/59663734/142724421-6f94ee75-3731-493b-86d7-3b0836847b80.png)

Face recognition is a ```1:K``` problem where K is the number of persons in our database.

### The One Shot Learning Dilemma
The problem with the face recognition for automating the attendance system at RT Knits is to be able for the neural network to recognize a particular person from only **one** image. Since we are going to have only one picture of the employees, the system should be able to recognize the person again. 

Suppose we have 100 employees then one simple solution would be to take this one image and feed it into a CNN and and output a softmax unit with 101 outputs(100 outputs is for the employees and the one left is to indicate none). However, this would not work well for two reasons:

1. We will have a very little training set hence, we will not be able to train a robust neural network.
2. Suppose we have another employee joining in then we would have to increment our outputs and re-train our Conv-Net. 

Instead we would want our neural network to learn a ```similarity function.```

### Similarity Function

The similarity function takes as input two images and output the degree of difference between the two images - <img src="https://latex.codecogs.com/svg.image?d(img_{1},&space;img_{2})" title="d(img_{1}, img_{2})" />

- For two images of the **same** person, we want <img src="https://latex.codecogs.com/svg.image?d(img_{1},&space;img_{2})" title="d(img_{1}, img_{2})" /> to be small.
- For two images of **different** persons, we want <img src="https://latex.codecogs.com/svg.image?d(img_{1},&space;img_{2})" title="d(img_{1}, img_{2})" /> to be big.


So how do we address the face verification problem?
- If <img src="https://latex.codecogs.com/svg.image?d(img_{1},&space;img_{2})&space;\leq&space;\tau&space;" title="d(img_{1}, img_{2}) \leq \tau " />, we predict as ```same```.
- If <img src="https://latex.codecogs.com/svg.image?d(img_{1},&space;img_{2})&space;>&space;&space;\tau&space;" title="d(img_{1}, img_{2}) > \tau " />, we predict as ```different```.

where <img src="https://latex.codecogs.com/svg.image?\tau&space;" title="\tau " /> is a threshold.

Given a new image, we use that function d to compare against the images in our database. If the image pairs are different then we would have a large number and if they are the same then we would have a small enough number that will be less than our ```threshold``` <img src="https://latex.codecogs.com/svg.image?\tau&space;" title="\tau " />.

![image](https://user-images.githubusercontent.com/59663734/142725497-32e644cd-0562-40ab-a569-46e57108918e.png)

For someone not in our database, when we will do the pairwise comparison and compute the function ```d``` then we would expect to get very large numbers for all the pairs as shown above. The ```d``` function solves the ```one shot learning``` problem whereby if someone new joins the team then we only need to add that new person's image to our database and it would work just fine. 

### Siamese Network
The idea of running two identical convolutional neural networks on two different inputs and comparing them is called a Siamese Neural Network. 

We feed in a picture of a person into a sequence of convolutions, pooling and fully connected layers and end up with a 128 feature vector. These 128 numbers is represented by <img src="https://latex.codecogs.com/png.image?\dpi{100}&space;f(x^{(i)})" title="f(x^{(i)})" /> and is called the ```encoding``` of the image where <img src="https://latex.codecogs.com/png.image?\dpi{100}&space;img_{(i)}&space;=&space;x^{(i)}" title="img_{(i)} = x^{(i)}" />.

![image](https://user-images.githubusercontent.com/59663734/142727867-2fa4f25e-3768-4c64-849b-6f4d5fe0e16d.png)

The to build a face recognition system would be to have a second picture feed in into that same CNN and compare their two 128 feature vectors. Then we need to define the function ```d``` which computes the norm of the difference of the two encodings:

![CodeCogsEqn (6)](https://user-images.githubusercontent.com/59663734/142728088-50968b17-3545-40af-84cb-071bfa9598eb.png)

To sum up: 
- our neural network define a **128-dimensional encoding** <img src="https://latex.codecogs.com/png.image?\dpi{100}&space;f(x^{(i)})" title="f(x^{(i)})" /> of an image. 
- we want to learn parameters such that the if two pictures are the **same** then the distance of the two encodings should be **small**.
- In contrast, for two **different** images, we want that distance to be **large**. 

When we vary the parameters of the different layers of our NN, we end up with different encodings. But we want to learn a specific set of parameters such that the above two conditions are met.

### Triplet Loss Function

![image](https://user-images.githubusercontent.com/59663734/142730472-09f4eace-cf55-4067-aeb2-1b279c8f428f.png)



## Phase 2: Mask Detection

## Phase 3: Face Recognition with Mask
![2-Figure1-1](https://user-images.githubusercontent.com/59663734/142723133-243c6b53-47ea-43e7-809b-c4dd790aa98f.png)

### 3.1 FaceNet
FaceNet is a deep neural network used for extracting features from an image of a person’s face. It was developed in 2015 by three researchers at Google: Florian Schroff, Dmitry Kalenichenko, and James Philbin.


![1-s2 0-S0925231220316945-gr3](https://user-images.githubusercontent.com/59663734/142723211-05e51b72-8794-442e-b1fa-ae9f5a6ed9bc.jpg)


## Conclusion

## References
