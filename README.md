# Automating Attendance System using Face-Recognition-with-Masks

## Abstract

## Action Plan

## Phase 1: Face Recognition

### 1.1 Face Verification vs Face Recognition

### 1.1.1 Verification - Is this the same person?
Face verification is quite simple. We can take our FaceID for example which we use to unlock our phone or at the airport when scanning our passport and verifying if it is really us. So it works in 2 steps:

- **We input the image or the ID of a person.**
- **We verify if the output is the same as the claimed person.**

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/142724385-df2203a3-5f8c-435f-8ea6-7f9cc690861a.png" />
  </p>

It is a ```1:1``` problem. We expect to have a high accuracy of the face verification system - ```>99%``` - so that it can further be used into the face recognition system.

### 1.1.2 Face Recognition - Who is this person?
Face recognition is much harder than face verification. It is used mainly for attendance system in offices, or when facebook automatically tags your friend. The process is as such:

- **We have a database of K persons.**
- **We take the image of a person.**
- **We output the ID if the image is any of the K persons or if it is not.**

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/142724421-6f94ee75-3731-493b-86d7-3b0836847b80.png" />
 </p>

Face recognition is a ```1:K``` problem where K is the number of persons in our database.

### 1.2 The One Shot Learning Dilemma
The problem with the face recognition for automating the attendance system at RT Knits is to be able for the neural network to recognize a particular person from only **one** image. Since we are going to have only one picture of the employees, the system should be able to recognize the person again. 

Suppose we have 100 employees then one simple solution would be to take this one image and feed it into a CNN and and output a softmax unit with 101 outputs(100 outputs is for the employees and the one left is to indicate none). However, this would not work well for two reasons:

1. We will have a very little training set hence, we will not be able to train a robust neural network.
2. Suppose we have another employee joining in then we would have to increment our outputs and re-train our Conv-Net. 

Instead we would want our neural network to learn a ```similarity function.```

### 1.3 Similarity Function

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

### 1.4 Siamese Network
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

### 1.5 Triplet Loss Function
In Triplet Loss, we will be looking at three images at a time: an ```anchor```, a ```positive``` image(one who is similar to the anchor image) and a ```negative``` image(one who is different from the anchor image). We want the distance between the anchor and the positive to be minimum and the distance between the anchor and the negative image to be maximum. 

We denote the ```anchor``` as ```A```, ```positive``` as ```P``` and ```negative``` as ```N```. 

For a robust face recognition, we want the following:

<img src="https://latex.codecogs.com/png.image?\dpi{100}&space;\left\|f(A)-f(P)&space;\right\|^2\leq&space;\left\|f(A)-f(N)&space;\right\|^2" title="\left\|f(A)-f(P) \right\|^2\leq \left\|f(A)-f(N) \right\|^2" /> where <img src="https://latex.codecogs.com/svg.image?d(A,P)&space;=&space;\left\|f(A)-f(P)&space;\right\|^2" title="d(A,P) = \left\|f(A)-f(P) \right\|^2" /> and <img src="https://latex.codecogs.com/svg.image?d(A,N)&space;=&space;\left\|f(A)-f(N)&space;\right\|^2" title="d(A,N) = \left\|f(A)-f(N) \right\|^2" />

We can also write the equation above as :

<img src="https://latex.codecogs.com/png.image?\dpi{100}&space;\left\|f(A)-f(P)&space;\right\|^2&space;-&space;&space;\left\|f(A)-f(N)&space;\right\|^2\leq&space;0" title="\left\|f(A)-f(P) \right\|^2 - \left\|f(A)-f(N) \right\|^2\leq 0" />

To make sure the neural network does not output zero for all the encodings, i.e, it does not set all the encodings equal to each other, we modify the above equation such that the differenve between ```d(A,P)``` and ```d(A,N)``` should be <img src="https://latex.codecogs.com/png.image?\dpi{100}&space;0-\alpha&space;" title="0-\alpha " /> where <img src="https://latex.codecogs.com/png.image?\dpi{100}&space;\alpha&space;" title="\alpha " /> is called a ```margin```

Finally, the equation becomes:

<img src="https://latex.codecogs.com/svg.image?\left\|f(A)-f(P)&space;\right\|^2&space;-&space;\left\|f(A)-f(N)&space;\right\|^2&space;&plus;&space;\alpha&space;&space;\leq&space;0" title="\left\|f(A)-f(P) \right\|^2 - \left\|f(A)-f(N) \right\|^2 + \alpha \leq 0" />

For example, if we have d(A,N) = 0.50 and d(A,P) = 0.49, then the two values are too close to each other and it is not good enough. We would want d(A,N) to be much bigger than d(A,P) like 0.7 instead of 0.50. To achive this gap of 0.2, we introduce the margin <img src="https://latex.codecogs.com/png.image?\dpi{100}&space;\alpha&space;" title="\alpha " /> which helps push d(A,N) up or push d(A,P) down to achieve better results. 

To define our loss fucntion on a single triplet we need 3 images: ```A```, ```P``` and ```N```:

<img src="https://latex.codecogs.com/svg.image?L(A,P,N)&space;=&space;max(\left\|f(A)-f(P)&space;\right\|^2&space;-&space;\left\|f(A)-f(N)&space;\right\|^2&space;&plus;&space;\alpha,&space;0)" title="L(A,P,N) = max(\left\|f(A)-f(P) \right\|^2 - \left\|f(A)-f(N) \right\|^2 + \alpha, 0)" />

- we take the ```max``` of the loss because as long as <img src="https://latex.codecogs.com/svg.image?\left\|f(A)-f(P)&space;\right\|^2&space;-&space;\left\|f(A)-f(N)&space;\right\|^2&space;&plus;&space;\alpha" title="\left\|f(A)-f(P) \right\|^2 - \left\|f(A)-f(N) \right\|^2 + \alpha" /> <img src="https://latex.codecogs.com/png.image?\dpi{100}&space;\leq&space;" title="\leq " /> 0, the loss = 0.
-  However if <img src="https://latex.codecogs.com/svg.image?\left\|f(A)-f(P)&space;\right\|^2&space;-&space;\left\|f(A)-f(N)&space;\right\|^2&space;&plus;&space;\alpha" title="\left\|f(A)-f(P) \right\|^2 - \left\|f(A)-f(N) \right\|^2 + \alpha" /> > 0 then the loss = <img src="https://latex.codecogs.com/svg.image?\left\|f(A)-f(P)&space;\right\|^2&space;-&space;\left\|f(A)-f(N)&space;\right\|^2&space;&plus;&space;\alpha" title="\left\|f(A)-f(P) \right\|^2 - \left\|f(A)-f(N) \right\|^2 + \alpha" />. We will have a ```positive``` loss.

To define our ```cost function```:
<img src="https://latex.codecogs.com/png.image?\dpi{100}&space;J&space;=&space;\sum_{i=1}^{m}&space;=&space;L(A^{(i)},P^{(i)},N^{(i)})" title="J = \sum_{i=1}^{m} = L(A^{(i)},P^{(i)},N^{(i)})" />

**Note**: We need atleast more than 1 picture of a person as we need a pair of ```A``` and ```P``` in order to train our NN.

![image](https://user-images.githubusercontent.com/59663734/142739186-12eab359-fe42-458f-99cd-662dbba9a34e.png)
<p align="center">
  Fig. The anchor(in orange) pulls images of the same person closer and pushes images of a different person further away.
  </p>

To summarise:

1. We randomly select an ```anchor``` image(orange border).
2. We randomly select  an image of the same person as the anchor image - ```positive```(green border).
3. We randomly select  an image of a different person as the anchor image - ```negative```(red border).
4. We train our model and adjust parameters so that the positive image is closest to the anchor and the negative one is far from the anchor. 
5. We repeat the process above so that all images of the same person are close to each other and further from the others.

The diagram above shows the steps described.

**Note:** One of the problem when we choose A,P and N randomly then the conditon <img src="https://latex.codecogs.com/svg.image?\left\|f(A)-f(P)&space;\right\|^2&space;-&space;\left\|f(A)-f(N)&space;\right\|^2&space;&plus;&space;\alpha&space;\leq&space;&space;0" title="\left\|f(A)-f(P) \right\|^2 - \left\|f(A)-f(N) \right\|^2 + \alpha \leq 0" /> is easily satisfied and the NN will not learn much from it. What we want is to choose triplets that are **hard** to train on. That is in order to satisfy this condition: <img src="https://latex.codecogs.com/png.image?\dpi{100}&space;d(A,P)&space;&plus;&space;\alpha&space;\leq&space;d(A,N)" title="d(A,P) + \alpha \leq d(A,N)" />, we want <img src="https://latex.codecogs.com/png.image?\dpi{100}&space;d(A,P)&space;\approx&space;d(A,N)" title="d(A,P) \approx d(A,N)" />. Now the NN will try hard to push d(A,N) and push d(A,P) up so that there is atleast a margin <img src="https://latex.codecogs.com/png.image?\dpi{100}&space;\alpha&space;" title="\alpha " /> between the two components. Thus it is important that is is only by choosing hard triplets that our gradient descent will really do some want in learning the similarity and differences in the images. 

At RT Knits we have 2000 employees and we assume we will have 20,000 images(10 pictures of each employee), then we need need to take these 20K pictures and generate triplets of ```(A,P,N)``` and then train our learning algorithm by using gradient descent to minimize the cost function defined above. This will have the effect of backpropagating to all the parameters in the NN in order to learn an encoding such that <img src="https://latex.codecogs.com/svg.image?d(x^{(i)},x^{(j)})" title="d(x^{(i)},x^{(j)})" /> is small for images of the same person and big for images of different person. 

### 1.6 Face Verification with Binary Classification
Another option to the Triplet Loss Function is to to take the Siamese Network and have them compute the 128D embedding to be then fed to a logistic regression unit to make prediction.

- Same person: <img src="https://latex.codecogs.com/svg.image?\hat{y}&space;=&space;1" title="\hat{y} = 1" />
- Different person: <img src="https://latex.codecogs.com/svg.image?\hat{y}&space;=&space;0" title="\hat{y} = 0" />

The output <img src="https://latex.codecogs.com/svg.image?\hat{y}" title="\hat{y}" /> will be a ```sigmoid function``` applied to difference between the two set of encodings. The formula below computes the element-wise differenece in absolute values between the two encodings:

<img src="https://latex.codecogs.com/svg.image?\hat{y}&space;=&space;\sigma&space;(\sum_{k=1}^{128}w_{i}\left|f(x^{(i)})_{k}&space;-&space;f(x^{(j)})_{k}&space;\right|&space;&plus;&space;b&space;)" title="\hat{y} = \sigma (\sum_{k=1}^{128}w_{i}\left|f(x^{(i)})_{k} - f(x^{(j)})_{k} \right| + b )" />

![image](https://user-images.githubusercontent.com/59663734/142739837-07e261de-9201-4eb4-857e-1c532f6138da.png)

In summary, we just need to create a training set of pairs of images where ```target label = 1``` of **same** person and ```target label = 0``` of **different** person.

## Phase 2: Mask Detection

## Phase 3: Face Recognition with Mask
![2-Figure1-1](https://user-images.githubusercontent.com/59663734/142723133-243c6b53-47ea-43e7-809b-c4dd790aa98f.png)

### 3.1 FaceNet
FaceNet is a deep neural network used for extracting features from an image of a person’s face. It was developed in 2015 by three researchers at Google: Florian Schroff, Dmitry Kalenichenko, and James Philbin.

![1-s2 0-S0925231220316945-gr3](https://user-images.githubusercontent.com/59663734/142723211-05e51b72-8794-442e-b1fa-ae9f5a6ed9bc.jpg)

### 3.2 Resnet Network
<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/142753156-d5bb6d19-ab56-42f2-9522-d06ac374dd66.png" />
 </p>
<p align="center">
  Fig. The Resnet architecture with the Resnet Block.
  </p>
  
We start by building our Resnet Block which will be duplicated ```4``` times in the whole architecture. In our function ```resnet_block```, we define a kernel size of  ```3 x 3```, ```32``` filters, with padding = ```"same"```, a ```l2``` regularizer and a ```relu``` activation function.
  
  
```
    #----models
    def resnet_block(self,input_x, k_size=3,filters=32):
        net = tf.layers.conv2d(
            inputs=input_x,
            filters = filters,
            kernel_size=[k_size,k_size],
            kernel_regularizer=tf.keras.regularizers.l2(0.1),
            padding="same",
            activation=tf.nn.relu
        )
        net = tf.layers.conv2d(
            inputs=net,
            filters=filters,
            kernel_size=[k_size, k_size],
            kernel_regularizer=tf.keras.regularizers.l2(0.1),
            padding="same",
            activation=tf.nn.relu
        )

        net_1 = tf.layers.conv2d(
            inputs=input_x,
            filters=filters,
            kernel_size=[k_size, k_size],
            kernel_regularizer=tf.keras.regularizers.l2(0.1),
            padding="same",
            activation=tf.nn.relu
        )

        add = tf.add(net,net_1)

        add_result = tf.nn.relu(add)

        return add_result
```

Next, we define a function ```simple_resnet``` where we will design the whole architecture. We coede the first Resnet block and the max pooling layer:

```
        net = self.resnet_block(tf_input,k_size=3,filters=16)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[2,2], strides=2)
        print("pool_1 shape:",net.shape)
```

We duplicate the above code and increase the number of filters as we go deeper:

```
        net = self.resnet_block(tf_input,k_size=3,filters=16)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[2,2], strides=2)
        print("pool_1 shape:",net.shape)

        net = self.resnet_block(net, k_size=3, filters=32)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
        print("pool_2 shape:", net.shape)

        net = self.resnet_block(net, k_size=3, filters=48)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
        print("pool_3 shape:", net.shape)

        net = self.resnet_block(net, k_size=3, filters=64)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
        print("pool_4 shape:", net.shape)
```

We then flatten our layer:

```
        #----flatten
        net = tf.layers.flatten(net)
        print("flatten shape:",net.shape)
```

We feed into into a fully connected layer with dropout and units = ```128``` which represent the ```encoding```.

```
        #----dropout
        net = tf.nn.dropout(net,keep_prob=tf_keep_prob)

        #----FC
        net = tf.layers.dense(inputs=net,units=128,activation=tf.nn.relu)
        print("FC shape:",net.shape)

        #----output
        output = tf.layers.dense(inputs=net,units=class_num,activation=None)
        print("output shape:",output.shape)
```

### 3.3 Inception Network
When designing a layer for a convNet, we need to pick the type of filters we want: ```1x1```, ```3x3``` or ```5x5``` or even the type of pooling. To get rid of this conundrum, the inception layer allowsus to implement them all. So why do we use filters of different size? For our example, our image will be of the same dimension but the **target** in the image may be of different size, i.e, a person may stand far from the camera or one may be close to it. Having different kernel size allow us to extract features of different size. 

We can start by understanding the Naive version of the Inception model where we apply different types of kernel on an input and concatenate the output as shown below. The idea is instead of us selecting the filter sizes, we use them all and concatanate their output and let the NN learn whichever combination of filter sizes it wants. However, the problem with this method is the **computational cost**. 

![image](https://user-images.githubusercontent.com/59663734/142761227-875b8713-1edb-4058-a6c6-7f396d8cce1e.png)

### 3.3.1 Network in Network

If we look at the computational cost of the ```5x5``` filters of the ```28x28x192``` input volume, we have a whopping ```120M``` multiplies to perform. It is important to remember that this is only for the ```5x5``` filter and we still need to computer for the other 2 filters and pooling layer. A soltution to this is to implement a ```1x1``` convolution before the ```5x5``` filter that will output the same ```28x28x32``` volume but will reduce the number of multiplies by one tenth.

![image](https://user-images.githubusercontent.com/59663734/142761522-9a60199a-2044-4e26-b975-aff7e6da3a81.png)

How does this work?
A ```1x1``` convolution also called a ```Network in network``` will take the element-wise product between the 192 numbers(example above) in the input and the 192 numbers in the filter and apply a relu activation function and output a single number. We will have a number of filters so the output will be ```HxWx#filters```.

If we want to reduce the height and width of an input then we can use pooling to do so, however, if we want to reduce the number of channels of an input(192) then we use a ```1x1x#channels``` filter with the numbers of filters equal to the number of channels we want to output. In the example above in the middle sectiom, we want the channel to be 16 so we use 16 filters. 

We create a bottle neck by shrinking the number of channels from 192 to 16 and then increasing it again to 32. This allow us to diminish dramatically the computational cost which is now about ```12.4M``` multiplies.  The ```1x1``` convolution is an important building block in the inception network which allow us to go deeper into the network by maintaining the computational cost and learn more features.


### 3.3.2 Inception with Dimension Reduction
To reduce our computational cost we should modify out architecture in fig 3.1 and add 1x1 convoltution to it. As shown above, the 1x1 filters will allow us to have fewer weights therefore fewer calculations and therefore faster inference. The figure below shows one Inception module. The Inception network just puts a lot of these modules together.

![image](https://user-images.githubusercontent.com/59663734/142762642-b684146a-28c7-4c16-b7ea-26d6a67d8b18.png)

We have ```9``` of the inception block concatanate to each other with some additional max pooling to change the dimension. We should note that the last layer is a fully connected layer followed by a softmax layer to make prediction but we also have two side branches comming from the hidden layers trying to make prediction with a softmax output. This help ensure that the features computed in the hidden layers are also good to make accurate predictions and this help the network from overfitting. 

## Conclusion

## References
1. https://arsfutura.com/magazine/face-recognition-with-facenet-and-mtcnn/
2. https://www.youtube.com/watch?v=0NSLgoEtdnw&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF&index=36
3. https://www.youtube.com/watch?v=-FfMVnwXrZ0&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF&index=32
4. https://www.youtube.com/watch?v=96b_weTZb2w&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF&index=33
5. https://www.youtube.com/watch?v=6jfw8MuKwpI&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF&index=34
6. https://www.youtube.com/watch?v=d2XB5-tuCWU&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF&index=35
