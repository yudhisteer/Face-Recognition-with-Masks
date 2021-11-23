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

### 3.5 Face Detection
Object detection refers to the task of identifying various objects within an image and drawing a bounding box around each of them. Initially researchers developed R-CNN for object detection, localization and classification. The output is a bounding box surrounding the object detected with the classification result of the object. With time, we improved the R-CNN network and came up with Fast R-CNN and Faster R-CNN. However, one major drawback of the network was that the inference time was too long for real-time object detection. New architectures such as ```YOLO``` and the ones describe below are better suited for real-time object detection.

There are several methods for face detection:
- SSD
- MTCNN
- Dlib
- OpenCV

Our goal is to use a face detection algorithm to detect faces and crop it with margin 20 or 40 as shown below. 

![image](https://user-images.githubusercontent.com/59663734/142823899-d1193e71-a01a-4844-9810-6185488384d5.png)

#### 3.5.1 Face Detection with SSD

In the first phase I used Dlib but now I will use SSD for face detection. While MTCNN is more widely used, SSD performs faster inference however has low accuracy. SSD uses lower resolution layers to detect larger scale objects. It speeds up the process by eliminating the need for the region proposal network.

The architecture of SSD consists of 3 main components:

1. Base network - VGG-16
2. Extra feature layers
3. Prediction layers

Before we dive in into SSD, it is important we understand our Object Localisation and Object Detection works.

**1. How does it work?**

By starting simple, suppose we want to perform object localisation whereby if we detect a person's face in a picture we want to know where in the picture is the face located. We may also have other objects in the picture, for example a car, we will also need to take this into consideration. Now our CNN softmax output will not just contain the object class label but also the parameters below:

![CodeCogsEqn](https://user-images.githubusercontent.com/59663734/142835042-35a3de01-1d18-4d8b-9c7f-e3d12080960d.png) where

- p: "0" if no object and "1" if object
- c1: "1" if face, "0" otherwise
- c2: "1" if car, "0" otherwise
- x: x-position - center point of object
- y: y-position - center point of object
- w: width of bounding box
- h: height of bounding box
 
If we have a picture as shown below we may divide it into a ```5x5``` grid cells and predict what value will our ```y``` predict in each individual cell.

![image](https://user-images.githubusercontent.com/59663734/142837554-05605efa-da36-480f-a710-f013153da318.png)

For the first grid cell which does not contain an object, we have p = 0 as the first parameter in out y value and for the rest we don't care so we use ? to represent as placeholder. For the thrid grid cell, we detect an object and a face so out p = 1 and c1 = 1 and the x,y,w and h represents values for the bounding box. During training we will try to make our network output similar vectors. 

SSD does not use a pre-defined region proposal network. Instead, it computes both the location and class scores using small convolution filters. After extracting the feature maps, SSD applies 3 × 3 convolution filters for each cell to make predictions.

**2. How to know our bounding box is correct?**

```IoU``` is used to measure the overlap between two bounding boxes. 

![image](https://user-images.githubusercontent.com/59663734/142840525-4db003b6-9a6a-4b08-9ee7-8002216cdc2b.png)

Normally if our IoU is greater than or equal to 0.5 we deem it to be a correct prediction. But we can be more stringent and increase the threshold where 1 is the maximum value.

**3. Anchor Boxes**

1. It is not possible for one object to be strictly within one grid cell. And when it is not, how do we determine which cell do we asscoiate to the object.  
- The solution for this is to associate the cell which contains the **center point** of the bounding box of the object. 

2. Each of the grid cell can detect only one object. But we may have one grid cell containing more than one object. How do we handle multiple center points?
- We can use a bigger grid - 19x19 - instead of a 5x5 which reduces this problem. Also, we need to do is predefined anchor boxes and associate perdiction with the anchor boxes. 

Previously, each obejct is assigned to a grid cell which contains that object's midpoint. Now, each obejct is assigned to a grid cell which contains that object's midpoint **and** anchor box for the grid cell with the highest IoU(similar shape).  

![image](https://user-images.githubusercontent.com/59663734/142845336-85efa649-4611-47d9-8885-f8660a13ad8f.png)

For the image above, both objects have their ceneterpoint in the same cell. So we set a tall anchor box which can be used to predict a standing person and a wide anchor box can be used to predict a car. We use these anchor boxes in each of the grid cell and output one vector y for every anchor box. 

**4. Non-Maximal Suppression (NMS)**

SSD contains 8732 default boxes. During inference, we have 8732 boxes for each class (because we output a confidence score for each box). Most of these boxes are negative and among the positive ones, there would be a lot of overlapping boxes.  Non-Maximal Suppression (NMS) is applied to get rid of overlapping boxes per class. It works as such: 

1. sort the boxes based on the confidence score
2. pick the box with the largest confidence score
3. remove all the other predicted boxes with Jaccard overlap > the NMS threshold (0.45 here)
4. repeat the process until all boxes are covered.

![image](https://user-images.githubusercontent.com/59663734/142847066-bb095fbe-e264-4029-a348-f04fdabd2978.png)

To sum up:
- The network is very sensitive to default boxes and it is important to choose the default boxes based on the dataset that it is being used on.
- SSD does not work well with small objects: earlier layers which have smaller receptive field and are responsible for small object detection, are too shallow. 

**5. Implementation**

I used a trained facemask detection algorithm to crop the pictures. Similar to the one explained above, I adjusted the bounding box to crop the images:

![image](https://user-images.githubusercontent.com/59663734/142854038-95faeefb-a400-4c99-ba09-b92d44424c7b.png)

I loaded the ```.pb``` file with a default margin of 44 and GPU-ration of 0.1:

```
    def __init__(self,pb_path,margin=44,GPU_ratio=0.1):
        # ----var
        node_dict = {'input': 'data_1:0',
                     'detection_bboxes': 'loc_branch_concat_1/concat:0',
                     'detection_scores': 'cls_branch_concat_1/concat:0'}
```

We restore the model to get the nodes and get the input shape which is used to resize the images:

```
        # ====model restore from pb file
        sess, tf_dict = model_restore_from_pb(pb_path, node_dict,GPU_ratio = GPU_ratio)
        tf_input = tf_dict['input']
        model_shape = tf_input.shape  # [N,H,W,C]
        print("model_shape = ", model_shape)
        img_size = (tf_input.shape[2].value,tf_input.shape[1].value)
        detection_bboxes = tf_dict['detection_bboxes']
        detection_scores = tf_dict['detection_scores']
```

In the ```inference``` function we use sess.run to get the detection boxes, the detection scores:

```
        y_bboxes_output, y_cls_output = self.sess.run([self.detection_bboxes, self.detection_scores],
                                                      feed_dict={self.tf_input: img_4d})
```

We then need to decode the bounding boxes and do the non-max suppression:

```
        # remove the batch dimension, for batch is always 1 for inference.
        y_bboxes = self.decode_bbox(self.anchors_exp, y_bboxes_output)[0]
        y_cls = y_cls_output[0]
        # To speed up, do single class NMS, not multiple classes NMS.
        bbox_max_scores = np.max(y_cls, axis=1)
        bbox_max_score_classes = np.argmax(y_cls, axis=1)

        # keep_idx is the alive bounding box after nms.
        keep_idxs = self.single_class_non_max_suppression(y_bboxes, bbox_max_scores,  conf_thresh=self.conf_thresh,
                                                          iou_thresh=self.iou_thresh )
        # ====draw bounding box
```

In order to draw the bounding box we need to get the (xmin,ymin) and (xmax,ymax) coordinates. Our bounding boxes are unit coordinates in the range [0,1]. We have to make them to real sizes. We need to get the width of the bounding box using (xmax-xmin) and height using (ymax-ymin).

```
        # ====draw bounding box
        for idx in keep_idxs:
            conf = float(bbox_max_scores[idx])
            #print("conf = ",conf)
            class_id = bbox_max_score_classes[idx]
            bbox = y_bboxes[idx]
            #print(bbox)

            xmin = np.maximum(0, int(bbox[0] * ori_width - self.margin / 2))
            ymin = np.maximum(0, int(bbox[1] * ori_height - self.margin / 2))
            xmax = np.minimum(int(bbox[2] * ori_width + self.margin / 2), ori_width)
            ymax = np.minimum(int(bbox[3] * ori_height + self.margin / 2), ori_height)

            re_boxes.append([xmin, ymin, xmax - xmin, ymax - ymin])
            re_confidence.append(conf)
            re_classes.append('face')
            re_mask_id.append(class_id)
        return re_boxes, re_confidence, re_classes, re_mask_id
```

Our original image is 250x250 so we want our cropped face image to be over 100x100. This enable us to detect faces which is well aligned(the main person in the image). We use two thresholds for the width and height:

```
    width_threshold = 100 + margin // 2 #allow us to get a full face and not cropped ones
    height_threshold = 100 + margin // 2
```

In an ```if``` condition we check if the width of our bbox is more than the width_threshold and the height is more than the height_threshold then the height of the cropped image  is ```bbox[1]:bbox[1] + bbox[3]``` and width is ```bbox[0]:bbox[0] + bbox[2]```. We then save the file. 

```
            for num,bbox in enumerate(bboxes):
                if bbox[2] > width_threshold and bbox[3] > height_threshold:
                    img_crop = img_ori[bbox[1]:bbox[1] + bbox[3],bbox[0]:bbox[0] + bbox[2], :]
                    save_path = os.path.join(save_dir,str(idx) + '_' + str(num) + ".png")
                    # print("save_path:",save_path)
                    cv2.imwrite(save_path,img_crop)
```

We display the images:

![image](https://user-images.githubusercontent.com/59663734/142858127-52759a18-4b40-4447-9426-1cc8fe851f50.png)


### 3.2 Data Cleaning
After cropping the pictures, we check the folders and we see that in folder ```0000157``` we got one mislabelled image as shown below. This signifies that the CASIA dataset is not a cleaned dataset and there may be other instances of mislabelled images. We cannot check each of the ```10,575``` folders individually so we need an algorithm that will do this for us. 

![image](https://user-images.githubusercontent.com/59663734/142935195-d12ee28e-7dc3-4f2f-8b91-538c3919e5fb.png)

We can set a process of removing mislabelled images using the distance function ```d``` described before:

1. In a subfolder in the main directory, we select one image one by one as the **target image** and the other images become the **reference images**.
2. We calculate the average distances between the target image and the reference image. 
3. We see that the average distance, when a correct image is selected as the target image, is not much as compared when the mislabelled image is selected as the target image. Also we might have more than one mislabelled image in a folder. That is the reason why we make each image the target image and calculate the average distance.
**Note:** The distance between a target image and itself is zero. 
5. We compare the average distances to a threshold.
6. We remove the target image(mislabelled image) when its average distance exceeds the threshold.

![image](https://user-images.githubusercontent.com/59663734/142935906-405ce329-48d7-43ab-8230-341271216ccc.png)

We use the pretrained weights of Inception Resnet V1 trained on VGGFace dataset and has an accuracy of 0.9965 on LFW dataset. We start by restoring the ```.pb``` file and create a fucntion ```img_removal_by_embed``` to do the following processes:

**1. Collect all folders:**

```
    # ----collect all folders
    dirs = [obj.path for obj in os.scandir(root_dir) if obj.is_dir()]
    if len(dirs) == 0:
        print("No sub-dirs in ", root_dir)
    else:
        #----dataset range
        if dataset_range is not None:
            dirs = dirs[dataset_range[0]:dataset_range[1]]
```

**2. Initialize our model:**

```
        # ----model init
        sess, tf_dict = model_restore_from_pb(pb_path, node_dict, GPU_ratio=GPU_ratio)
        tf_input = tf_dict['input']
        tf_embeddings = tf_dict['embeddings']
```

**3. Set the method to calculate the distance d:**

```
        # ----tf setting for calculating distance
        with tf.Graph().as_default():
            tf_tar = tf.placeholder(dtype=tf.float32, shape=tf_embeddings.shape[-1])
            tf_ref = tf.placeholder(dtype=tf.float32, shape=tf_embeddings.shape)
            tf_dis = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(tf_ref, tf_tar)), axis=1))
            # ----GPU setting
            config = tf.ConfigProto(log_device_placement=True,
                                    allow_soft_placement=True,
                                    )
            config.gpu_options.allow_growth = True
            sess_cal = tf.Session(config=config)
            sess_cal.run(tf.global_variables_initializer())
```

**4. Process each folder and create subfolders to move the mislabelled images:**

```
        #----process each folder
        for dir_path in dirs:
            paths = [file.path for file in os.scandir(dir_path) if file.name.split(".")[-1] in img_format]
            len_path = len(paths)
            if len_path == 0:
                print("No images in ",dir_path)
            else:
                # ----create the sub folder in the output folder
                save_dir = os.path.join(output_dir, dir_path.split("\\")[-1])
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
```

**5. Calculate the embeddings:**

```
                # ----calculate embeddings
                ites = math.ceil(len_path / batch_size)
                embeddings = np.zeros([len_path, tf_embeddings.shape[-1]], dtype=np.float32)
                for idx in range(ites):
                    num_start = idx * batch_size
                    num_end = np.minimum(num_start + batch_size, len_path)
```

**6. Calcuate the average distance using the embeddings:**

```
                # ----calculate avg distance of each image
                feed_dict_2 = {tf_ref: embeddings}
                ave_dis = np.zeros(embeddings.shape[0], dtype=np.float32)
                for idx, embedding in enumerate(embeddings):
                    feed_dict_2[tf_tar] = embedding
                    distance = sess_cal.run(tf_dis, feed_dict=feed_dict_2)
                    ave_dis[idx] = np.sum(distance) / (embeddings.shape[0] - 1)
```

**7. Remove the mislabelled images if average distance greater than threshold(1.25):**

```
                # ----remove or copy images
                for idx,path in enumerate(paths):
                    if ave_dis[idx] > threshold:
                        print("path:{}, ave_distance:{}".format(path,ave_dis[idx]))
                        if type == "copy":
                            save_path = os.path.join(save_dir,path.split("\\")[-1])
                            shutil.copy(path,save_path)
                        elif type == "move":
                            save_path = os.path.join(save_dir,path.split("\\")[-1])
                            shutil.move(path,save_path)
```

We run the file and check the folders. We got ```3981``` folders which had ```20079```wrong images in total. In the folders, we see that our algorithm correctly identified the wrong person in Linda Hamilton's folder and in Bill Murray's folder we had more than one mislabelled images. However, we see that the algorithm also removed the images of the correct label. Mainly because the images were blurred or fuzzy, or the subjet had sunglasses in them or there were pictures when the person was too young or too old. Nevertheless, data cleaning will now allow our NN to train on a more accurate dataset to make better predictions.

![image](https://user-images.githubusercontent.com/59663734/142937113-3235d10a-9a9b-47fe-b4d8-c5d4c6e776de.png)



### 3.3 Custom Face Mask Dataset
Our end goal is to be able to recognize faces with mask. The CASIA dataset already have half a million of pictures of faces and we know that by using the Inception Resnet V1 model we can create a face recognition model. What we want to do now is have the same CASIA dataset with the same folders and same pictures but with the persons wearing a mask. We don't have such as dataset so we need to ceate one. What we want to do is to show our AI the picture of a person **without** a mask, then a picture of the same person **with** a mask and tell him that it is the same person. 

![image](https://user-images.githubusercontent.com/59663734/142997230-bc4e6fb0-4122-4cdd-8e2f-8a7eb61b3a4b.png)

In order to achieve the process above, we need to have our mask in ```png``` format. PNG formats had 4 channels. The fourth channel is used to describe the transparency. I will use the Dlib library which is a pre-trained to recognize 68 landmark points that cover the jaw, chin, eyebrows, nose, eyes, and lips of a face. The numbers 48 to 68 are those for the mouth as shown below.

![image](https://user-images.githubusercontent.com/59663734/143016227-3df5cba5-8a75-4c4f-8556-28ebc819e8ad.png)


We start by creating a function ```detect_mouth``` which we will use to read the face landmarks from 48 to 68 and calculate the coordinates:

```
            #----get the mouth part
            for i in range(48, 68):
                x.append(landmark.part(i).x)
                y.append(landmark.part(i).y)

            y_max = np.minimum(max(y) + height // 3, img_rgb.shape[0])
            y_min = np.maximum(min(y) - height // 3, 0)
            x_max = np.minimum(max(x) + width // 3, img_rgb.shape[1])
            x_min = np.maximum(min(x) - width // 3, 0)

            size = ((x_max-x_min),(y_max-y_min))#(width,height)
```


In another function ```mask_wearing``` we first process the folders and create directories for each folder in the main folder. Then we randomly select a PNG mask image from the folder:

```
                        if size is not None: #there is a face
                            # ----random selection of face mask
                            which = random.randint(0, len_mask - 1)
                            item_name = mask_files[which]
```

We read the image with ```cv2.IMREAD_UNCHANGED``` to make sure the image has 4 channels. We resize it based on our mouth detection coordinates found before. We use ```cv2.threshold``` to make the values of the image of the mask ```0```(black) or ```255```(white), i.e, we have the image of a white mask with black background. We use ```cv2.bitwise_and``` to create the mask of the face mask:

```
                            # ----face mask process
                            item_img = cv2.imread(item_name, cv2.IMREAD_UNCHANGED)
                            item_img = cv2.resize(item_img, size)
                            item_img_bgr = item_img[:, :, :3]
                            item_alpha_ch = item_img[:, :, 3]
                            _, item_mask = cv2.threshold(item_alpha_ch, 220, 255, cv2.THRESH_BINARY)
                            img_item = cv2.bitwise_and(item_img_bgr, item_img_bgr, mask=item_mask)
```

We declare the coordinates of our Region of Interest(ROI) from the mouth detection values. We create an invert mask with ```cv2.bitwise_not``` of the face mask and then use ```cv2.bitwise_and``` to mask the face mask onto the person's face:

```
                            # ----mouth part process
                            roi = img[y_min:y_min + size[1], x_min:x_min + size[0]]
                            item_mask_inv = cv2.bitwise_not(item_mask)
                            roi = cv2.bitwise_and(roi, roi, mask=item_mask_inv)
```

We then add the two images: face mask and face:

```
                            # ----addition of mouth and face mask
                            dst = cv2.add(roi, img_item)
                            img[y_min: y_min + size[1], x_min:x_min + size[0]] = dst
```










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

#### 3.3.1 Network in Network

If we look at the computational cost of the ```5x5``` filters of the ```28x28x192``` input volume, we have a whopping ```120M``` multiplies to perform. It is important to remember that this is only for the ```5x5``` filter and we still need to computer for the other 2 filters and pooling layer. A soltution to this is to implement a ```1x1``` convolution before the ```5x5``` filter that will output the same ```28x28x32``` volume but will reduce the number of multiplies by one tenth.

![image](https://user-images.githubusercontent.com/59663734/142761522-9a60199a-2044-4e26-b975-aff7e6da3a81.png)

How does this work?
A ```1x1``` convolution also called a ```Network in network``` will take the element-wise product between the 192 numbers(example above) in the input and the 192 numbers in the filter and apply a relu activation function and output a single number. We will have a number of filters so the output will be ```HxWx#filters```.

If we want to reduce the height and width of an input then we can use pooling to do so, however, if we want to reduce the number of channels of an input(192) then we use a ```1x1x#channels``` filter with the numbers of filters equal to the number of channels we want to output. In the example above in the middle sectiom, we want the channel to be 16 so we use 16 filters. 

We create a bottle neck by shrinking the number of channels from 192 to 16 and then increasing it again to 32. This allow us to diminish dramatically the computational cost which is now about ```12.4M``` multiplies.  The ```1x1``` convolution is an important building block in the inception network which allow us to go deeper into the network by maintaining the computational cost and learn more features.


#### 3.3.2 Inception with Dimension Reduction
To reduce our computational cost we should modify out architecture in fig 3.1 and add 1x1 convoltution to it. As shown above, the 1x1 filters will allow us to have fewer weights therefore fewer calculations and therefore faster inference. The figure below shows one Inception module. The Inception network just puts a lot of these modules together.

![image](https://user-images.githubusercontent.com/59663734/142762642-b684146a-28c7-4c16-b7ea-26d6a67d8b18.png)

We have ```9``` of the inception block concatanate to each other with some additional max pooling to change the dimension. We should note that the last layer is a fully connected layer followed by a softmax layer to make prediction but we also have two side branches comming from the hidden layers trying to make prediction with a softmax output. This help ensure that the features computed in the hidden layers are also good to make accurate predictions and this help the network from overfitting. 

![image](https://user-images.githubusercontent.com/59663734/142762952-90a602b5-5fb7-43c7-8589-e41c02f22647.png)

### 3.4 Inception-Resnet V1 Network

![image](https://user-images.githubusercontent.com/59663734/142763781-1a990187-307c-45db-9f61-01bf89b1c861.png)


 


## Conclusion

## References
1. https://arsfutura.com/magazine/face-recognition-with-facenet-and-mtcnn/
2. https://www.youtube.com/watch?v=0NSLgoEtdnw&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF&index=36
3. https://www.youtube.com/watch?v=-FfMVnwXrZ0&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF&index=32
4. https://www.youtube.com/watch?v=96b_weTZb2w&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF&index=33
5. https://www.youtube.com/watch?v=6jfw8MuKwpI&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF&index=34
6. https://www.youtube.com/watch?v=d2XB5-tuCWU&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF&index=35
7. https://www.aiuai.cn/aifarm465.html
8. https://jonathan-hui.medium.com/ssd-object-detection-single-shot-multibox-detector-for-real-time-processing-9bd8deac0e06
9. https://medium.com/inveterate-learner/real-time-object-detection-part-1-understanding-ssd-65797a5e675b
