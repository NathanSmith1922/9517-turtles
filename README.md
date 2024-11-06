<p align="center"><img src="assets/vision.png" alt="Vision Icon"></p>
<h1 align="center">COMP9517 Group Project</h1>

<p align="center">The goal of this group project is to develop and compare different computer vision methods
for segmenting sea turtles from photographs. More specifically, the task is to segment the
head, flippers, and the carapace of each turtle.</p>

---

## Documentation

**Mostly going to use this to write notes for what we can use.**

### Object Detection

#### Selective Search 
While it is for **object detection**, the pre-processing method is interesting. It starts by over-segmenting the image based on intensity of the pixels using a graph-based segmentation (Felzenszwalb and Huttenlocher).

Add all bounding boxes corresponding to segmented parts to the list of region proposals then groups adjacent segments based on similarity.

### Semantic Segmentation

#### Sliding Window Approach
Extract “patches” from entire image, classify centre pixel based on the neighbouring context.

#### Convolution
Design a network having convolutional layers, with downsampling and upsampling inside the network (learning in an end-to-end manner)

### Instance Segmentation

#### Mask R-CNN

---

<p align="right"><a target="_blank" href="https://icons8.com/icon/g5JjVIjdQ1uC/visionn">Vision</a> icon by <a target="_blank" href="https://icons8.com">Icons8</a></p>
