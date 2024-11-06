<p align="center"><img src="assets/vision.png" alt="Vision Icon"></p>
<h1 align="center">COMP9517 Group Project</h1>

<p align="center">The goal of this group project is to develop and compare different computer vision methods
for segmenting sea turtles from photographs. More specifically, the task is to segment the
head, flippers, and the carapace of each turtle.</p>

---

## Documentation
> Mostly going to use this to write notes for what we can use.

### Pre-processing
> Most studied methods have an elaborate pre-processing method in place to help with assisting the accuracy of the algorithm. Could use the more traditional segmentation methods to assist with seperating unneeded picture information with those we can perform predictions on.

- **Thresholding**
    - **Selective Search** - While it is for **object detection**, the **pre-processing** method is interesting. It starts by over-segmenting the image based on intensity of the pixels using a graph-based segmentation (Felzenszwalb and Huttenlocher).
- **Edge Segmentation**
- **Clustering-based Segmentation**

### Semantic Segmentation
> Could be useful in this particular problem **given** no two body parts of the same class overlap each other in the turtle image dataset. This could reduce the computational requirements of the algorithm.

### Instance Segmentation
> If the former is not an option, we may have to look into this.

---

## References
> References are in IEEE format for when we transfer into report.

[1] A. Hynes and S. Czarnuch, “Human Part Segmentation in Depth Images with Annotated Part Positions,” Sensors, vol. 18, no. 6, p. 1900, Jun. 2018, doi: https://doi.org/10.3390/s18061900.

[2] L. Bonaldi, A. Pretto, C. Pirri, F. Uccheddu, C. G. Fontanella, and C. Stecco, “Deep Learning-Based Medical Images Segmentation of Musculoskeletal Anatomical Structures: A Survey of Bottlenecks and Strategies,” Bioengineering, vol. 10, no. 2, pp. 137–137, Jan. 2023, doi: https://doi.org/10.3390/bioengineering10020137.

[3] “A Gentle Introduction to Image Segmentation for Machine Learning,” www.v7labs.com. https://www.v7labs.com/blog/image-segmentation-guide

[4] A. Sharma, “Image Segmentation with U-Net in PyTorch: The Grand Finale of the Autoencoder Series,” PyImageSearch, Nov. 06, 2023. https://pyimagesearch.com/2023/11/06/image-segmentation-with-u-net-in-pytorch-the-grand-finale-of-the-autoencoder-series/

[5] V. Kookna, “Semantic vs. Instance vs. Panoptic Segmentation,” PyImageSearch, Jun. 29, 2022. https://pyimagesearch.com/2022/06/29/semantic-vs-instance-vs-panoptic-segmentation/

[6] A. Rosebrock, “Mask R-CNN with OpenCV,” PyImageSearch, Nov. 19, 2018. https://pyimagesearch.com/2018/11/19/mask-r-cnn-with-opencv/

‌---

<p align="right"><a target="_blank" href="https://icons8.com/icon/g5JjVIjdQ1uC/visionn">Vision</a> icon by <a target="_blank" href="https://icons8.com">Icons8</a></p>
