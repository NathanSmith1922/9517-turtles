<p align="center"><img src="assets/vision.png" alt="Vision Icon"></p>
<h1 align="center">COMP9517 Group Project</h1>

<p align="center">The goal of this group project is to develop and compare different computer vision methods
for segmenting sea turtles from photographs. More specifically, the task is to segment the
head, flippers, and the carapace of each turtle.</p>

---

## Documentation

**Mostly going to use this to write notes for what we can use.**

### Pre-processing
Most studied methods have an elaborate pre-processing method in place to help with assisting the accuracy of the algorithm. Could use the more traditional segmentation methods to assist with seperating unneeded picture information with those we can perform predictions on.

- **Thresholding**
    - **Selective Search** - While it is for **object detection**, the **pre-processing** method is interesting. It starts by over-segmenting the image based on intensity of the pixels using a graph-based segmentation (Felzenszwalb and Huttenlocher).
- **Edge Segmentation**
- **Clustering-based Segmentation**

### Semantic Segmentation
Could be useful in this particular problem **given** no two body parts of the same class overlap each other in the turtle image dataset. This could reduce the computational requirements of the algorithm.

### Instance Segmentation
If the former is not an option, we may have to look into this.

---

<p align="right"><a target="_blank" href="https://icons8.com/icon/g5JjVIjdQ1uC/visionn">Vision</a> icon by <a target="_blank" href="https://icons8.com">Icons8</a></p>
