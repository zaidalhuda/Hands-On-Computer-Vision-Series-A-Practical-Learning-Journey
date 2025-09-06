<!-- Hero -->
<h1 align="center">🔬 Hands-On Computer Vision Series - A Practical Learning Journey</h1>

<p align="center">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.8%2B-blue.svg">
  <a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-Latest-red.svg"></a>
  <a href="https://tensorflow.org/"><img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-Latest-orange.svg"></a>
  <img alt="Status" src="https://img.shields.io/badge/Status-25_Practicals-green.svg">
  <img alt="Level" src="https://img.shields.io/badge/Level-Undergraduate_Friendly-brightgreen.svg">
  <a href="https://colab.research.google.com/github/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME">
    <img alt="Open in Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
  </a>
</p>

<!-- Enhanced welcome section -->
Welcome to the **Hands-On Computer Vision Series** from **Stirling College, Chengdu University** - a comprehensive, **undergraduate-friendly** guide that teaches computer vision through practical hands-on learning.

🎓 **Perfect for Undergraduates & Beginners**: These practicals are specifically designed for students who are just starting to learn computer vision. Each practical provides clear explanations and step-by-step guidance. The difficulty gradually increases, building your expertise from basic concepts to advanced applications.

This repository provides clean, beginner-friendly **Jupyter notebooks** that walk you through key machine learning and computer vision concepts using real datasets and visual feedback. Whether you're working on your graduation project or just starting your CV journey, these practicals will give you solid foundations.

Each notebook focuses on a single core concept - from building your first neural network to progressively enhancing it with modern deep learning techniques.

> *"Master computer vision through progressive hands-on examples and real-world applications"* - Build your expertise step-by-step with practical notebooks and clear explanations.

**By Dr. Zaid Al-Huda**

---

## 🎯 Understanding Computer Vision: The Big Picture

Before diving into the practicals, let's understand what Computer Vision really involves. In CV, we primarily work with **three fundamental tasks** that form the backbone of most applications:

### **1. Detection**
- **Image Classification**: Teaching models to classify images (cat vs dog, etc.)
- **Object Detection**: Locating objects in images and drawing bounding boxes around them
- **Applications**: Counting objects, direction detection, shelf monitoring, traffic analysis

### **2. Segmentation** 
- **Semantic Segmentation**: Identifying which pixels belong to which object class
- **Instance Segmentation**: Separating individual object instances
- **Panoptic Segmentation**: Combining both semantic and instance segmentation
- **Applications**: Smart photo editing, background removal, medical imaging (tumor detection), autonomous driving

### **3. Tracking**
- **Multi-Object Tracking**: Following objects across video frames
- **Challenges**: Re-identification after occlusion, handling appearance changes
- **Applications**: Surveillance, sports analysis, autonomous vehicles

### **Advanced Applications**
Built upon these foundations:
- **Image Generation** (GANs, Stable Diffusion)
- **Image Captioning** 
- **Text-to-Video**
- **3D Reconstruction**

*The key insight: Advanced CV applications are built upon mastering these fundamental tasks!*

---

## 🚀 What You'll Learn

- **Hands-on Colab notebooks** - run anywhere with minimal setup  
- **Progressive path** - foundations → modern CV → real applications  
- **Undergraduate-friendly approach** - clear explanations, visual feedback, manageable complexity
- **Clarity first** - concise cells, readable code, and visual feedback  
- **End-to-end workflow** - data → training → evaluation → (optional) deployment  
- **Modern architectures** - CNN families, ResNet/EfficientNet, ViT, YOLO, U‑Net, MOT  
- **Cohesive sequence** - each practical prepares you for the next
- **Industry-ready skills** - techniques used in real-world applications

---

## ⚡ Quick Start (Colab)
No installs. No setup. Perfect for students!

1. Click any **Open in Colab** button below.  
2. In Colab: **Runtime → Change runtime type → GPU** (if available).  
3. Run cells from top to bottom and experiment!

---

## 🗺️ Learning Roadmap - 25 Practicals

> 🚀 Click the Colab badge to run instantly (don't forget to set GPU runtime!).

| # | Practical | Open in Colab | Topics Covered |
|---:|-----------|:-------------:|----------------|
| 1 | Building a Simple Neural Network *(No Activation Functions)* | <a href="https://colab.research.google.com/drive/1jkdGNJ6cDcGEuPZSF7Mdgxh1EZTphUCU?usp=drive_link"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" height="50" width="170"></a> | Perceptron basics, forward pass, loss computation |
| 2 | Enhancing the Neural Network with Activation Functions | <a href="https://colab.research.google.com/drive/1Wotdb8Z_a8MRugs6iCDfmGO6eMqj6mhV?usp=drive_link"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" height="50" width="170"></a> | Non-linear activations, ReLU, Sigmoid, Tanh |
| 3 | Overfit Prevention (Regularization, Dropout, Early Stopping, Batch Normalization) | <a href="https://colab.research.google.com/drive/1YsxCPgjMnWtJGJSGeVDsnkWNG1vaOv7R?usp=drive_link"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" height="50" width="170"></a> | Regularization (L2), Dropout, Early stopping |
| 4 | Transfer Learning (Pretrained Embeddings, Fine-Tuning, Differential LR) | <a href="https://colab.research.google.com/drive/14qZ3fKJH1GUv_UqzNgO9_wwpjB4o3grp?usp=drive_link"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" height="50" width="170"></a> | Pretrained models, feature extraction |
| 5 | Hyperparameter Tuning with Weights & Biases (W&B) | <a href="https://colab.research.google.com/drive/1K1zuaS-kwECaEThVSDYrTLltVzYxVeJ8?usp=drive_link"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" height="50" width="170"></a> | Experiment logging, sweeps, tuning strategy |
| 6 | Evaluation Metrics for Classification, Detection, Segmentation | <a href="https://colab.research.google.com/drive/14LGtIx89s6xpFKCYIm6LazrncTClV9GL?usp=drive_link"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" height="50" width="170"></a> | Accuracy, Precision, Recall, F1, ROC/AUC, IoU |
| 7 | CNN Fundamentals & AlexNet | <a href="https://colab.research.google.com/drive/1DHmJyh-7GDG5F4H8ryGfVN2RjbMVcZM0?usp=drive_link"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" height="50" width="170"></a> | Convolutions, pooling, AlexNet architecture |
| 8 | Deep Dive into VGG-16 | <a href="https://colab.research.google.com/drive/17aVTc4IZfwvlUm7xyj-ZLL6kiE-QbMrW?usp=drive_link"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" height="50" width="170"></a> | VGG blocks, deep feature hierarchies |
| 9 | GoogLeNet (InceptionV1) | <a href="https://colab.research.google.com/drive/1_3vlBh5aDXkSPRdGkYwp9VZ8M7ArBAe1?usp=drive_link"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" height="50" width="170"></a> | Inception modules, multi-scale feature extraction |
| 10 | SqueezeNet & Fire Modules | <a href="https://colab.research.google.com/drive/1KhxXTsdB4S23TKrhf_0sYcHk0ohndTf-?usp=drive_link"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" height="50" width="170"></a> | Parameter-efficient CNNs, lightweight design |
| 11 | ResNet - Residual Learning | <a href="https://colab.research.google.com/drive/1Aw0ZHGN6FE8CNm4stbebH4Tvq2NlXOqJ?usp=drive_link"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" height="50" width="170"></a> | Skip connections, solving vanishing gradients |
| 12 | MobileNet - Depthwise Separable | <a href="https://colab.research.google.com/drive/1xUjy_brkeHh7JR4J6jb16MR-n6aQsTJU?usp=drive_link"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" height="50" width="170"></a> | Depthwise separable convolutions, efficiency |
| 13 | DenseNet - Densely Connected | <a href="https://colab.research.google.com/drive/1z0_HEgqFpfCBOLESh7F-JdDg-cupscDY?usp=drive_link"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" height="50" width="170"></a> | Dense blocks, feature reuse |
| 14 | EfficientNet - Compound Scaling | <a href="https://colab.research.google.com/drive/1QwXS56_qH1NKaNMUnibYkAZZWV5i4KhM?usp=drive_link"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" height="50" width="170"></a> | Width/depth/resolution scaling |
| 15 | Transfer Learning: Freeze vs Fine-Tune | <a href="https://colab.research.google.com/drive/1D4N2vL5BnX7pVONSoJWD0Lo01RsKxpBW?usp=drive_link"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" height="50" width="170"></a> | Layer freezing vs full fine-tuning |
| 16 | Vision Transformers (ViT) | <a href="https://colab.research.google.com/drive/1LOXeGfltYfTLaGJvwJAcmybW2fXDjgo_?usp=drive_link"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" height="50" width="170"></a> | Patch embeddings, self-attention |
| 17 | Object Detection - From R-CNN to Mask-RCNN | <a href="https://colab.research.google.com/drive/1IvjRmhAIFQWQl_pGv1oPV74bp-8B-Jpe?usp=drive_link"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" height="50" width="170"></a> | Region proposals, object detection, segmentation |
| 18 | YOLO Object Detection | <a href="https://colab.research.google.com/drive/1-K2NbGL1lD_uIUBCWBo8LohqljoIeXlI?usp=drive_link"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" height="50" width="170"></a> | Real-time detection pipeline |
| 19 | Classical Object Tracking & Counting | <a href="https://colab.research.google.com/drive/1SLXMUAto7dj6jtngpGSTHZraaZ-bl9Va?usp=drive_link"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" height="50" width="170"></a> | Optical flow, background subtraction |
| 20 | Multi-Object Tracking (YOLOv11 + DeepSORT + ByteTrack) | <a href="https://colab.research.google.com/drive/1w17oqSJ1S1-L-D7Gm5a3NOGwsd54UlL3?usp=drive_link"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" height="50" width="170"></a> | Detection-to-tracking integration |
| 21 | Classic Image Segmentation | <a href="https://colab.research.google.com/drive/1fqVKThbbLiXAMpmY56dKwhJag5xgndvP?usp=drive_link"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" height="50" width="170"></a> | Thresholding, Otsu, Watershed |
| 22 | Binary Segmentation with U-Net | <a href="https://colab.research.google.com/drive/122inN5JQFL-_tY-tFmYH-A3R7VoRTZmq?usp=drive_link"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" height="50" width="170"></a> | U-Net basics, binary masks |
| 23 | Multi-Class Segmentation | <a href="https://colab.research.google.com/drive/1xCygToxJ7UAgdgSI90QhdQdkypWw2jeW?usp=drive_link"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" height="50" width="170"></a> | Multi-class U-Net variants |
| 24 | U-Net Family Comparative Study | <a href="https://colab.research.google.com/drive/1aUxzIFtwAViXp2pVOTMmLDioVll79Cq2?usp=drive_link"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" height="50" width="170"></a> | U-Net++, Attention U-Net comparison |
| 25 | CV Studio - Multitask & Deployment | <a href="https://colab.research.google.com/drive/13rkK0y7MA6YvEm2VpT4eKMjQuCTrWi9z?usp=drive_link"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" height="50" width="170"></a> | Joint Classification + detection + segmentation with Interface |

---

## 🎯 Learning Outcomes
- ✅ Understand core → advanced CV architectures & training patterns  
- ✅ Implement classification, detection, tracking, segmentation, transformers  
- ✅ Build clean experiments with solid metrics & visual diagnostics  
- ✅ Master the three fundamental CV tasks that power all applications
- ✅ Be ready to apply CV in research and industry
- ✅ Strong foundation for advanced topics like GANs and Stable Diffusion

---

## 🛠️ Prerequisites
- **Programming**: Basic Python  
- **Math**: Intro linear algebra & calculus  
- **ML**: Basics helpful (optional)  
- **Hardware**: Colab GPU recommended
- **Level**: Undergraduate-friendly (designed for students new to CV)

---

## 📚 Additional Learning Resources

To complement these practicals, we highly recommend these additional resources for a complete Computer Vision education:

### 🎓 **Essential Foundations**

**Classical Image Processing** (Essential foundation - many modern CV solutions still use these techniques):
- [First Principles of Computer Vision - Columbia University](https://fpcv.cs.columbia.edu/)
  - High-quality content explaining the true science behind image processing
  - Essential for understanding data preprocessing and classical CV techniques

### 🏛️ **University Courses** (Highly Recommended)

**Technical University of Munich (TUM) - Excellent CV Course Series**:

1. **Introduction to Deep Learning (I2DL)**
   - [YouTube Playlist](https://www.youtube.com/playlist?list=PLQ8Y4kIIbzy_OaXv86lfbQwPHSomk2o2e)
   - Provides a different perspective on deep learning fundamentals
   - Excellent complement to hands-on practicals

2. **Computer Vision: Detection, Segmentation & Tracking (CV3DST)**
   - [YouTube Playlist](https://www.youtube.com/playlist?list=PLog3nOPCjKBkamdw8F6Hw_4YbRiDRb2rb)
   - Perfect name that captures the three fundamental CV tasks
   - Advanced techniques for the core CV applications

3. **Advanced Deep Learning for Computer Vision (ADL4CV)**
   - [YouTube Playlist](https://www.youtube.com/playlist?list=PLog3nOPCjKBkngkkF552-Hiwa5t_ZeDnh)
   - Cutting-edge CV research and advanced techniques
   - Perfect for those ready to explore state-of-the-art methods

### 🚀 **Recommended Learning Path**

1. **Start Here**: Complete these 25 practicals (undergraduate-friendly, hands-on)
2. **Supplement with**: Classical Image Processing course for theoretical foundations
3. **Deepen Understanding**: TUM's I2DL course for different perspectives
4. **Specialize**: TUM's CV3DST for advanced applications  
5. **Advanced Research**: TUM's ADL4CV for cutting-edge techniques

This combination provides both practical skills and theoretical depth needed for real-world CV applications!

---

## 📖 Citations & References

If you use this series in your research or work, please cite:

```bibtex
@misc{computer_vision_series_2025,
  title={Hands-On Computer Vision Series - A Practical Learning Journey},
  author={Dr. Zaid Al-Huda},
  year={2025},
  publisher={GitHub},
  url={https://github.com/zaidalhuda/Hands-On-Computer-Vision-Series-A-Practical-Learning-Journey}
}
```

### Key References:

- **"Practical Machine Learning for Computer Vision"** - Foundational reference for this series ([Link](https://www.amazon.co.uk/Practical-Machine-Learning-Computer-Vision/dp/1098102363))
- LeCun, Y., et al. "Deep learning." Nature 521.7553 (2015): 436-444.
- Krizhevsky, A., et al. "ImageNet classification with deep convolutional neural networks." Communications of the ACM 60.6 (2017): 84-90.
- Dosovitskiy, A., et al. "An image is worth 16x16 words: Transformers for image recognition at scale." arXiv preprint arXiv:2010.11929 (2020).


---

## 🌟 Ready to Start Your Computer Vision Journey?

Whether you're an undergraduate student working on your graduation project or someone completely new to Computer Vision, these practicals will guide you from fundamentals to advanced applications.

Remember: **Advanced CV techniques like GANs and Stable Diffusion are built upon mastering these fundamental tasks.** Start with the basics, understand the core concepts, and you'll be ready for anything!

📁 [Browse All Practicals](./notebooks/)

---

**⭐ Star this repository if you find it helpful!**

*Let's build the future of computer vision together - one practical at a time.*

**Made with ❤️ by Dr. Zaid Al-Huda**
