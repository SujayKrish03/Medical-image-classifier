# Medical-image-classifier
A CNN-powered image classification tool that distinguishes medical images (like X-rays, CT scans) from non-medical content. Supports input from direct uploads, web URLs, and PDFs by extracting and analyzing images using deep learning. Built with Gradio for an interactive user interface.


# Problem Statement
Medical imaging plays a key role in healthcare, but real-world systems often need to:
-Automatically identify whether an image is medical (e.g., X-ray) or not
-Handle unstructured content from the web and documents (PDFs)
-Present predictions in a clear, visual format for non-technical users
-This project solves that with a CNN classifier + scraping pipeline + Gradio interface.

# Why Cnn?
In this project, we aim to classify images as medical or non-medical using a Convolutional Neural Network (CNN), a deep learning architecture specifically designed for image data. CNNs are particularly well-suited for this task because they automatically learn spatial hierarchies of features from raw pixel data, eliminating the need for manual feature extraction. This is especially important in medical imaging, where critical patterns and anomalies—such as subtle textures or abnormal shapes—can be difficult to handcraft as features. By applying convolutional filters, CNNs can detect low-level patterns like edges and textures in early layers, and more complex structures like organs or pathologies in deeper layers. Additionally, CNNs use parameter sharing and local receptive fields, which make them efficient and capable of generalizing well even with limited labeled data—common in the medical domain. Compared to traditional machine learning models or general-purpose neural networks, CNNs are more accurate, scalable, and robust when dealing with high-resolution images such as X-rays, CT scans, or other diagnostic visuals. This makes CNNs a natural and powerful choice for building a reliable medical image classifier.

# Approach
The objective of this project is to develop a system that can accurately classify images as medical or non-medical using a Convolutional Neural Network (CNN). The approach begins with gathering and preparing a combined dataset sourced from the Chest X-ray Kaggle dataset and the Medical MNIST dataset to represent medical images, while images from datasets like CIFAR-10 are used to represent non-medical visuals. All images are resized and normalized to ensure consistency across the dataset.
A CNN model is then designed and trained on this labeled dataset. The CNN architecture typically consists of convolutional layers to extract features, pooling layers to reduce dimensionality, and fully connected layers to perform the final classification. The model is trained using cross-entropy loss and optimized using an algorithm like Adam. After sufficient training, the model learns to distinguish between patterns found in medical vs. non-medical images.
To extend usability, the system includes modules for classifying images not only from direct uploads but also from web URLs and PDFs using scraping and image extraction tools. These inputs are passed through the trained CNN model to predict and return the result along with thumbnail previews. A Gradio interface wraps the entire pipeline to allow users to easily interact with the tool in a web-based environment.

