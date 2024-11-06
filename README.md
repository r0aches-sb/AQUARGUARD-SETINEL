# AQUARGUARD-SENTINEL
Aquarguard sentinel is a project aimed at providing real-time monitoring and alert systems that make use of satellite images and sensor data to pinpoint locations with bad drainage and predict imminent failures by use of machine learning techniques

Problem Statement
   
Cities are in a great predisposition towards flooding, in most cases due to poor drainage. Conventional means employed for surveillance and maintenance are reactive in most instances. Indeed, this has led to delays in fixing the problems and hence exacerbating flood damage. AquaGuard Sentinel shall provide a real-time monitoring and alert system that makes use of satellite images and sensor data to pinpoint locations with bad drainage and predict imminent failures, alerting the concerned authorities to take action on time and forestall the event.

Some AI Methods Applied include;
 
Machine Learning Models used include;

1. Convolutional Neural Networks
The chosen architecture was a custom CNN designed to balance complexity with performance, given the constraints of satellite image analysis. The model was used to analyze satellite images to identify water accumulation and blockages in drainage systems. CNN was also used to classify satellite images of drainage pipes into clogged, almost clogged or unclogged.
 
2.Anomaly Detection Algorithms;

ADAs were useful in identifying unusual patterns in drainage system data that could result in failures.
The algorithms implemented were used to track anomalies in normal behavior, generate warnings at the earliest possible opportunity that there may be a problem.

Data Sources;

•	 Satellite Images: High-resolution images to monitor surface water and vegetation
•	 Sensor Data: Real-time measurements of flow rates and pressure of water, rainfall and capacity of drainage system.

 System Architecture
 
 Data Collection;
   
•	Satellite Data Provider: Was used to gather real-time satellite images
•	Sensor Network: The Sensor Network is spread across the city and measures the flow of water and its pressure, rainfall, and other parameters.

 Data Processing Layer 
 
• Preprocessing Module: Data was cleaned and pre-processed to  reduce noise and for normalization purposes.
• Feature Extraction Module: Relevant features from both the satellite images and the sensor data were extracted to use in the machine learning model.

Machine Learning Layer:

• Image Analysis Module: The module was used for analyzing satellite images using CNNs and detecting regions having water accumulation and possible blockages. The CNN model was also used for image classification (Clogged, almost clogged and unclogged).
• Anomaly Detection Module: Anomaly detection algorithms were also applied against the sensor data for any unusual patterns indicative of potential failures.

 Alert System Layer:
• Notification Module: Sends alerts to municipal authorities, including actionable insights and recommendations, from the results produced by the ML module .

 User Interface Layer:
• Dashboard: Real-time data, predictions, and alerts for municipal authorities.
• Mobile App: On-the-go access to alerts and system status for emergency responders.

 Preliminary Development of the ML Model
 
Phase 1: Data Collection and Preprocessing
•	Satellite Images: Sample images from a satellite data provider were obtained. Preprocessing amounted to resizing, normalization, and noise reduction.
•	Sensor Data: Historical sensor data was gathered from installed sensors in drainage pipes around the city. Cleaning, handling missing values, and normalizing the data was also done as part of preprocessing.

Phase 2: Feature Extraction

• Satellite Images:  Features related to water bodies, vegetation, and urban structures were extracted using image processing techniques.
• Sensor Data: Features like average flow rate and pressure, peak flow rate, and rainfall intensity over time were also extracted.

Phase 3: Model Development

 Image Analysis (CNN):
o Architecture:  CNN was used for analysis of the satellite images and the classification.
   Anomaly Detection:
o Algorithm: An isolation forest algorithm enabled us to identify anomalies in sensor data.
o Training: The models were trained on normal operational data to establish a baseline.
O Validation: The initial validation of anomaly detection with a minimum precision of 75%, and further tuning of the algorithm was planned.
 
 Model Selection and Training
 Convolutional Neural Networks (CNNs)  were used due to their effectiveness in image classification tasks. 
The model was trained and set to a binary labels for training. 0 to represent no flooding risks detected and 1 for a flooding risk detected.
We also used an LSTM model as a time series prediction based on the sensor data since it involves information such as water flow rates. An LSTM model was best fitted to work on the long term dependencies. In this project, the LSTM model predicts future values of sensor data to identify potential flood conditions.
In addition, we opted for an Isolation Forest algorithm to detect and predict anomalies. It is trained on sensor features to identify outliers or anomalies that might indicate potential flooding.

Testing and Evaluation
The trained model was evaluated using the testing dataset. The primary metrics for evaluation included accuracy, precision, recall, and F1-score. These metrics provided a comprehensive view of the model’s performance in correctly identifying the condition of the drainages.

 Testing Results
The initial results from testing the model were below expectations:
We achieved an accuracy: 58%
Precision: Varying significantly across the two categories, with clogged(potential flood risk detected) and unclogged(No flood risks detected).
Recall and F1-Score: Similarly inconsistent, indicating difficulties in reliably distinguishing between the different states of drainage.


 Challenges Encountered
 
 Subpar Accuracy

One of the major challenges was the model’s accuracy, which stood at 58% against a target of 94% or higher. Several factors contributed to this issue:
Class Imbalance: The dataset had a disproportionate number of images in each category, with semi-clogged drainages being underrepresented. This imbalance likely caused the model to be biased towards the more frequent categories.
Quality of Images: Variations in image quality and resolution affected the model's ability to accurately classify the images. Satellite images can have inconsistencies due to weather conditions, lighting, and other environmental factors.

Finding a Comprehensive Dataset

Another significant challenge was sourcing a dataset that included all necessary variables:
Limited Availability: There were no  publicly available datasets containing comprehensive and accurately labeled images of drainage systems in all three categories that we had required.
Diversity of Conditions: The available datasets lacked diversity in the conditions eg: Rain clogging, Garbage Clogging and Soil clogging under which images were taken, making it difficult for the model to generalize well to new, unseen images.

 Steps Forward
Enhancing Data Collection
   
To address the data-related challenges, the following steps were planned:
Data Sourcing: Partnering with more municipal bodies and satellite image providers to obtain a larger and more diverse dataset.
Crowdsourcing Labels: Utilizing platforms where users can assist in labeling images to expand the ground accuracy of the dataset.

Model Improvement

To improve model accuracy, the following strategies will be employed:

Advanced Architectures: Experimenting with more complex architectures, such as ResNet or DenseNet, which may better capture the intricacies of satellite images.
Transfer Learning: Leveraging pre-trained models on similar image classification tasks to improve the model’s performance with limited data.

Addressing Class Imbalance
Balancing the dataset to ensure fair representation of all categories through:
Synthetic Data Generation: Using techniques like SMOTE (Synthetic Minority Over-sampling Technique) to create synthetic examples of underrepresented categories.
Data Augmentation: Further expanding the dataset with augmented images to balance the classes.

Conclusion

The AquaGuard project has made significant progress in developing an AI-based solution for classifying the condition of drainage systems using satellite imagery. Despite the challenges of achieving the desired accuracy and sourcing comprehensive datasets, the project has established a solid foundation. Continued efforts in data enhancement and model optimization are expected to improve performance and achieve the project’s goals.



