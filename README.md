# BinaryClassification-Cats-Dogs-Flask-tflearn
  - This is a very simple project which classifies an image into a dog or a cat. 
  - This project uses tflearn which is the best alternative to Keras.
  - A pre-trained model is also uploaded with an accuracy of 86.2%.
  - Using flask web framework, the images can be uploaded on a web page and the predicted class will displayed according to the predictions.

## Preview
![](screenshots/ss1.png)\
![](screenshots/ss2.png)\
![](screenshots/ss3.png)\
![](screenshots/ss4.png)\

## Packages
  - pip install tensorflow==1.15.3
  - pip install tflearn
  - pip install tqdm
  - pip install numpy
  - pip install opencv-python
  - pip install flask
  - pip install flask_bootstrap
  - Or you could: pip install -r requirements.txt
 
## Training our model:
  - After downloading the training and testing data (link provided below), extract them into test and train directories respectively.
  - From trainer.py run the file and keep changing the "n_epoch" value to get a higher accuracy. 
  - You can see the loss function and accuracy from the tensorboard platform. This is also mentioned in the code.
  - After training the model, load the saved model in classifier.py
  - Now, run the app.py file to access the web page on your local host.
  - There is a pre-trained model also available, download all three files named "dogsvscats-0.001-6conv-basic-15epochs.model" and load them into classifier.py.
 
## Resources
  - The training and testing data can be downloaded from here:
    - https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data
  - Other resources:
    - https://towardsdatascience.com/a-beginners-guide-to-convolutional-neural-networks-cnns-14649dbddce8
    - https://missinglink.ai/guides/convolutional-neural-networks/fully-connected-layers-convolutional-neural-networks-complete-guide/
    - https://opensource.com/article/18/4/flask
