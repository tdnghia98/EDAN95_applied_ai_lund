# Image classification report

## Describe the architecture used in the lab

The architecture I used in the lab was a fairly basic one that is very common. I alternated different layers of convolution and max pooling.

### Basic network

The first basic network looks like this:

 ![model_basic](model_basic.png)

This network suffers overfitting problem and only has an accuracy of 64%.

## Basic network with dropout to fight overfit

To resolve the problem of overfitting, I added a dropout layer which has improved the performance.

![model_basic_with_dropout](model_basic_with_dropout.png)

The accuracy bumped from 64% to 72%, which represents a 8% gain of accuracy with just one dropout layer added to the end of the network.

## Using a pretrained convolution base

Using the pretrained InceptionV3 base, I have tried two approaches.

### Use base output as input for my own network

I use the features extracted using the pretrained convolution as input for my network which only has 3 layers.

```
model_3 = models.Sequential()
model_3.add(layers.Dense(256, activation='relu', input_dim=3*3*2048))
model_3.add(layers.Dropout(0.5))
model_3.add(layers.Dense(5, activation='softmax'))

model_3.compile(optimizer=optimizers.RMSprop(learning_rate=2e-5), loss='categorical_crossentropy', metrics=['acc'])
```

The network stopped converging after only 5 epochs, with a gain of 10% of performance resulting to an accuracy of 82%.
This is a very notable gain which does not require a powerful machine and a long training time.

### Integrate the base convolution to my network with data augmentation

It is also possible to integrate the the pretrained base convolutional to our own network. This approach allows us to use data augmentation during training which was not available when using the features extraction as I previously did.

```
from keras import models, layers

model_4 = models.Sequential()
model_4.add(conv_base)
model_4.add(layers.Flatten())
model_4.add(layers.Dense(256, activation='relu'))
model_4.add(layers.Dense(5, activation='softmax'))

model_4.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(), metrics=['acc'])
```

Here we are extending the model and I have freezed the base so that the weight are not influenced by the other layers that I have added after.
On this model I have an accuracy of 71%.

## Experiment Chollet's 5.4 Notebook and read article

Being able to understand the way the network take decision is important, especially when it comes to debugging. Using Class Activation Map visualization, a technique that produces a heatmap indicating how important each location in the image is with respect to a given class.

In the Chollet's notebook, the example was on a photo of two elephants with a natural background. Knowing that the prediction 'African_elephant' has the highest propability, the Grad-CAM process was used to highlight the region where the activation was the strongest. Indeed, when stacking the activation heatmap on the original image, the most important zones were located at the elephants which is what we expected. A little interesting detail was that the ears of the elephants were strongly highlighted which gave us some insights about how the network could tell it was 'African' elephants.

## Run GCAM on a flower image

See PDF file attached