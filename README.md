# Bytehoven - Sheet music recognition with ResNet50

![Byethoven](images/Bytehoven.jpeg)

This repository contains the resources used in the development of Bytehoven, a deep learning sheet music recognition model currently in development. [Please find the full tutorial here.](http://www.paulvangent.com/2017/12/07/deep-learning-music/) The current version recognises piano music from Bach, Beethoven, Brahms, Chopin, Grieg, Liszt, and Mozart.

Trained model weights are available under 'releases'. We'll keep training on more data and composers to increase Bytehoven's knowledge base.

# Included files

- datasets/Musicdata_Small.rar -- Dataset of small sized images (200*35px)
- datasets/Musicdata_Medium.rar -- Dataset of medium sized images (400*70px)
- Release V0.1: bytehoven-7-weights.hdf5 -- Model weights trained on medium sized set (full training log included)
- ResNet50.py -- ResNet50 architecture implemented in Keras
- run_ResNet50.py -- Example to initiate training run

# Network activation and filter visualisation

Final network layer activation for two random pieces:
![Beethoven](images/Beethoven_Visualisation.jpg)
![Chopin](images/Chopin_Visualisation.jpg)

Random filters visualised (filters lower in the image are deeper in the network):
![Filters](images/Filters.jpg)