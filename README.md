# diverse-colorization
This project explores diverse colorization of anime sketches.

The dataset:

https://www.kaggle.com/ktaebum/anime-sketch-colorization-pair

The colorization works as follows:
1. Use a modified version of U-Net to predict N(here N=32) colorizations of the sketch. So instead of outputting a 3 x H x W image, output a N x 3 x H x W tensor.
2. Compute the L1 loss between the ground truth and the closest prediction to the GT in terms of L1 distance. The rationale is that there are many valid ways to color a sketch, so ignore the N-1 colorizations which are distant from the GT.
3. In order to encourage all of the N predictions to be used, at the end of every epoch, check every group of 3 kernels which correspond to 3(RGB) feature maps in the colorization, if a group of kernels was not updated from the previous epoch, this means that its corresponding colorization isn't used and their colorization is probably bad, therefore - copy the weights from some different group of kernels which their weights have been updated during the epoch. At the start it will produce identical colorization, but because these kernels now produce colorizations which are the closest to some of the GT, the kernels weights will be updated and the colorizations will diverge from the ones produced from the copied kernels.

Remark: the training was not performed until convergence due to a lack of computational resources, 150 epochs were performed.
The code is a bit messy and condensed as I programmed on a Google Collab Jupyter notebook.

A ground truth colorization and the sketch from the validation set:

![alt text](/colorizations/sk1_1.png?raw=true)


The 32 predictions on the sketch:
![alt text](/colorizations/grid_32_1.png?raw=true)

