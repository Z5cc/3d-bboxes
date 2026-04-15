# 3D Bounding Boxes

2026-04-15 by David Nicklaser  

The project inferences 3D bounding boxes from a point cloud and an image segmentation mask.

## Setup

Move into the project directory:
```bash
cd 3d-bboxes
```

Make sure you have *Python 3.12* installed and activated. You can check with:
```bash
python3 --version
```

Create a virtual environment, activate it and install the requirements:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run the *train.py* file for training. Make sure to have data samples in the *./dl_challenge_train* directory. I did not upload the samples for this directory.
Running the *train.py* script should print logs of the following form:
```bash
python3 train.py
```


Run the *inference.py* file for inference and visualization.  Make sure to have data samples in the *./dl_challenge_train* directory. I uploaded four samples for this directory. These samples have not been used during the training of the model.
Running the *train.py* script should show logs of the following form:
```bash
python3 inference.py
```

To adjust parameters, modify the *constants.py* file.


## Code Structure

![png](docu/code-structure.drawio.svg)



## Methodology

**Data Loader** [`utils/dataset_dl_challenge.py`]

When inspecting the data, I found out that the order of bounding boxes in a file and the order of masks in the other file are the same. then i decided to run one inference per mask or per bounding box. 
The advantage of this approach is the following: having only to inference one object is easier for the model. also the the archtiecture is simpler because no variable size has to be infereenced anymore.
The disadvantage of this appraoch is the following: the total inference operations necessary to predicat all might be higher. it might take also longer to run the small model several times than inferencing all objects at once with a big model.
i cut out a 256x256 field out of the point cloud. the center of this field is the center of the mask. if i am close to the border of the image, the center is shifted respectively to still have 256x256. as input to the neural network is 4x256x256. 1 channel for the mask 0.0 for false 1.0 for true.  3 channels for pc: x,y,z no preprocessing. rgb values have not been used. though might be worse an investigation, because of the shadows which can not be retrieved from the point cloud.

**Architecture** [`utils/network.py`]

4@256x256  -> **conv(3x3)** -> 8@256x256  -> **conv(3x3)** -> 16@256x256 -> **avgPool(2x2)** -> <br>
16@128x128 -> **conv(3x3)** -> 32@128x128 -> **conv(3x3)** -> 32@128x128 -> **avgPool(2x2)** -> <br>
32@64x64   -> **conv(3x3)** -> 32@64x64   -> **conv(3x3)** -> 32@64x64   -> **avgPool(2x2)** -> <br>
32@32x32   -> **conv(3x3)** -> 32@32x32   -> **conv(3x3)** -> 32@32x32   -> **avgPool(2x2)** -> <br>
32@16x16   -> **conv(3x3)** -> 32@16x16   -> **conv(3x3)** -> 32@16x16   -> **avgPool(2x2)** -> <br>
32@8x8     -> **flatten** -> 2048       -> **FC**      -> 512        -> **FC**         -> 9 (y)

Strides of convolutions are all 1, strides of pooling layers are all 2. Average Pooling and not Max Pooling was selected, because averaging is bettern than maxing for a regression task like finding a middle point. Also diverging any other way from this homogenous architecture structure did not bring any improvements. such diverging has been done via for example. increasing convolution in begining from 3x3 to 5x5. or replacing a pooling with a stride 2 of convolution. amount of linear layers have been increased until loss started to go down.


**Converting Neural Network Output to Bounding Boxes** [`utils/geometry.py`]

code:
bb = create_bb(y)

$bb = rot\_fn(base*size) + center$

rot_fn, size and center are all derived from y:<br>
$center = y[0:3]$<br>
$size = softplus(y[3:6])$<br>
$angles = tanh(y[6:9])*(π/4)$

+-45 degrees for rotation chosen, because after 45 degree you can just change the widths. Regarding rotation of the cube, maybe quaternion or another rotation representaiton performs better. my approach to implementing the rotation of the cube was limited by time. the tanh function probably is problematic because values close to +-45 degrees are hard too reach. I choose euler here, because it is simple. though no risk of gimble lock because of chosen degrees of +-45,but non homogenous distribution makes euler problematic

**Loss Function** [`utils/geometry.py`]

code:
loss = loss_bb(bb, bb_truth)


$\mathcal{L} = \left( \min_{\sigma \in \mathcal{P}} \sum_{i=0}^{7} \| \mathbf{b}_{\sigma(i)} - \mathbf{b}^{\text{truth}}_i \| \right)^2$

this is the loss for one bounding box. over one batch, the loss is averaged. b is a corner of bounding box bb. P are all 24 rotation permutations of a cube. 6 different faces which can point upwards. and for each face pointing up you have 4 rotations.
calcultaing loss via center, size and angles directly would also be possible, but you would need to toake care of balance factors.
calculating loss via cutting volume is complicated.
In this loss however only distance vectors need to be calculated, that is for all rotation permutation of a cube, that are 24 permutation, the destiance vectors between the 8 ground trouth corner and 8 prediction corner are summed up. the smallest of these 24 sums is squared, which is then the loss.

## Demo

loss function....
both infernce and train loss to know weither the model under- or overfits.

image of inference.....

## Credits and Next Steps

I did this project without the help of others. Furthermore, I conciously did not do any research on how 3D bounding boxes are solved nowadays to have a higher learning effect.
The loss function came from my own thoughts and is not a suggestion from an AI. Neither did I read it up somewhere.
Next thing on this project, I would improve is code quality and structure. Time was running out. Some functions need to be done smaller. comment more and better. Also I would add more pytests to the project.
After that Hyperparameters like the learning rate I would adjust.
Also architecture needs improvement.
Investigation weither quaternion or other rotaion representation performs better.
