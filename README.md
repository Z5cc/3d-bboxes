# 3d-bboxes

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
I saved the logs from the current *model.pth* under *./demo/train_logs*.
```bash
python3 train.py
```


Run the *inference.py* file for inference and visualization.  Make sure to have data samples in the *./dl_challenge_train* directory. I uploaded four samples for this directory. These samples have not been used during the training of the model.
Running the *train.py* script should show logs of the following form:
I saved the logs from the current *model.pth* under *./demo/inference_logs*.
```bash
python3 inference.py
```

To adjust parameters, modify the *constants.py* file.


## Code Structure





## Methodology

**Architecture**

DQN is used for the algorithm because of its simplicity. The neural network for the policy and the target network is mainly built from convolutional layers, based on the following idea. Kernels with a high–low–high pattern can detect far–close–far structures in the image, which correspond to good grasp locations. The policy network, which takes a state $s \in \mathcal{S}$ as input and outputs an action $a \in \mathcal{A}$, is designed as follows:  

4x256x256 -> **conv(3)** -> 8x16x16 -> **pool(2)** -> 8x8x8 -> **conv(3)** -> 16x8x8 -> **conv(3)** -> 16x8x8 -> **Flatten** -> 1024 -> **FC** -> 13

**Loss Function**

**Training**


## TODO




## Credits

I did this project without the help of others. Furthermore, I conciously did not do any research on how 3D bounding boxes are solved nowadays to have a higher learning effect.
