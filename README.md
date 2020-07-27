# Point Cloud Segmentation(real, indoor data)


* This is a result of direct application on real, indoor data with JSIS3D(CVPR 2019) as a PointNet-based model. 
* results of another model, as a voxelization & sparse-convolution based model, will be added later.

> **JSIS3D: Joint Semantic-Instance Segmentation of 3D Point Clouds with**<br/>
> **Multi-Task Pointwise Networks and Multi-Value Conditional Random Fields**<br/>
> Quang-Hieu Pham, Duc Thanh Nguyen, Binh-Son Hua, Gemma Roig, Sai-Kit
> Yeung<br/> *Conference on Computer Vision and Pattern Recognition (CVPR),
> 2019* (**Oral**)<br/>
> [Paper](https://arxiv.org/abs/1904.00699) |
> [Homepage](https://pqhieu.github.io/research/cvpr19/) |
> [Github](https://github.com/pqhieu/jsis3d)


## Usage

### Prerequisites
This code is tested in Manjaro Linux with CUDA 10.0 and Pytorch 1.0.

- Python 3.5+
- Pytorch 0.4.0+

### Installation
To use MV-CRF (optional), you first need to compile the code:

    cd external/densecrf
    mkdir build
    cd build
    cmake -D CMAKE_BUILD_TYPE=Release ..
    make
    cd ../../.. # You should be at the root folder here
    make

### Dataset
We have preprocessed the S3DIS dataset ([2.5GB](https://drive.google.com/open?id=1s1cFfb8cInM-SNHQoTGxN9BIyNpNQK6x))
in HDF5 format. After downloading the files, put them into the corresponding
`data/s3dis/h5` folder.

### Training & Evaluation
To train a model on S3DIS dataset:

    python train.py --config configs/s3dis.json --logdir logs/s3dis

Log files and network parameters will be saved to the `logs/s3dis` folder.

After training, we can use the model to predict semantic-instance segmentation
labels as follows:

    python pred.py --logdir logs/s3dis --mvcrf

To evaluate the results, run the following command:

    python eval.py --logdir logs/s3dis

For more details, you can use the `--help` option for every scripts.


### Prepare your own dataset
1. Add your point cloud data in './data/s3dis/raw_data' directory. The folder and .ply file in it should have the same name.
![data_add](./_images/data_add.png)
2. process data with process_data.py in ./scripts. This will make numpy data with .ply files. <br>
`python process_data.py --root data/s3dis`
3. prepare .h5 files with prepare_h5.py in './scripts' This will make .h5 files in my_h5 folder. <br>
`python prepare_h5.py --root data/s3dis`
4. predict with the trained model and make the prediction file. This will make my_pred.npz file in my_s3dis.<br>
`python my_pred.py --logdir logs/my_s3dis`
5. visualize the results with main.py<br>
`python main.py`

### Experiments on the real data

#### JSIS3D
There are 13 segmentation categories.
 > ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']

* If there is not matching object in the list, just 'clutter' category would be selected. <br>
* If there is a matching object(ex. chair), we will select it + 'clutter' category too.

![cup1](./_images/cup1.png)
The cups are segmented as 'bookcase' together. It needs a more detailed algorithm to segment cups out. 

![bottle1](./_images/bottle1.png)
All the points are segmented as 'clutter'. Sofa is not segmented. It might be because that number of points is small.

![bottle2](./_images/bottle2.png)
 Wall is segmented out right. However, some part is wrongly segmented as 'bookcase'.
 
 
 In overall, the category is too small to use & the real data is too noisy.<br>
 For example, to detect walls, the surface should be flat. In our data, surface is so bumpy which is quite different from 
S3DIS trained for this model. It is not robust enough to apply directly on noisy, incomplete data. 
 
 #### PointGroup

...updating
        

**Contact**: SeongJu Kang(djflstkddk@gmail.com)
