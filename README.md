
# LAD-RCNN

![](https://img.shields.io/static/v1?label=python&message=3.8&color=blue)
![](https://img.shields.io/static/v1?label=TensorFlow&message=2.8&color=<COLOR>)
![](https://img.shields.io/static/v1?label=license&message=MIT&color=green)

This is official TensorFlow implementation of "[LAD-RCNN:A Powerful Tool for Livestock Face Detection and Normalization](https://arxiv.org/abs/2210.17146)"


# Framework
![The overall pipeline of the LAD-RCNN](https://github.com/SheepBreedingLaboratory/LAD-RCNN/blob/main/Figure/LAD-RCNN.jpg)




#  Train Your Data

## 1 Calculate rotation angle

 - Please refer to our paper to calculate the rotation angle。

## 2 Prepare dataset 1 which has angle data
### The structure of the dataset is as follows
 - imgHeight and imgWidth are list of int containg one item
 - encoded\_jpg can be generated form function tf.io.gfile.GFile({filepath},'rb').read()
 - xmins, xmaxs, ymins, ymaxs, and angles are list of float
###
	tf_example = tf.train.Example(features=tf.train.Features(feature={  
        'image/height':tf.train.Feature(int64_list=tf.train.Int64List(value=imgHeight)),  
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=imgWidth)),  
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(filename)),  
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(encoded_jpg)),
        'image/format':  tf.train.Feature(bytes_list=tf.train.BytesList(image_format)),  
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),  
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),  
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),  
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
        'image/object/bbox/angle': tf.train.Feature(float_list=tf.train.FloatList(value=angles))
    }))
## 3 Prepare dataset 2 which has no angle data
### The structure of the dataset is as follows
 - The structure of dataset2 is similar to dataset1, but there is no angle
 - However, even if angle data is included in dataset2, LAD-RCNN can work normally
###
	tf_example = tf.train.Example(features=tf.train.Features(feature={  
        'image/height':tf.train.Feature(int64_list=tf.train.Int64List(value=imgHeight)),  
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=imgWidth)),  
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(filename)),  
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(encoded_jpg)),
        'image/format':  tf.train.Feature(bytes_list=tf.train.BytesList(image_format)),  
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),  
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),  
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),  
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs))
    }))


## 4 Parameter Setting
 - You can set parameter in config file. The file path is tool/config.py
 - Set config.TFRECORD_PATH1 to the url of dataset1
 - Set config.TFRECORD_PATH2 to the url of dataset2
 - You could directly used the original parameters on frist training, and then modify the transformation parameters of dataset1 and dataset2 according to the test result.
 - Image input size is NOT restricted in `400 * 400`, You can adjust your input sizes for a different size or different input ratio. Larger input size could help detect smaller targets, but may be slower and GPU memory exhausting.

 - All the parameter in config file could be adjusted according to the prompts 



## 5 Start training

```
 python train.py -ts 25000

-ts train steps

If you have any questions, please contact Ling Sun,E-mail:ling.sun-01@qq.com
```
 
# Requirement
- [ ] TensorFlow  2.8.0
- [ ] Python > 3.7
# Detection Results
 - Detection examples on sheep bird-view image with LAD-RCNN.
![Detection examples on sheep bird-view image with LAD-RCNN.](https://github.com/SheepBreedingLaboratory/LAD-RCNN/blob/main/Figure/Sheep.jpg)
The small image in the upper right corner is the normalized sheep face extracted according to the detection result
 - Detection examples on infrared image with LAD-RCNN.
![Detection examples on infrared image with LAD-RCNN.](https://github.com/SheepBreedingLaboratory/LAD-RCNN/blob/main/Figure/infrared.jpg)
If you want to detection object in infrared images, please set INPUT_CHANNEL=1 in config file. 

# Citation

We would be happy to hear back from you in you find LAD-RCNN useful. If you use the LAD-RCNN for a research publication, please consider citing:

    @article{
        Author = {Ling Sun, Guiqiong Liu, Xunping Jiang*,Junrui Liu, Xu Wang, Han Yang, Shiping Yang},
        Title = {LAD-RCNN: A Powerful Tool for Livestock Face Detection and Normalization},
        Journal = {arxiv},
        Year = {2022}
    }