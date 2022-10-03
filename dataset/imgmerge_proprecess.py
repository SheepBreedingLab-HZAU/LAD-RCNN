# Copyright (c) 2022 Jiang Xunping and Sun Ling.
#
# Licensed under the MIT license;

# -*- coding: utf-8 -*-


import tensorflow as tf
import tools.standard_fields as fields
import boxlist.box_list_ops as box_list_ops
import boxlist.box_list as box_list
import tools.config as config
InputFields=fields.InputDataFields()
ADD_RATIO=1e-8
def get_coordinate_box(box,wind):
    box.set_shape([None,5])
    return box_list_ops.change_coordinate_frame(box_list.BoxList(box),wind).get()
def _1X1(image,num_groundtruth_boxes,groundtruth_boxes):
    out_tensor_dict={}
    out_tensor_dict[InputFields.image]=image[0]
    out_tensor_dict[InputFields.num_groundtruth_boxes]=num_groundtruth_boxes[0]
    out_tensor_dict[InputFields.groundtruth_boxes]=groundtruth_boxes[0]
    
    return out_tensor_dict
def _1X2(image,num_groundtruth_boxes,groundtruth_boxes):
    #tf.print(num_groundtruth_boxes)
    out_tensor_dict={}
   
    outimg=tf.concat([image[0],image[1]],axis=1,name='img_comb_1X2')
    groundtruth_boxe0=groundtruth_boxes[0][:num_groundtruth_boxes[0],0:5]
    groundtruth_boxe1=groundtruth_boxes[1][:num_groundtruth_boxes[1],0:5]
    
    groundtruth_boxe0=get_coordinate_box(groundtruth_boxe0,[0.,0.,1.,2.])
    groundtruth_boxe1=get_coordinate_box(groundtruth_boxe1,[0.,-1.,1.,1.])
    out_gt_boxes=tf.concat([groundtruth_boxe0,groundtruth_boxe1],axis=0)
    
    #out_num_boxes=tf.constant(2)#tf.add(num_groundtruth_boxes[0],num_groundtruth_boxes[1])
  
    
    out_num_boxes=tf.reduce_sum([num_groundtruth_boxes[0],num_groundtruth_boxes[1]])
    out_tensor_dict[InputFields.image]=outimg
    out_tensor_dict[InputFields.num_groundtruth_boxes]=out_num_boxes
    out_tensor_dict[InputFields.groundtruth_boxes]=out_gt_boxes
    return out_tensor_dict
def _1X3(image,num_groundtruth_boxes,groundtruth_boxes):
    #tf.print(num_groundtruth_boxes)
    out_tensor_dict={}
   
    outimg=tf.concat([image[0],image[1],image[2]],axis=1,name='img_comb_1X3')
    
    groundtruth_boxe0=groundtruth_boxes[0][:num_groundtruth_boxes[0],0:5]
    groundtruth_boxe1=groundtruth_boxes[1][:num_groundtruth_boxes[1],0:5]
    groundtruth_boxe2=groundtruth_boxes[2][:num_groundtruth_boxes[2],0:5]
    
    groundtruth_boxe0=get_coordinate_box(groundtruth_boxe0,[0.,0.,1.,3.])
    groundtruth_boxe1=get_coordinate_box(groundtruth_boxe1,[0.,-1.,1.,2.])
    groundtruth_boxe2=get_coordinate_box(groundtruth_boxe2,[0.,-2.,1.,1.])
    
    out_gt_boxes=tf.concat([groundtruth_boxe0,groundtruth_boxe1,groundtruth_boxe2],axis=0)
    #.print(num_groundtruth_boxes[0],num_groundtruth_boxes[1],num_groundtruth_boxes[2])
    
    out_num_boxes=tf.reduce_sum([num_groundtruth_boxes[0],num_groundtruth_boxes[1],num_groundtruth_boxes[2]])
    
    out_tensor_dict[InputFields.image]=outimg
    out_tensor_dict[InputFields.num_groundtruth_boxes]=out_num_boxes
    out_tensor_dict[InputFields.groundtruth_boxes]=out_gt_boxes
    return out_tensor_dict
    
    
def _2X1(image,num_groundtruth_boxes,groundtruth_boxes):
    #tf.print(num_groundtruth_boxes)
    out_tensor_dict={}
   
    outimg=tf.concat([image[0],image[1]],axis=0,name='img_comb_2X1')
    
    groundtruth_boxe0=groundtruth_boxes[0][0:num_groundtruth_boxes[0],0:5]
    groundtruth_boxe1=groundtruth_boxes[1][0:num_groundtruth_boxes[1],0:5]
    
    groundtruth_boxe0=get_coordinate_box(groundtruth_boxe0,[0.,0.,2.,1.])
    groundtruth_boxe1=get_coordinate_box(groundtruth_boxe1,[-1.,0.,1.,1.])
    out_gt_boxes=tf.concat([groundtruth_boxe0,groundtruth_boxe1],axis=0)
    
    out_num_boxes=tf.reduce_sum([num_groundtruth_boxes[0],num_groundtruth_boxes[1]])
    
    out_tensor_dict[InputFields.image]=outimg
    out_tensor_dict[InputFields.num_groundtruth_boxes]=out_num_boxes
    out_tensor_dict[InputFields.groundtruth_boxes]=out_gt_boxes
    return out_tensor_dict
def _2X2(image,num_groundtruth_boxes,groundtruth_boxes):
    out_t1=_1X2(image[0:2],num_groundtruth_boxes[0:2],groundtruth_boxes[0:2])
    out_t2=_1X2(image[2:4],num_groundtruth_boxes[2:4],groundtruth_boxes[2:4])
    kwgs=[[a,b] for a,b in zip(out_t1.values(),out_t2.values())]
    return _2X1(*kwgs)
def _2X3(image,num_groundtruth_boxes,groundtruth_boxes):
    out_t1=_1X3(image[0:3],num_groundtruth_boxes[0:3],groundtruth_boxes[0:3])
    out_t2=_1X3(image[3:6],num_groundtruth_boxes[3:6],groundtruth_boxes[3:6])
    kwgs=[[a,b] for a,b in zip(out_t1.values(),out_t2.values())]
    return _2X1(*kwgs)
def _3X1(image,num_groundtruth_boxes,groundtruth_boxes):
    #tf.print(num_groundtruth_boxes)
    out_tensor_dict={}
    outimg=tf.concat([image[0],image[1],image[2]],axis=0,name='img_comb_3X1')
    
    groundtruth_boxe0=groundtruth_boxes[0][0:num_groundtruth_boxes[0],0:5]
    groundtruth_boxe1=groundtruth_boxes[1][0:num_groundtruth_boxes[1],0:5]
    groundtruth_boxe2=groundtruth_boxes[2][0:num_groundtruth_boxes[2],0:5]
    groundtruth_boxe0=get_coordinate_box(groundtruth_boxe0,[0.,0.,3.,1.])
    groundtruth_boxe1=get_coordinate_box(groundtruth_boxe1,[-1.,0.,2.,1.])
    groundtruth_boxe2=get_coordinate_box(groundtruth_boxe2,[-2.,0.,1.,1.])
    out_gt_boxes=tf.concat([groundtruth_boxe0,groundtruth_boxe1,groundtruth_boxe2],axis=0)  
    
    out_num_boxes=tf.reduce_sum([num_groundtruth_boxes[0],num_groundtruth_boxes[1],num_groundtruth_boxes[2]])
    
    out_tensor_dict[InputFields.image]=outimg
    out_tensor_dict[InputFields.num_groundtruth_boxes]=out_num_boxes
    out_tensor_dict[InputFields.groundtruth_boxes]=out_gt_boxes
    return out_tensor_dict
def _3X2(image,num_groundtruth_boxes,groundtruth_boxes):
    out_t1=_1X2(image[:2],num_groundtruth_boxes[:2],groundtruth_boxes[0:2])
    out_t2=_1X2(image[2:4],num_groundtruth_boxes[2:4],groundtruth_boxes[2:4])
    out_t3=_1X2(image[4:6],num_groundtruth_boxes[4:6],groundtruth_boxes[4:6])
    kwgs=[[a,b,c] for a,b,c in zip(out_t1.values(),out_t2.values(),out_t3.values())]
    return _3X1(*kwgs)
def _3X3(image,num_groundtruth_boxes,groundtruth_boxes):
    out_t1=_1X3(image[:3],num_groundtruth_boxes[:3],groundtruth_boxes[:3])
    out_t2=_1X3(image[3:6],num_groundtruth_boxes[3:6],groundtruth_boxes[3:6])
    out_t3=_1X3(image[6:9],num_groundtruth_boxes[6:9],groundtruth_boxes[6:9])
    kwgs=[[a,b,c] for a,b,c in zip(out_t1.values(),out_t2.values(),out_t3.values())]
    return _3X1(*kwgs)

def process(tensor_dict,record_seq):
    
    dick=[InputFields.image,InputFields.num_groundtruth_boxes,InputFields.groundtruth_boxes]
    image                   =tensor_dict[InputFields.image]
    
    num_groundtruth_boxes   =tensor_dict[InputFields.num_groundtruth_boxes]
    groundtruth_boxes       =tensor_dict[InputFields.groundtruth_boxes]
    image = tf.unstack(image,axis=0)
    num_groundtruth_boxes=tf.unstack(num_groundtruth_boxes,axis=0)
    groundtruth_boxes=tf.unstack(groundtruth_boxes,axis=0)
    
    if len(image)==1:
        image=image*9
        num_groundtruth_boxes=num_groundtruth_boxes*9
        groundtruth_boxes=groundtruth_boxes*9
    elif len(image)==4:
        image=image*3
        num_groundtruth_boxes=num_groundtruth_boxes*3
        groundtruth_boxes=groundtruth_boxes*3
        
    out_tensor_dict={}
    for i in [k for k in tensor_dict if k not in dick]:
        out_tensor_dict[i]=tensor_dict[i][0]

    random=tf.random.uniform([])

    if record_seq==1:
        p1=config.MERGERED_1X1PROBABILITY1
        p2=config.MERGERED_2X2PROBABILITY1
        p3=config.MERGERED_3X3PROBABILITY1
    elif record_seq==2:
        p1=config.MERGERED_1X1PROBABILITY2
        p2=config.MERGERED_2X2PROBABILITY2
        p3=config.MERGERED_3X3PROBABILITY2
    else :
        raise Exception("record_seq should be 1 or 2")
    sump=p1+p2+p3
    if sump == 0:
        raise Exception("sum of config.MERGERED_1X1PROBABILITY,config.MERGERED_2X2PROBABILITY and config.MERGERED_3X3PROBABILITY should not be 0")
    if p1 <0 :
        raise Exception("config.MERGERED_1X1PROBABILITY should not less then 0")
    if p2 <0 :
        raise Exception("config.MERGERED_1X1PROBABILITY should not less then 0")
    if p3 <0 :
        raise Exception("config.MERGERED_1X1PROBABILITY should not less then 0")
    p1=p1/sump
    p2=p1+p2/sump
    p3=p2+p3/sump
    # tf.print(f"record_seq:{record_seq},p1:{p1},p2:{p2},p3:{p3}")
    # tf.print(f"random:{random}")

    # tf.print("less:{}".format(tf.less(random,p1)))
    mergered_tensor=tf.cond(tf.less(random,p1),
            lambda:_1X1(image,num_groundtruth_boxes,groundtruth_boxes),
            lambda:tf.cond(tf.less(random,p2),
                    lambda:_2X2(image,num_groundtruth_boxes,groundtruth_boxes),   
                    lambda:_3X3(image,num_groundtruth_boxes,groundtruth_boxes))
            )
    mergered_tensor['image']=tf.image.resize(mergered_tensor['image'],[config.IMG_HEIGHT,config.IMG_WIDTH])
    #print('mergered_tensor shape',mergered_tensor['image'].shape)
    out_tensor_dict.update(mergered_tensor)
    #print('out_tensor_dict_shape',out_tensor_dict['image'].shape)
    return out_tensor_dict