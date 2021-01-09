1. rtabmap.launch  
`/rtabmap/cloud_map` has smaller resolution than `/rtabmap/mapData`. This has been answered by rtabmap_ros author in [ROS ANSWERS](https://answers.ros.org/question/239760/how-to-get-maps-point-cloud-from-rtab_map/). 
   These two parameters have been added [rtabmap.launch](https://github.com/ZhiangChen/target_mapping/blob/master/launch/rtabmap.launch).
```
<param name="Grid/CellSize" type="string" value="0.01"/>
<param name="Grid/DepthDecimation" type="string" value="1"/>
```