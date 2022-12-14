<h1 align="center">IRSTD-1k：The dataset of ISNet [CVPR 2022]</h1>

<p align="center">

<a href=""> <img  src="https://img.shields.io/badge/license-MIT-blue"></a>
</p>

<h4 align="center">This is the official repository of the paper ''ISNet: Shape Matters for Infrared Small Target Detection''</a>.</h4>


<h5 align="center"><em>Mingjing Zhang, Rui Zhang<sup>&#8727;</sup>, Yuxiang Yang, Haichen Bai, Jing Zhang<sup>&#8727;</sup>, and Jie Guo</em></h5>

[//]: # (<p align="center">)

[//]: # (  <a href="#introduction">Introduction</a> |)

[//]: # (  <a href="#ppt-setting-and-p3m-10k-dataset">PPT and P3M-10k</a> |)

[//]: # (  <a href="#p3m-net">P3M-Net</a> |)

[//]: # (  <a href="#benchmark">Benchmark</a> |)

[//]: # (  <a href="#results">Results</a> |)

[//]: # (  <a href="https://github.com/JizhiziLi/P3M/tree/master/core">Train and Test</a> |)

[//]: # (  <a href="##inference-code---how-to-test-on-your-images">Inference code</a> |)

[//]: # (  <a href="#statement">Statement</a>)

[//]: # (</p>)

[//]: # (<img src="demo/gif/p_2c2e4470.gif" width="25%"><img src="demo/gif/p_4dfffce8.gif" width="25%"><img src="demo/gif/p_d4fd9815.gif" width="25%"><img src="demo/gif/p_64da52e3.gif" width="25%">)

***
><h3><strong><i>:postbox: News</i></strong></h3>
>
> [2022-3-20]: Publish the dataset [<strong>IRSTD-1k</strong>](#ppt-setting-and-p3m-10k-dataset) (the <strong>largest</strong> realistic infrared small target detection dataset, which consists of <strong>1,001</strong> manually labeled realistic images with various target shapes, different target sizes, and rich clutter back-grounds from diverse scenes.), the train code and the test code will come soon. 
> 

[//]: # (> [2021-12-06]: Publish the face mask of the training set and P3M-500-P validation set of [<strong>P3M-10k</strong>]&#40;#ppt-setting-and-p3m-10k-dataset&#41; dataset.)
> 
> | Dataset | <p>Dataset Link<br>(Google Drive)</p> | <p>Dataset Link<br>(Baidu Wangpan 百度网盘)</p> |       Dataset Release Agreement              |
>| :----:| :----: |:----------------------------------------------------------------------------------------------------------------------:| :----: | 
>|IRSTD-1k|[Link](https://drive.google.com/file/d/1JoGDGF96v4CncKZprDnoIor0k1opaLZa/view?usp=sharing)|[Link](https://caiyun.139.com/m/i?0K5CIsRSEZZiU) (pw: VZtn)|                                               [Agreement (MIT License)]                                                |

[//]: # (>|P3M-10k facemask &#40;optional&#41;|[Link]&#40;https://drive.google.com/file/d/1I-71PbkWcivBv3ly60V0zvtYRd3ddyYs/view?usp=sharing&#41;|[Link]&#40;https://pan.baidu.com/s/1D9Kj_OIJbFTsqWfbMPzh_g&#41; &#40;pw: f772&#41;|[Agreement &#40;MIT License&#41;]&#40;https://jizhizili.github.io/files/p3m_dataset_agreement/P3M-10k_Dataset_Release_Agreement.pdf&#41;| )
>

[//]: # (> [2021-11-20]: Publish the <a href="#inference-code---how-to-test-on-your-images">inference code</a> and the pretrained model &#40;[Google Drive]&#40;https://drive.google.com/uc?export=download&id=1smX2YQGIpzKbfwDYHAwete00a_YMwoG1&#41; | [Baidu Wangpan &#40;pw: 2308&#41;]&#40;https://pan.baidu.com/s/1zGF3qnnD8qpI-Z5Nz0TDGA&#41;&#41; that can be used to test on your own privacy-preserving or normal portrait images. Some test results on P3M-10k can be viewed from this [demo page]&#40;https://github.com/JizhiziLi/P3M/tree/master/demo&#41;.)

!!!!
Code is available at master-branch
