# face-pix2pixHD face editing

In this work, I achieved face editing with pix2pixHD. In my work, I used a label mask to generate a face and by using Adain normalization to guide the style of generated image.
## Prerequisites

* Python 3
* Ptyorch 0.41+
* NVIDIA GPU + CUDA CuDNN
* Opencv

## Results

Face style transfer:

<img src="https://github.com/577816569/face-pix2pixHD/blob/master/images/000085.jpg" width = "150" height = "150" alt="Original style image" align=center /><img src="https://github.com/577816569/face-pix2pixHD/blob/master/images/dsdsd.jpg" width = "150" height = "150" alt="Original style image" align=center />
<img src="https://github.com/577816569/face-pix2pixHD/blob/master/images/000014.jpg" width = "150" height = "150" alt="Original style image" align=center />
<img src="https://github.com/577816569/face-pix2pixHD/blob/master/images/1.jpg" width = "150" height = "150" alt="Original style image" align=center />

&emsp;&emsp;&emsp;Original  image &emsp;&emsp;&emsp;                  Mask&emsp;&emsp;&emsp;&emsp;&emsp;    Style&emsp;&emsp;&emsp;&emsp;    Result

Face editing:

<img src="https://github.com/577816569/face-pix2pixHD/blob/master/images/000125.jpg" width = "150" height = "150" alt="Original style image" align=center /><img src="https://github.com/577816569/face-pix2pixHD/blob/master/images/312.jpg" width = "150" height = "150" alt="Original style image" align=center />
<img src="https://github.com/577816569/face-pix2pixHD/blob/master/images/4343.jpg" width = "150" height = "150" alt="Original style image" align=center />
<img src="https://github.com/577816569/face-pix2pixHD/blob/master/images/fdf.jpg" width = "150" height = "150" alt="Original style image" align=center />

&emsp;&emsp;&emsp;Original  image &emsp;&emsp;&emsp;                     Edited Mask&emsp;&emsp;&emsp;Original Mask&emsp;&emsp;&emsp;Result

## Reference
Pix2pixHD: [https://github.com/NVIDIA/pix2pixHD](https://github.com/NVIDIA/pix2pixHD)
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE0Mzg5ODIwODksMTI0NzQ1NDE4MSwtMT
czOTAzMjY0LDE5NDQ5MDE2NjEsLTI5Mzc4ODc0Miw5NTAwMDI5
NDIsMTkwMjA5Nzg4NSwxMzMxODI5OTQzXX0=
-->