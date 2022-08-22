# Style transfer

## 1. Introduction
 paper : https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf
 논문에서는 어떤 이미지의 style을 다른 이미지로 전달하는것을 textuer transfer 문제라고 한다.
 기존의 texture transfer에서도 가능했지만, 이는 low-level feature만을 사용해서 문제점이 있었다.
 이 연구에서는 high-level feature에서 high-level semantic feature을 추출하여 사용할 수 있기 때문에 좀 더 의미있는 style transfer이 가능하다고 한다.
 ![image](https://user-images.githubusercontent.com/102507688/185020424-a3dfb0a1-a830-4cec-a1ab-8b9c1ca5f537.png)

## 2. Deep image representations
  * base : normalised된 16개의 convolutional layers, 5개의 pooling layers로 구성된 VGG19 network를 baisis model로 사용, fully connected layer들을 사용하지 않았다.
  
  ### Content representation
   $N_l : 한 layer의 filter 수(channel 수)$   
   $M_l : feature map의 내적$  
   $F^l_{i,g} : F \in R^{N_lXM_l}$  
   $p : 원본 이미지$  
   $x : 생성된 이미지$     
   $P^l: 원본 이미지 layer l feature map $  
   $F^l : 생성된 이미지 layer l feature map$  
 
 
 Content Loss : 
 
 ![image](https://user-images.githubusercontent.com/102507688/185025370-7d4d1b91-47ea-4185-830b-a19f23f0683f.png)  
 
 
 ![image](https://user-images.githubusercontent.com/102507688/185032861-b5f1ee9b-5f13-4ea6-b130-6598f049b9c1.png)

 network의 higher layer는 이미지의 물체나 배치에 대한 high-level content를 잡아낸다. 그러나 정확한 픽셀값을 기대하기는 어렵다.
 이 연구에서는 higher layer의 feature map을 content representation에 활용한다.
 
 ### Style representation
 * input 이미지의 style representation을 얻으려면, texture information을 잡아내는 feature space를 활용해야한다.
 이 때 활용하는 것이 Gramm matrix이다.
 
 $G^l \in R^{N_lXM_l}$  
 $G^l_{ij} = \sigma F^l_{ik}F^l_{jk}$  
 
 
 * G는 layer l에서 vectorized feature map i, j 의 내적을 한 것이다.
 원본 이미지의 Gram matrics와 생성된 이미지의 Gram matrics 간 mean-squared distance를 최소화 하도록 Loss를 구성한다.
 
$a : style original image$  
$x : 생성된 image$  
$A^l : style original image layer l feature map$  
$F^l : 생성된 image layer l feature map$  

![image](https://user-images.githubusercontent.com/102507688/185036722-2254f37f-1a7f-4de8-ae54-0180f0266af3.png)


![image](https://user-images.githubusercontent.com/102507688/185036763-0e56b1c0-5892-489f-a057-dca43e15273c.png)  


![image](https://user-images.githubusercontent.com/102507688/185040342-b20763a3-3d21-4ffc-bae2-2a2ed8cdd2d4.png)  


* 위 2번째 수식의 w는 layer 별 weighting factor이다.
* 결과적으로 style original image의 정보를 가진 a와 content original image의 정보를 가진 p를 합성하여 input image x를 얻고자 함이다.  

![image](https://user-images.githubusercontent.com/102507688/185041133-820eca2b-5c7b-459a-b33d-d89984c00e16.png)  

$\alpha, \beta : content, style reconstruction's weighting factor$  
*optimizer : L-BFGS

![image](https://user-images.githubusercontent.com/102507688/185040500-73c56268-395d-4d34-8189-b34c24f993c0.png)  
*알파와 베타값을 통해 content, style 각각의 반영 정도를 조정할 수 있다.

## 3. Result
  

* content image, style image가 각각 존재하고, 생성할 이미지 x는 white noise 상태에서 content, style information을 합성하여 얻어낸다.
* pretrained VGG net을 활용하고, 이때의 학습은 VGG net이 아니라 x가 backpropagation을 통해 값을 찾아가는 것을 말한다.  
* Content image example  
![image](https://user-images.githubusercontent.com/102507688/185042655-fcf83688-df7a-4c10-bf19-3c00a7eb077f.png)  

* Style image example  
![image](https://user-images.githubusercontent.com/102507688/185042700-74cbffdd-8363-4666-90a7-48cc66aa5235.png)  

* Combined image example  
  
![image](https://user-images.githubusercontent.com/102507688/185041839-c822c185-bd80-4c91-8c6e-f999fb5a67aa.png)
