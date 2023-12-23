# ICCV 2023 - AdaptGuard: Defending Against Universal Attacks for Model Adaptation

## [**[ICCV-2023] AdaptGuard: Defending Against Universal Attacks for Model Adaptation**](https://arxiv.org/abs/2303.10594)

### Framework:  

<img src="figs/framework.png" width="600"/>

<br/>
<br/>

Our implementation is based on [SHOT](https://github.com/tim-learn/SHOT).

### Prerequisites:
- python == 3.8.5
- pytorch == 1.12.1
- torchvision == 0.13.1

### Dataset:
Please manually download the datasets [office](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view), [office-home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view), [DomainNet](https://ai.bu.edu/M3SDA/), and modify the path of images in each '.txt' under the folder 'data/'.

### Training:
1. ##### Training Source Models with Embeded Backdoor
	```python
	python -u image_source_backdoor.py --trte val --output 'ckps/source/' --da uda --gpu_id 0 --dset office-home --max_epoch 50 --s 0 --seed 2020 --backdoor 'Blended'
	```
   
2. ##### Preprocessing Models with AdaptGuard for Source-Side Defense Against Universal Attacks
    ```python
    python -u image_ft_adaptguard.py --dset office-home --gpu_id $gpu  --da uda --s 0 --t 1 --kd_max_epoch 50 --output_src 'ckps/source/' --output 'ckps/source_adaptguard/' --seed 2020 --backdoor 'Blended'
	```
	
3. ##### Unsupervised Model Adaptation using existing algorithms (e.g., SHOT) for Enhanced Performance
	```python
	python -u image_target.py --cls_par 0.3 --da uda --dset office-home --gpu_id $gpu --s 0 --t 1 --output_src 'ckps/source_adaptguard/' --output 'ckps/target_adaptguard/' --seed 2020 --backdoor 'Blended'
	```
   
### Citation
If you find this code useful for your research, please cite our papers

```

@inproceedings{sheng2023adaptguard,
  title={AdaptGuard: Defending Against Universal Attacks for Model Adaptation},
  author={Sheng, Lijun and Liang, Jian and He, Ran and Wang, Zilei and Tan, Tieniu},
  booktitle={International Conference on Computer Vision (ICCV)}, 
  year={2023}
}

```

### Contact
- [slj0728@mail.ustc.edu.cn](mailto:slj0728@mail.ustc.edu.cn)
- [liangjian92@gmail.com](mailto:liangjian92@gmail.com)


