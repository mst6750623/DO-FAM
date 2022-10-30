# DO-FAM
Official implementation of DO-FAM

## Quickstart
### Pretrained models
Please download the pretrained models and move them to ./pretrained_models/

Note: when test, there is no need to use classifier, so you don't need to download it.

|  Model   | Description  |
|  ----  | ----  |
| [DOLL](https://drive.google.com/drive/folders/1N2B1UFjqwRNK5ZydCIa_6QEYKiH6PGYE)  | Pretrained models of DOLL on attribute *Gender, Eyeglasses, Age, Smiling*|
| [classifier](https://drive.google.com/drive/folders/1YWxoMKx6k9XCZZfCLJNSIUYOH0JNLgFF)  | Pretrained models of Latent Classifier on hyperstyle latent codes for 40 attributes|
| [hyperstyle+stylegan](https://drive.google.com/file/d/1C3dEIIH1y8w1-zQMCyx7rDF0ndswSXh4/view) | Pretrained models released by [hyperstyle](https://github.com/yuval-alaluf/hyperstyle) |

### Test datas
Please download the test dataset(30 images with corresponding latent codes and weight deltas) in [this link](https://drive.google.com/drive/folders/1DkDyM_kHJrKjo4C37irwdrEWnZpNfK2C) and put them in ./test_data/



## Test on own images
for lower GPU usage, we seperate the editing process to 2 steps: inversion step and editing step

### Step1: run inversion to get latent codes and weights deltas
**1.**  Modify data_paths.test_image to your own image paths in ./configs/path_config.py.

**2.** Run the following command to get the latent codes and weights of images(will take 7G RAM in GPU):
'''python
python generate_latents_and_weights.py  --exp_dir './test_data/' --save_weight_deltas --gpu '0'
'''

### Step2: run editing step to get result
**1.** Modify data_paths.test_latent and data_paths.test_weights_delta in ./configs/path_config.py  to your own in **Step1**

**2.** Run the following command to get the edited image(will take 8G RAM in GPU):
'''python
python test.py --attribute 'Smiling' --coeff_min -1.5 --coeff_max 1.5 --step 0.5 --gpu '0'
'''
You can choose 'Eyeglasses', 'Gender', 'Smiling' and 'Age' to manipulate.