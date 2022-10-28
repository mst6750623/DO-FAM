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