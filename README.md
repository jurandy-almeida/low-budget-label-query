# Low-budget Label Query through Domain Alignment Enforcement (CVIU 2022)


Official PyTorch github repository for the paper [Low-budget Label Query through Domain Alignment Enforcement](https://doi.org/10.1016/j.cviu.2022.103485) published in the Computer Vision and Image Understanding (**CVIU**) journal, 2022.

### Prerequisites
* Pytorch 1.4.0
* Python 3.5

### Usage
- Install the required packages:

```bash
pip install -r requirements.txt
```

- CIFAR-10 -> STL: 

```bash
./digits.sh cifar9 stl9 dialnet
```

- STL -> CIFAR-10: 

```bash
./digits.sh stl cifar9 dialnet
```

- SVHN -> MNIST: 

```bash
./digits.sh svhn mnist dialnet
```

- MNIST -> USPS: 

```bash
./digits.sh mnist usps dialnet
```

- USPS -> MNIST: 

```bash
./digits.sh usps mnist dialnet
```

- ImageNet -> STL: 

```bash
./imagenet.sh stl9 resnet50
```

- ImageNet -> CIFAR-10: 

```bash
./imagenet.sh cifar9 resnet50
```

- ImageNet -> MNIST: 

```bash
./imagenet.sh mnist resnet50
```

- ImageNet -> USPS: 

```bash
./imagenet.sh usps resnet50
```

- ImageNet -> SVHN: 

```bash
./imagenet.sh svhn resnet50
```

- Office-31: To run the experiments on the [Office-31](https://faculty.cc.gatech.edu/~judy/domainadapt/) dataset first you need to download the dataset from [this](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view?resourcekey=0-gNMHVtZfRAyO_t2_WrOunA) page. 

```bash
for SOURCE in amazon dslr webcam
do
	for TARGET in amazon dslr webcam
	do
		if [[ "$SOURCE" != "$TARGET" ]]
		then
			./office-31.sh "$SOURCE" "$TARGET" resnet50
		fi
	done
done
```

- Office-Home: To run the experiments on the [OfficeHome](http://hemanthdv.org/OfficeHome-Dataset/) dataset first you need to download the dataset from [this](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw) page. 

```bash
for SOURCE in Art Clipart Product RealWorld
do
	for TARGET in Art Clipart Product RealWorld
	do
		if [[ "$SOURCE" != "$TARGET" ]]
		then
			./office-home.sh "$SOURCE" "$TARGET" resnet50
		fi
	done
done
```

If you find this code useful for your research, please cite our paper:

```tex
@Article{journals-cviu-saltori-rsa-22,
  author    = {Cristiano Saltori and
               Paolo Rota and
               Nicu Sebe and
               Jurandy Almeida},
  doi       = {10.1016/j.cviu.2022.103485},
  journal   = {Computer Vision and Image Understanding},
  pages     = {103485},
  title     = {Low-budget Label Query through Domain Alignment Enforcement},
  volume    = {222},
  year      = {2022},
  publisher = {Springer}
}
```