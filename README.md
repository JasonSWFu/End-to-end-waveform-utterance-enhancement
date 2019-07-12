# End-to-end waveform utterance enhancement for direct evaluation metrics optimization by fully convolutional neural networks (TASLP 2018)


### Introduction
MetricGAN is a Generative Adversarial Networks (GAN) based black-box metric scores optimization method.
By associating the discriminator (D) with the metrics of interest, MetricGAN can be treated as an iterative
process between surrogate loss learning and generator learning as shown in the following figure.

This code (developed with Keras) applies MetricGAN to optimize PESQ or STOI score for Speech Enhancement.
It can be easily extended to optimize other metrics.

For more details and evaluation results, please check out our  [paper](https://ieeexplore.ieee.org/document/8331910).

![teaser](https://github.com/JasonSWFu/MetricGAN/blob/master/images/MetricGAN_learning.png)

### Dependencies:
* Python 2.7
* keras=1.1.0 (recommended)

### Note! 
For the STOI loss function optimization, please e-mail me.

### Citation

If you find the code useful in your research, please cite:
    
    @article{fu2018end,
      title={End-to-end waveform utterance enhancement for direct evaluation metrics optimization by fully convolutional neural   networks},
      author={Fu, Szu-Wei and Wang, Tao-Wei and Tsao, Yu and Lu, Xugang and Kawai, Hisashi},
      journal={IEEE/ACM Transactions on Audio, Speech and Language Processing (TASLP)},
      volume={26},
      number={9},
      pages={1570--1584},
      year={2018},
      publisher={IEEE Press}}
      
    @inproceedings{fu2017raw,
      title={Raw waveform-based speech enhancement by fully convolutional networks},
      author={Fu, Szu-Wei and Tsao, Yu and Lu, Xugang and Kawai, Hisashi},
      booktitle={2017 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)},
      pages={006--012},
      year={2017},
      organization={IEEE}}
    
### Contact

e-mail: jasonfu@iis.sinica.edu.tw or d04922007@ntu.edu.tw

