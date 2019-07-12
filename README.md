# End-to-end waveform utterance enhancement for direct evaluation metrics optimization by fully convolutional neural networks (TASLP 2018)


### Introduction
This paper tries to solve the mismatch (as in Fig.1) between training objective function and evaluation metrics which are usually highly correlated to human perception. Due to the inconsistency, there is no guarantee that the trained model can provide optimal performance in applications. In this study, we propose an end-to-end utterance-based speech enhancement framework using fully convolutional neural networks (FCN) to reduce the gap between the model optimization and the evaluation criterion. Because of the utterance-based optimization, temporal correlation information of long speech segments, or even at the entire utterance level, can be considered to directly optimize perception-based objective functions.

### Major Contribution
1) Utterance-based waveform enhancement
2) Direct short-time objective intelligibility (STOI) score optimization (without any approximation)


For more details and evaluation results, please check out our  [paper](https://ieeexplore.ieee.org/document/8331910).

![teaser](https://github.com/JasonSWFu/End-to-end-waveform-utterance-enhancement/blob/master/images/Fig1_3.png)

Waveform enhancement process:

![teaser](https://github.com/JasonSWFu/End-to-end-waveform-utterance-enhancement/blob/master/images/t2.gif)

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

