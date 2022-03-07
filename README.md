## Commonality in Natural Images Rescues GANs: Pretraining GANs with Generic and Privacy-free Synthetic Data<br/>(CVPR 2022)
![teaser](./resrc/teaser.png)
> **_Caption_**

__Official pytorch implementation of "Commonality in Natural Images Rescues GANs: Pretraining GANs with Generic and Privacy-free Synthetic Data"__

> __[Commonality in Natural Images Rescues GANs: Pretraining GANs with Generic and Privacy-free Synthetic Data](TBA)__   
> Kyungjune Baek<sup>1</sup>*, Hyunjung Shim<sup>1</sup>  
> <sup>1</sup> Yonsei University  
>  
> __Absract__ _Transfer learning for GANs successfully improves generation performance under low-shot regimes. However, existing studies show that the pretrained model using a single benchmark dataset is not generalized to various target datasets. More importantly, the pretrained model can be vulnerable to copyright or privacy risks as membership inference attack advances. To resolve both issues, we propose an effective and unbiased data synthesizer, namely Primitives-PS, inspired by the generic characteristics of natural images. Specifically, we utilize 1) the generic statistics on the frequency magnitude spectrum, 2) the elementary shape (i.e., image composition via elementary shapes) for representing the structure information, and 3) the existence of saliency as prior. Since our synthesizer only considers the generic properties of natural images, the single model pretrained on our dataset can be consistently transferred to various target datasets, and even outperforms the previous methods pretrained with the natural images in terms of Fr\'echet inception distance. Extensive analysis, ablation study, and evaluations demonstrate that each component of our data synthesizer is effective, and provide insights on the desirable nature of the pretrained model for the transferability of GANs._

### Requirement 
__Library__
```
pip install -r requirements.txt
* 
```

## Citation
If you find this work useful for your research, please cite our paper:
```
@InProceedings{Baek_2022_CVPR,
    author    = {Baek, Kyungjune and Shim, Hyunjung},
    title     = {Commonality in Natural Images Rescues GANs: Pretraining GANs with Generic and Privacy-free Synthetic Data},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year      = {2022}
}
```
