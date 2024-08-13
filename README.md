Repo for ACL 2024 paper: 

# Never Lost in the Middle: Mastering Long-Context Question Answering with Position-Agnostic Decompositional Training 

The code and data for each experiment are in LongBench, Lost_Retrieval and RGB directories. Original results for models in the paper are in ** results ** directories. 

Please download hf models and change parameters in codes before running.

Ziya-Reader: <https://huggingface.co/IDEA-CCNL/Ziya-Reader-13B-v1.0>

Paper url: <https://aclanthology.org/2024.acl-long.736.pdf>

If there is need for training data of the PAM QA stage, please fill in the table below and sent it to hejunqing2013@gmail.com.

https://1drv.ms/x/c/389d4fd6b214d5a8/EUkuVymjKL5Bu13qIzugX9ABTSV6Jzas_ExKw4lYzVGDyQ?e=duwJaP&nav=MTVfezAwMDAwMDAwLTAwMDEtMDAwMC0wMDAwLTAwMDAwMDAwMDAwMH0

## please cite

```
@inproceedings{he-etal-2024-never,
    title = "Never Lost in the Middle: Mastering Long-Context Question Answering with Position-Agnostic Decompositional Training",
    author = "He, Junqing  and
      Pan, Kunhao  and
      Dong, Xiaoqun  and
      Song, Zhuoyang  and
      YiBo, Liu  and
      Qianguo, Sun  and
      Liang, Yuxin  and
      Wang, Hao  and
      Zhang, Enming  and
      Zhang, Jiaxing",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.736",
    pages = "13628--13642",
    abstract = "While large language models (LLMs) are equipped with longer text input capabilities than before, they are struggling to seek correct information in long contexts. The {``}lost in the middle{''} problem challenges most LLMs, referring to the dramatic decline in accuracy when correct information is located in the middle. To overcome this crucial issue, this paper proposes to enhance the information searching and reflection ability of LLMs in long contexts via specially designed tasks called Position-Agnostic Multi-step QA (PAM QA). Trained in this task, our model excels in focusing more precisely on the desired information. Experimental results show substantial improvement in Multi-doc QA and other benchmarks, superior to state-of-the-art models by 13.7{\%} absolute gain in shuffled settings, by 21.5{\%} in passage retrieval task. We release our model and code to promote related research in the community.",
}

```
```
@misc{he2023lostmiddleimprovinglarge,
      title={Never Lost in the Middle: Improving Large Language Models via Attention Strengthening Question Answering}, 
      author={Junqing He and Kunhao Pan and Xiaoqun Dong and Zhuoyang Song and Yibo Liu and Yuxin Liang and Hao Wang and Qianguo Sun and Songxin Zhang and Zejian Xie and Jiaxing Zhang},
      year={2023},
      eprint={2311.09198},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2311.09198},
}
```
