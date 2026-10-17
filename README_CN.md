<p align="center" width="100%">
<img src="./images/logo.png" alt="AnomalyGPT_logo" style="width: 40%; min-width: 300px; display: block; margin: auto;" />
</p>

# AnomalyGPT：基于大型视觉-语言模型的工业异常检测

![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-red.svg)

<p align="left">
   🌐 <a href="https://anomalygpt.github.io" target="_blank">项目主页</a> • 🤗 <a href="https://huggingface.co/spaces/FantasticGNU/AnomalyGPT" target="_blank">在线 Demo</a> • 📃 <a href="https://arxiv.org/abs/2308.15366" target="_blank">论文</a> • 🤖 <a href="https://huggingface.co/FantasticGNU/AnomalyGPT" target="_blank">模型权重</a> • 📹 <a href="https://www.youtube.com/watch?v=lcxBfy0YnNA" target="_blank">视频</a>
</p>

作者：Zhaopeng Gu, Bingke Zhu, Guibo Zhu, Yingying Chen, Ming Tang, Jinqiao Wang（中科院自动化所）

> 本文件为官方英文 README 的中文翻译（译注均已明确标注）。原版：[README.md](./README.md)

****

<span id='all_catelogue'/>

## 目录：

* <a href='#introduction'>1. 简介</a>
* <a href='#environment'>2. 运行 AnomalyGPT Demo</a>
    * <a href='#install_environment'>2.1 环境安装</a>
    * <a href='#download_imagebind_model'>2.2 准备 ImageBind 权重</a>
    * <a href='#download_vicuna_model'>2.3 准备 Vicuna 权重</a>
    * <a href='#download_anomalygpt'>2.4 准备 AnomalyGPT 的 Delta 权重</a>
    * <a href='#running_demo'>2.5 部署 Demo</a>
* <a href='#train_anomalygpt'>3. 训练你自己的 AnomalyGPT</a>
    * <a href='#data_preparation'>3.1 数据准备</a>
    * <a href='#training_configurations'>3.2 训练配置</a>
    * <a href='#model_training'>3.3 训练 AnomalyGPT</a>
* <a href='#examples'>4. 效果示例</a>
* <a href='#license'>许可证</a>
* <a href='#citation'>引用</a>
* <a href='#acknowledgments'>致谢</a>

****

<span id='introduction'/>

### 1. 简介 <a href='#all_catelogue'>[返回目录]</a>

<p align="center" width="100%">
<img src="./images/compare.png" alt="AnomalyGPT_logo" style="width: 80%; min-width: 400px; display: block; margin: auto;" />
</p>

**AnomalyGPT** 是首个基于大型视觉-语言模型（LVLM）的工业异常检测（IAD）方法，能够**无需人工设定阈值**即可检测工业图像中的异常。现有 IAD 方法只能输出异常分数并需要手工设置判决阈值，而现有 LVLM 又无法检测图像中的异常。AnomalyGPT 不仅能指出异常的有无与位置，还能提供关于该图像的信息（对话式回答）。

<img src="./images/AnomalyGPT.png" alt="AnomalyGPT" style="zoom:100%;" />

我们利用预训练图像编码器和大型语言模型（LLM），通过**模拟异常数据**（NSA 合成）将 IAD 图像与其对应的文本描述对齐。具体地：采用一个轻量的、基于视觉-文本特征匹配的图像解码器获得异常定位结果；设计一个 **Prompt Learner（提示学习器）** 向 LLM 提供细粒度语义，并通过 prompt 嵌入对 LVLM 进行微调。在仅提供少量正常样本的情况下，本方法还能对**此前未见过的物品**进行异常检测（few-shot 设定）。

****

<span id='environment'/>

### 2. 运行 AnomalyGPT Demo <a href='#all_catelogue'>[返回目录]</a>

<span id='install_environment'/>

#### 2.1 环境安装

将仓库克隆到本地：

```
git clone https://github.com/CASIA-IVA-Lab/AnomalyGPT.git
```

安装所需依赖包：

```
pip install -r requirements.txt
```

<span id='download_imagebind_model'/>

#### 2.2 准备 ImageBind 权重：

从[这个链接](https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth)下载预训练 ImageBind 模型，下载完成后将文件（imagebind_huge.pth）放入 [[./pretrained_ckpt/imagebind_ckpt/]](./pretrained_ckpt/imagebind_ckpt/) 目录。

<span id='download_vicuna_model'/>

#### 2.3 准备 Vicuna 权重：

准备预训练 Vicuna 模型，请按照 [[此处]](./pretrained_ckpt#1-prepare-vicuna-checkpoint)（pretrained_ckpt/README.md）的说明操作。

> **译注**：pretrained_ckpt/README.md 中的流程（向 Meta 申请 LLaMA 权重 + 下载 Vicuna delta + FastChat 合并）是 2023 年 LLaMA 许可证限制时期的旧流程。如今 HuggingFace 上 `lmsys/vicuna-7b-v0` 已直接提供合并后的完整权重，可下载后放入 `pretrained_ckpt/vicuna_ckpt/7b_v0/`，无需再走申请与合并步骤；如该仓库无法访问，再按官方 README 的 delta 流程操作。

<span id='download_anomalygpt'/>

#### 2.4 准备 AnomalyGPT 的 Delta 权重：

我们使用 [PandaGPT](https://github.com/yxuansu/PandaGPT) 的预训练参数初始化模型。下表列出以不同策略训练的 PandaGPT 权重。受算力限制，我们的实验和在线 Demo 使用 Vicuna-7B 和 `openllmplayground/pandagpt_7b_max_len_1024`；切换到 Vicuna-13B 预期可获得更好效果。

| **基础语言模型** | **最大序列长度** | **HuggingFace Delta 权重地址** |
| :---------------: | :--------------: | :----------------------------: |
|  Vicuna-7B (v0)   |       512        | [openllmplayground/pandagpt_7b_max_len_512](https://huggingface.co/openllmplayground/pandagpt_7b_max_len_512) |
|  Vicuna-7B (v0)   |       1024       | [openllmplayground/pandagpt_7b_max_len_1024](https://huggingface.co/openllmplayground/pandagpt_7b_max_len_1024) |
|  Vicuna-13B (v0)  |       256        | [openllmplayground/pandagpt_13b_max_len_256](https://huggingface.co/openllmplayground/pandagpt_13b_max_len_256) |
|  Vicuna-13B (v0)  |       400        | [openllmplayground/pandagpt_13b_max_len_400](https://huggingface.co/openllmplayground/pandagpt_13b_max_len_400) |

请将下载好的 7B/13B delta 权重文件（pytorch_model.pt）放入 [./pretrained_ckpt/pandagpt_ckpt/7b/](./pretrained_ckpt/pandagpt_ckpt/7b/) 或 [./pretrained_ckpt/pandagpt_ckpt/13b/](./pretrained_ckpt/pandagpt_ckpt/13b/) 目录。

之后，从下表下载 AnomalyGPT 权重：

|                        设定与数据集                        |              权重地址              |
| :--------------------------------------------------------: | :--------------------------------: |
|                    MVTec-AD 无监督训练                     | [AnomalyGPT/train_mvtec](https://huggingface.co/FantasticGNU/AnomalyGPT/blob/main/train_mvtec/pytorch_model.pt) |
|                      VisA 无监督训练                       | [AnomalyGPT/train_visa](https://huggingface.co/FantasticGNU/AnomalyGPT/blob/main/train_visa/pytorch_model.pt) |
| MVTec-AD、VisA、MVTec-LOCO-AD、CrackForest 有监督联合训练  | [AnomalyGPT/train_supervised](https://huggingface.co/FantasticGNU/AnomalyGPT/blob/main/train_supervised/pytorch_model.pt) |

下载完成后，将 AnomalyGPT 权重放入 [./code/ckpt/](./code/ckpt/) 目录。

在我们的[在线 Demo](https://huggingface.co/spaces/FantasticGNU/AnomalyGPT) 中，默认使用有监督设定的模型以获得更好的用户体验。你也可以在本地尝试其他权重。

<span id='running_demo'/>

#### 2.5 部署 Demo

完成上述步骤后，即可在本地运行 Demo：

```bash
cd ./code/
python web_demo.py
```

****

<span id='train_anomalygpt'/>

### 3. 训练你自己的 AnomalyGPT <a href='#all_catelogue'>[返回目录]</a>

**前提条件**：训练模型之前，请确认环境已正确安装，且 ImageBind、Vicuna、PandaGPT 的权重均已下载就绪。

<span id='data_preparation'/>

#### 3.1 数据准备：

MVTec-AD 数据集从[[这个链接]](https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads)下载，VisA 从[[这个链接]](https://github.com/amazon-science/spot-diff)下载。PandaGPT 的预训练数据可从[[这里]](https://huggingface.co/datasets/openllmplayground/pandagpt_visual_instruction_dataset/tree/main)下载（用于交替训练防止灾难性遗忘）。下载完成后，将数据放入 [[./data]](./data/) 目录。

[[./data]](./data/) 目录结构应如下所示：

```
data
|---pandagpt4_visual_instruction_data.json
|---images
|-----|-- ...
|---mvtec_anomaly_detection
|-----|-- bottle
|-----|-----|----- ground_truth
|-----|-----|----- test
|-----|-----|----- train
|-----|-- capsule
|-----|-- ...
|----VisA
|-----|-- split_csv
|-----|-----|--- 1cls.csv
|-----|-----|--- ...
|-----|-- candle
|-----|-----|--- Data
|-----|-----|-----|----- Images
|-----|-----|-----|--------|------ Anomaly
|-----|-----|-----|--------|------ Normal
|-----|-----|-----|----- Masks
|-----|-----|-----|--------|------ Anomaly
|-----|-----|--- image_anno.csv
|-----|-- capsules
|-----|-----|----- ...
```

> **译注**：MVTec-AD 与 VisA 均需在官网提交使用申请（有审批周期），建议尽早提交。MVTec-LOCO-AD（有监督训练用）同样需在 MVTec 官网申请。

<span id='training_configurations'/>

#### 3.2 训练配置：

下表为我们实验中使用的训练超参数。超参数基于我们的算力条件（2 × RTX 3090）选取。

| **基础语言模型** | **训练轮数** | **批大小** | **学习率** | **最大序列长度** |
| :--------------: | :----------: | :--------: | :--------: | :--------------: |
|    Vicuna-7B     |      50      |     16     |    1e-3    |       1024       |

<span id='model_training'/>

#### 3.3 训练 AnomalyGPT：

在 MVTec-AD 数据集上训练 AnomalyGPT，请运行以下命令：

```yaml
cd ./code
bash ./scripts/train_mvtec.sh
```

训练脚本的关键参数如下：

* `--data_path`：json 文件 `pandagpt4_visual_instruction_data.json` 的路径。
* `--image_root_path`：PandaGPT 训练图像的根目录。
* `--imagebind_ckpt_path`：ImageBind 权重路径。
* `--vicuna_ckpt_path`：预训练 Vicuna 权重所在目录。
* `--max_tgt_len`：训练样本的最大序列长度。
* `--save_path`：保存训练得到的 delta 权重的目录，该目录会自动创建。
* `--log_path`：保存日志的目录，该目录会自动创建。

注意：训练轮数在 [./code/config/openllama_peft.yaml](./code/config/openllama_peft.yaml) 文件的 `epochs` 参数中设置，学习率在 [./code/dsconfig/openllama_peft_stage_1.json](./code/dsconfig/openllama_peft_stage_1.json) 中设置。

****

<span id='examples'/>

### 4. 效果示例 <a href='#all_catelogue'>[返回目录]</a>

![](./images/demo_1.png)
<h4 align='center'>带裂缝的混凝土图像。</h4>

****
![](./images/demo_5.png)
<h4 align='center'>有裂缝的胶囊。</h4>

****
![](./images/demo_8.png)
<h4 align='center'>被割伤的榛子图像。</h4>

****
![](./images/demo_7.png)
<h4 align='center'>损坏的瓶子。</h4>

****
![](./images/demo_2.png)
<h4 align='center'>正常地毯的照片。</h4>

****
![](./images/demo_4.png)
<h4 align='center'>有缺陷的木块照片。</h4>

****
![](./images/demo_3.png)
<h4 align='center'>正常的织物。</h4>

****

<span id='license'/>

### 许可证 <a href='#all_catelogue'>[返回目录]</a>

AnomalyGPT 采用 [CC BY-NC-SA 4.0 许可证](./LICENSE)（署名-非商业性使用-相同方式共享）。

> **译注**：毕业论文/学术研究用途一般符合非商业条款，但需注意署名与相同方式共享的要求。

****

<span id='citation'/>

### 引用 <a href='#all_catelogue'>[返回目录]</a>

如果 AnomalyGPT 对你的研究或应用有帮助，请使用以下 BibTeX 引用：

```
@article{gu2023anomalyagpt,
  title={AnomalyGPT: Detecting Industrial Anomalies using Large Vision-Language Models},
  author={Gu, Zhaopeng and Zhu, Bingke and Zhu, Guibo and Chen, Yingying and Tang, Ming and Wang, Jinqiao},
  journal={arXiv preprint arXiv:2308.15366},
  year={2023}
}
```

****

<span id='acknowledgments'/>

### 致谢 <a href='#all_catelogue'>[返回目录]</a>

本项目借用（复用）了 [PandaGPT](https://github.com/yxuansu/PandaGPT) 的部分代码与预训练权重，感谢他们的出色工作！

[![Star History Chart](https://api.star-history.com/svg?repos=CASIA-IVA-Lab/AnomalyGPT&type=Date)](https://star-history.com/#CASIA-IVA-Lab/AnomalyGPT&Date)
