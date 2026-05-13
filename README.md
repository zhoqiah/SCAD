# SCAD: Semantic Cluster-Aware Asymmetric Disentanglement for Multi-Domain Fake News Detection
This is an implementation.
## Dataset
The splited dataset (i.e., train, val, test) are in the `data` folder.
 
## Code
### Requirements
Refer to requirements.txt

You can run `pip install -r requirements.txt` to deploy the environment quickly.
### pretrained_model 
You can download pretrained model (Roberta) from https://drive.google.com/drive/folders/1y2k22iMG1i1f302NLf-bj7UEe9zwTwLR?usp=sharing and move all the files in the folder into the path `SCAD/pretrained_model/chinese_roberta_wwm_base_ext_pytorch`.
### Data Preparation
After you download the **Weibo21** dataset, move the `train.pkl`, `val.pkl` and `test.pkl` into the path `MDFEND-Weibo21/data`.
### Run
You can run this model through:
```python
python main.py --model_name scad --batchsize 32 --lr 0.0007
```
### Reference

