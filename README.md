# OSMO: Online Specific Models for Occlusion in Multiple Object Tracking under Surveillance Scene

Created by Xu Gao, Peking University

### Introduction

**OSMO** is an online multi-object tracking framework which deals with two kinds of occlusion in the surveilalnce scene.

### Citation

If you find MDP_Tracking useful in your research, please consider citing:

     @inproceedings{xu2018osmo,
        author = {Gao, Xu and Jiang, Tingting},
        title = {OSMO: Online Specific Models for Occlusion in Multiple Object Tracking Under Surveillance Scene},
        booktitle = {Proceedings of the 26th ACM International Conference on Multimedia},
        year = {2018},
        pages = {201--210},
    } 

### Usage of the demo

1. Our project is writing in Python 2.7 and PyTorch 0.2. Please set the environment at first.

2. Download the CampusStone dataset from https://drive.google.com/open?id=1nL60VdWkOkvjkdkAY53cuuLbV6wBStSn and unzip the file.

3. Download the code using "git clone https://github.com/gaoxu1024/OSMO.git".

4. Put the unziped dataset into ./dataset/.

5. For testing, use "sh osmo_run_list.sh".

We provide our tracking results in the folder ./result/. Besides, we only present the CampusStone dataset here currently, since other three datasets may have some private problems with the government.

### References

[1] Xu Gao, and Tingting Jiang. OSMO: Online Specific Models for Occlusion in Multiple Object Tracking Under Surveillance Scene. Proceedings of the 26th ACM International Conference on Multimedia (MM), 2018.

### Contact

If you find any bug or issue of the software, please contact gaoxu1024 at pku dot edu dot cn
