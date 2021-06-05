# CV_skeleton_provider
![LISENCE](https://img.shields.io/github/license/LiiNen/CV_skeleton_provider)
![release](https://img.shields.io/github/v/release/LiiNen/CV_skeleton_provider)
![release_date](https://img.shields.io/github/release-date/LiiNen/CV_skeleton_provider)
![last_commit](https://img.shields.io/github/last-commit/LiiNen/CV_skeleton_provider)
![dependency](https://img.shields.io/librariesio/github/LiiNen/CV_skeleton_provider)
![contributors](https://img.shields.io/github/contributors/LiiNen/CV_skeleton_provider)
<br>
![pypi](https://img.shields.io/pypi/v/CV_skeleton_provider)
![python](https://img.shields.io/pypi/pyversions/CV_skeleton_provider)
![implementation](https://img.shields.io/pypi/implementation/CV_skeleton_provider)
![wheel](https://img.shields.io/pypi/wheel/CV_skeleton_provider)

## OpenCV Project
CV_skeleton_provider project made for detecting a skeleton structure easily. <br>
This project contains how to use pretrained models and analize video with frames, using openCV. <br>
For solving some discomports, we make it from the beginning and providing argparse to use it easily. <br>
Also providing img preprocessing methods, you can use it with simple options when you use specific imgs <br>
<br>

## How to use
! prototxt file & caffemodel file are not included in both pypi package & project <br>
! you should download these pre-trained models in website, and you can easily find it <br>
### using pip package (recommended)
1. install pypi package
```bash
$ pip install CV_skeleton_provider
```
2. python code
```python
from CV_skeleton_provider.SkeletonProvider import defaultDict, skprovider
skprovider(defaultDict()) # show skeleton image over default img
```
3. custom parameter (when customizing)
```python
input_dict = defaultDict()
input_dict['source']       =  './example.jpeg'   # input filepath. img(jpg, jpeg, png) or video(mp4, avi, mkv) supported
input_dict['output']       =  './output'         # output filepath(exclude format). file format will be set by automatically.
input_dict['option']       =  'skl'              # s for skeleton, k for keypoints, l for label. if string include these char, show it
input_dict['exclude']      =  []                 # 0~17 interger list. that point will not be shown on result.
input_dict['thres']        =  0.1                # threshold (float)
input_dict['gray']         =  False              # using grayscale (bool)
input_dict['back']         =  False              # remove background (bool)
input_dict['selectRect']   =  False              # when removing background, you can set the object(human) size by drag img (bool)
input_dict['autolocation'] =  False              # when removing background, code will automatically 'detect' human (bool)
input_dict['comp']         =  1                  # for only video. video frame will be reduced to 1/comp (int)
input_dict['gamma']        =  -1                 # img preprocessing gamma value. under 0 means not processing gamma (float)
input_dict['b_propo']      =  False              # check black proportion and preprocessing reducing black part (bool)
input_dict['show']         =  False              # (only for img) if you want to show img with cv2.imshow, set to True (bool)
input_dict['save']         =  True               # (only for img) if you want not to save file in local, set to False (bool)
input_dict['proto']        =  './pose/coco/pose_deploy_linevec.prototxt'    # prototxt filepath
input_dict['weight']       =  './pose/coco/pose_iter_440000.caffemodel'     # caffemodel filepath
skprovider(input_dict)
```

### using clone project
```bash
$ git clone https://github.com/LiiNen/CV_skeleton_provider.git
$ pip install -r ./CV_skeleton_provider/requirements.txt
$ cd ./CV_skeleton_provider/CV_skeleton_provider
$ python ./SkeletonProvider.py # show skeleton image over default img
```
you can use argparse with SkeletonProvider.py to change parameter <br>
check it with following command in CLI
```bash
$ python SkeletonProvider.py --help
```
! as mentioned, prototxt & caffemodel must be existed. followings are default path <br>
```
./CV_skeleton_provider/CV_skeleton_provider/pose/coco/pose_deploy_linevec.prototxt
./CV_skeleton_provider/CV_skeleton_provider/pose/coco/pose_iter_440000.caffemodel
```
<br>

## Main contributors
<table>
  <tr><td align='center'>Github</td><td align='center'>Contact</td></tr>
  <tr><td><a href="https://github.com/LiiNen"><img src="http://img.shields.io/badge/LiiNen-655ced?style=social&logo=github"/></td>
  <td><a href="mailto:kjeonghoon065@gmail.com"><img src="https://img.shields.io/static/v1?label=&message=kjeonghoon065@gmail.com&color=green&style=flat-square&logo=gmail"></td></tr>
  <tr><td><a href="https://github.com/Seunggyun98"><img src="http://img.shields.io/badge/Seunggyun98-655ced?style=social&logo=github"/></td>
  <td><a href="mailto:dmstmdrbs98@gmail.com"><img src="https://img.shields.io/static/v1?label=&message=dmstmdrbs98@gmail.com&color=green&style=flat-square&logo=gmail"></td></tr>
  <tr><td><a href="https://github.com/Coreight98"><img src="http://img.shields.io/badge/Coreight98-655ced?style=social&logo=github"/></td>
  <td><a href="mailto:maxcha98@ajou.ac.kr"><img src="https://img.shields.io/static/v1?label=&message=maxcha98@ajou.ac.kr&color=green&style=flat-square&logo=gmail"></td></tr>
