**Semi-supervise learning For Landmark Detection**
-
- This project is about applying semi-supervise deep learning technique for landmark detection. 
Currently, only facial landmark detection on **ALFW** dataset is available 
- The project is based on Cross Pseudo Supervision: https://github.com/charlesCXK/TorchSemiSeg


**Getting started**
- 
- Prepare the environment:

    ```
  pip install -r requirements.txt
  ```
- Download AFLW dataset, it should come with 3 folders ``0/ 2/ 3/`` containing images, and 2 annotations files ``AFLWinfo_release.mat``
and ``aflw-sqlite.pth``. Put all of them under the same directory, for example ``data/``


- run
  ```
  python prepare_annotations.py --root [data-directory]
  ```        
  for example:
  ```
  python prepare_annotations.py --root data/
  ```
  After this, there should be 2 new sub-folders under ``[data-directory]`` named ``train_preprocessed`` and ``test_preprocessed``


- run the following for splitting the labeled/unlabeled data again (optional)
  ```
  python split_dataset.py --train_path [path-to-train_preprocessed] --test_path [path-to-test_preprocessed]
  ```


- go to ``config/config.py``. Edit the following lines:
  ```
  C.train_annotations_path = '[path-to-train_preprocessed]/annotations'
  C.train_images_path = '[path-to-train_preprocessed]/images'
  C.test_annotations_path = '[path-to-test_preprocessed]/annotations'
  C.test_images_path = '[path-to-test_preprocessed]/images'
  ```
  and
  ```
  C.train_text_labeled = './data/split/train_labeled_[ratio].txt'
  C.train_text_unlabeled = './data/split/train_unlabeled_[ratio].txt'
  ```
  with ``[ratio]`` being the ratio of labeled/total data (can be ``1_8``, ``1_4``, or ``1_2``)


**Training**
-
- run (add flag ``--resume`` to continue training from last checkpoint)
  ```
  python train.py 
  ```

**Evaluation**
-
- run 
  ```
  python test_ema.py
  ```