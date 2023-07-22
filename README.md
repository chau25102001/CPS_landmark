# Semi-supervise learning For Landmark Detection

- This project is about applying semi-supervise deep learning technique for landmark detection.
  Currently, only facial landmark detection on **ALFW** dataset is available
- The repo is based on Cross Pseudo Supervision: https://github.com/charlesCXK/TorchSemiSeg

# Getting started

- pip (recommended):

    ```
  pip install -r requirements.txt
  ```
- docker:
  
  Download this docker image from https://drive.google.com/file/d/1EaTSbKKj0I3ApwVUhXHOtZOPHomk0XSV/view?usp=sharing and run:
  ```
  docker load < path_to_docker_image
  ```

## Dataset

### AFLW-DA and SideFace-Da

These two datasets are not published yet, you need to contact the authors for access.

### AFLW-19

- Download AFLW dataset, it should come with 3 folders ``0/ 2/ 3/`` containing images, and 2 annotations
  files ``AFLWinfo_release.mat``
  and ``aflw-sqlite.pth``. Put all of them under the same directory, for example ``data/``

- run
  ```
  python prepare_annotations.py --root [data-directory]
  ```        
  for example:
  ```
  python prepare_annotations.py --root data/
  ```
  After this, there should be 2 new sub-folders under ``[data-directory]`` named ``train_preprocessed``
  and ``test_preprocessed``


- run the following for splitting the labeled/unlabeled data.
  ```
  python split_dataset.py --train_path [path-to-train_preprocessed] --test_path [path-to-test_preprocessed]
  ```

This data preparation step goal is to generate heatmap label from coordinates. This repo work with data annotations in
the .mat format. In general, we need 2 folders for train and test (for example **train/** and **test/**). Each folder
should be structured as follows:

```
- train/
  - images/
    - train_1.png
    - train_2.png
    ...
  - annotations/
    - train_1.mat
    - train_2.mat
    ...
    
- test/
  - images/
    - test_1.png
    - test_2.png
    ...
  - annotations/
    - test_1.mat
    - test_2.mat
    ...
```

each .mat file should contains a dictionary with the follows in information

```
{
"landmark": np.ndarray of shape K x 2   # landmark coordinates
"heatmap": np.ndarray of shape (K+1) x H x W   # landmark heatmap
"mask_landmark" : np.ndarray of shape (K+1) x 1   # one-hot visibility mask of the landmarks 
"headpose": np.ndarray of shape 3   # headpose, optional
"image_name": name of the corresponding face image
}
```

Then we need to split the train, test set as well as the labeled and unlabeled train partition, you will need to
generate some text files containing the name of the .mat files (refer to ./data/split for examples)

You need to follow to steps above to be able to train and evaluate

## Configurations

This step is optional, but you can find helpful configuration for your custom training here

- go to ``config/config.py`` for configure G2LCPS training, or ``config/config_ema_vit.py`` for semi-supervised ViT
  training. Edit the following lines:
  ```
  C.train_annotations_path = 'path_to_train_mat_annotations_folder'
  C.train_images_path = 'path_to_train_images_folder'
  C.test_annotations_path = 'path_to_test_mat_annotations_folder'
  C.test_images_path = 'path_to_test_images_folder'
  ```
  and
  ```
  C.train_text_labeled = './data/split/train_labeled_[ratio].txt'
  C.train_text_unlabeled = './data/split/train_unlabeled_[ratio].txt'
  ```
  with ``[ratio]`` being the ratio of labeled/total data (can be ``1_8``, ``1_4``, or ``1_2``)


- Optional: toggle the fields: ``C.mean_teacher`` to enable semi-supervised mode or ``C.fully_supervised`` to enable
  fully-supervised training.

## Training

### G2LCPS

- run (add flag ``--resume`` to continue training from last checkpoint)
  ```
  python train.py --mean_teacher \
                  --train_text_labeled path_to_labeled_train_split_text \
                  --train_text_unlabeled path_to_unlabeled_train_split_text \
                  --test_text path_to_test_text \
                  --train_annotations_path path_to_train_mat_annotations_folder \
                  --test_annotations_path path_to_test_mat_annotations_folder \
                  --train_images_path path_to_train_images_folder \
                  --test_images_path path_to_test_images_folder \
                  --num_classes 20\ #for AFLW-19
                    
  ```

### Semi-supervised ViT

  ```
  python train_ema_vit.py --mean_teacher \
                  --train_text_labeled path_to_labeled_train_split_text \
                  --train_text_unlabeled path_to_unlabeled_train_split_text \
                  --test_text path_to_test_text \
                  --train_annotations_path path_to_train_mat_annotations_folder \
                  --test_annotations_path path_to_test_mat_annotations_folder \
                  --train_images_path path_to_train_images_folder \
                  --test_images_path path_to_test_images_folder \
                  --num_classes 20 \ #for AFLW-19
                  --pretrained_path path_to_pretrained_MAE
                    
  ```

For MAE training, refer to MAE/train_mae.py

**Evaluation**
-

## G2LCPS

- run
  ```
  python test_ema.py --checkpoint_path path_to_pt_checkpoint \
                     --test_text path_to_test_text\
                     --test_annotations_path path_to_test_mat_annotations_folder\
                     --test_images_path path_to_test_images_folder\
                     --num_classes 20
  ```

## Semi-supervised ViT

- run
  ```
  python test_ema_vit.py --checkpoint_path path_to_pt_checkpoint \
                     --test_text path_to_test_text\
                     --test_annotations_path path_to_test_mat_annotations_folder\
                     --test_images_path path_to_test_images_folder\
                     --num_classes 20
  ```

# Pretrained weights

