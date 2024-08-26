# Biting The Bytes: Transformers For Diacritic Restoration

This project focuses on diacritic restoration for the Turkish language using transformer models. It was developed as part of the Natural Language Processing course at Istanbul Technical University in Spring 2024.

## Authors

- Emircan Erol - 150200324 (erole20@itu.edu.tr)
- Muhammed Rüşen Birben - 150220755 (birben20@itu.edu.tr)

## Project Overview

The goal of this project is to restore diacritics in Turkish text using transformer models. Diacritics are important in Turkish as they change the meaning and pronunciation of words. However, in many digital contexts, Turkish text is often written without diacritics. This project aims to automatically restore the missing diacritics.

The base model used is [google-byt5-small](https://huggingface.co/google/byt5-small). The project notebook has been run iteratively to select the best checkpoint models, modify hyperparameters, and refine the dataset.

## Resources

All resources including the final model used in this project can be found in this [Google Drive folder](https://drive.google.com/drive/folders/1nfAvnj_-EB4FMa83mUCtGNel9DkdC0PL?usp=sharing).

## Setup

### Dependencies

The project uses the following main libraries:

- PyTorch
- Transformers
- PEFT (Parameter-Efficient Fine-Tuning)
- Pandas
- Weights & Biases (wandb)
- [TurboT5](https://github.com/Knowledgator/TurboT5)

Refer to the notebook for the full list of imports.

### Hyperparameters

Key hyperparameters used:

- Learning Rate (LR): 2<sup>-12</sup>
- Minimum Sequence Length (MIN_LEN): 0
- Maximum Sequence Length (MAX_LEN): 512
- Batch Size (BATCH_SIZE): 64

Notice that these parameters are altered slightly for different experiments/training data.

## Data Preparation

The project includes functions to prepare the data for training:

- `asciify_turkish_chars(text)`: Removes diacritics from Turkish text.
- `txt_to_input_output(fp, skip=500_000, split='all')`: Reads a text file and writes it to a JSONL file to be used as a dataset.
- `mask_label(data, batch_size=BATCH_SIZE)`: Masks the padded tokens in the input and creates batches for training.
- `test_mask(data)`: Masks the padded tokens in the input and creates batches for testing.

The main dataset used is `data.jsonl`.It is created by running the `txt_to_input_output` function on the Turkish text data.

## Model Training

The project uses the PEFT library to reduce the trainable parameter size of the base model. If a checkpoint is provided, the model is loaded from the checkpoint (which is our pre-trained model for the task of diacrtic restoration). Otherwise, the model is initialized from scratch using the `google/byt5-small` model.

The model is trained using the prepared dataset. The notebook includes the training loop and evaluation metrics.

## Results

The project achieves promising results in restoring diacritics for Turkish text. The test result can be seen on the official Kaggle competition page [here](https://www.kaggle.com/competitions/yzv405e-term-project-2023-2024/leaderboard).

## Future Work

Potential areas for future improvement include:

- Experimenting with different transformer architectures
- Fine-tuning the hyperparameters further
- Expanding the dataset with more diverse Turkish text

Feel free to reach out to the authors for any questions or collaboration opportunities related to this project.
