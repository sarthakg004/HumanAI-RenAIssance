# HumanAI RenAissance Selection Task Solution

This repository contains the solution to the selection task for HumanAI-RenAissance project for GSOC 2025.

## Project
Synthetic renaissance text generation with generative models

## Task (Optical Character Recognition)
Build a model based on convolutional-recurrent, transformer, or self-supervised architectures for optically recognizing the text of each data source. 
Your model should be able to detect the main text in each page, while disregarding other embellishments. Pick the most appropriate approach and discuss your strategy.

## Workflow

To solve the task I have developed an CRNN model using pytorch which perform OCR at word level.

### 1. Dataset generation for model training
 - I have also added *Luisa de Padilla’s Nobleza Virtuosa* to the 6 PDFs that were provided.

   I have renamed all the PDF files and respecive transcript for convenience.

```bash
Padilla - 1 Nobleza virtuosa : book1
Padilla - 2 Nobleza virtuosa : book2
PORCONES.228.35 – 1636.pdf : book3
Constituciones sinodales Calahorra 1602.pdf : book4
Ezcaray - Vozes.pdf : book5
Mendo - Principe perfecto.pdf : book6
Paredes - Reglas generales.pdf : book7
Buendia - Instruccion.pdf : book8
```
   

 - Some pages of the pdf contain 2 pages in a single page so we need to split them.
 - 
   <img src="images/image.png" alt="Example" width="800">

 - after creating images of individual pages from the PDFs, I have applied various transformation to the Images. Preprocessing images allows for better text detection and recognition.

   `IMAGE ---> GRAYSCALE ---> SKEW CORRECTION ---> NORMALIZING ---> SCALING ---> INK BLEED REMOVAL ---> BINARIZING ---> DENOISING ---> SAHRPENING ---> CONTRAST ENHANCEMENT ---> ERODING ---> Layout Detection ---> PADDING`

    <img src="images/preprocessing.png" alt="Example" width="1000">

- For the purpose of layout detection I experimented with 2 different pretrained models, the **YOLOv11** and the **Paddle DocLayout** model. Although Layout detection was performing well for some pdf file with clear structure(like in below example), for some books it was not able to detect even the main text properly. SO I have not used it in the final transformation pipeline. ( the performance can be improved by fine tuning these models but will require annotated data)

  <img src="images\layout_paddle.png" alt="layout detection example" width="400" height="500">

  - For text detection I used two different models. The DBNET+ model and the CRAFT model. CRAFT was able to produce word level bounding boxes.
    
  <img src="images\paddle_detected.png" alt="text detected by DBNET+" width="400" height="500"> <img src="images\craft_detected.png" alt="text detected by CRAFT" width="400" height="500">

- Now that I have the bounding box coordinates for individual words on the page. I need to map them to their corresponding word in the transcript provided.
    - I decided to use a **pretrained OCR model (Tesseract)**, specifically one trained for **Spanish text recognition**.
    - For each bounding box on a page, the model predicts the recognized text. Then, we compute a **similarity score** between the prediction and all words in the transcript.
       The word with the highest similarity score is assigned to the bounding box, provided that the score exceeds a predefined threshold.
    - I set a **similarity threshold of 0.7**, ensuring that only confident matches are used for labeling the word images.
 
      <img src="images\mapping.png" alt="mapping example" width="1300" height="400">

- Once the mapping is them I saved the bounding box regions as word images and created a dataframe containing the image_name and their corresponding label.

- I have applied various augmentation techniques to increase the size of our dataset. I have generated 4 augmented images for each word image.

    <img src="images\augmentation.png" alt="augmented image example" width="1200" height="300">

  Now that we have our data in the desired form we can use it for training our CRNN model.

## Model Building
I have used pytorch for training a CRNN model.

- I created a vocabulary from the labels ( all the unique character available in our label or transcripts).
  
```bash
Total number of unique characters before : 61
Unique Characters : 
{'s', 'l', 'e', 'v', 'x', 'G', 'X', 'I', 'h', 'H', 'r', 'í', 't', 'n', 'q',
 '9', '2', 'M', 'B', 'Y', 'é', 'N', 'Ñ', '5', 'ñ', 'á', 'O', 'J', 'g', 'b',
  'a', 'U', 'D', 'p', 'K', 'V', 'd', 'S', 'R', 'ó', '6', 'i', '7', 'm', 'E',
   'P', 'z', 'L', '1', 'j', 'f', 'F', 'Z', 'C', 'T', 'u', 'Q', 'o', 'c', 'A',
    'y'}
```

- Based on the noted provided in the transcript files, I have modified our vocabulary using a rule based approach.

```bash
Total number of unique characters after normalization: 31
Normalized unique characters: 
{'s', 'l', 'e', 'x', 'h', 'r', 't', 'n', 'q', '9', '2', '5',
 'ñ', 'g', 'b', 'a', 'p', 'd', '7', '6', 'i', 'm', 'z', '1', 'j',
  'f', 'u', 'o', 'k', 'c', 'y'}
  ```

- Created a character-to-idx and idx-to-character mapping for our generated vocabulary.

```bash
Character to index mapping: {'1': 1, '2': 2, '5': 3, '6': 4, '7': 5, '9': 6, 'a':7,
 'b': 8, 'c': 9, 'd': 10, 'e': 11, 'f': 12, 'g': 13, 'h': 14, 'i': 15, 'j': 16,
  'k': 17, 'l': 18, 'm': 19, 'n': 20, 'o': 21, 'p': 22, 'q': 23, 'r': 24, 's': 25,
  't': 26, 'u': 27, 'x': 28, 'y': 29, 'z': 30, 'ñ': 31, '<blank>': 0}
```

- created DataSet class and a custom collate function to get the data.
  
Our data contain images of variable size but a NN model requires an input of fixes dimension so We need to add padding to our word Images.
1. We can just resize our images but this will result in loss of information or distortion of the text font.
2. we can map all the images to the maximun image size in our dataset. But:
    * there are chances that average size of images are less compared to max size resulting in very large amount of padding. This result in unnecessary weights in our model increasing the compute.
    * During inference there are chance that input is greater than the largest image in training dataset.

SO I have updated the the Dataset Class to transform all the images to a fixed height of 129 pixels and a variable width based on the actual aspect ratio.

Instead of padding all images to a fixed size before training, the collate function ensures that each batch is padded to match the largest image in that batch.

A collate function transforms individual samples into batches suitable for model training.

Why is this beneficial?
- **Reduces unnecessary padding:** Instead of padding all images to the largest in the dataset, we only pad to the largest in each batch, reducing computational overhead.
- **Preserves aspect ratios:** Since images are resized based on height while maintaining width proportionally, text distortion is minimized.
- **Flexible for inference:** During testing, we can adapt to varying image sizes without constraints from the training set's maximum size.

  
    <img src="images\data_loader.png" alt="augmented image example" width="800" height="1000">


- I have used the following model architecture, with early stopping callback(patience:5) to prevent overfitting and learning rate scheduling for stablized learning:

### Defining Model Architecture
 
```mermaid
flowchart TB
    classDef convblock fill:#faf88e,stroke:#333,stroke-width:1px
    classDef poolblock fill:#2196F3,stroke:#333,stroke-width:1px
    classDef bnblock fill:#4CAF50,stroke:#333,stroke-width:1px
    classDef dropoutblock fill:#9C27B0,stroke:#333,stroke-width:1px
    classDef lstmblock fill:#FF5722,stroke:#333,stroke-width:1px
    classDef fcblock fill:#607D8B,stroke:#333,stroke-width:1px
    classDef reshapeblock fill:#E91E63,stroke:#333,stroke-width:1px

    input["Input (B, C, 128, W)"] --> CNN_Stage_1
    
    subgraph CNN_Stage_1 [CNN Stage 1]
        conv1["Conv2D (3×3, 32 filters)"] --> bn1["BatchNorm2D"] --> relu1["ReLU"] --> conv1_2["Conv2D (3×3, 32 filters)"] --> bn1_2["BatchNorm2D"] --> relu1_2["ReLU"] --> pool1["MaxPool2D (2×2)"]
    end
    
    CNN_Stage_1 --> CNN_Stage_2
    
    subgraph CNN_Stage_2 [CNN Stage 2]
        conv2["Conv2D (3×3, 64 filters)"] --> bn2["BatchNorm2D"] --> relu2["ReLU"] --> conv2_2["Conv2D (3×3, 64 filters)"] --> bn2_2["BatchNorm2D"] --> relu2_2["ReLU"] --> pool2["MaxPool2D (2×2)"]
    end
    
    CNN_Stage_2 --> CNN_Stage_3
    
    subgraph CNN_Stage_3 [CNN Stage 3]
        conv3["Conv2D (3×3, 128 filters)"] --> bn3["BatchNorm2D"] --> relu3["ReLU"] --> conv3_2["Conv2D (3×3, 128 filters)"] --> bn3_2["BatchNorm2D"] --> relu3_2["ReLU"] --> pool3["MaxPool2D (2×2)"]
    end
    
    CNN_Stage_3 --> CNN_Stage_4
    
    subgraph CNN_Stage_4 [CNN Stage 4]
        conv4["Conv2D (3×3, 256 filters)"] --> bn4["BatchNorm2D"] --> relu4["ReLU"] --> conv4_2["Conv2D (3×3, 256 filters)"] --> bn4_2["BatchNorm2D"] --> relu4_2["ReLU"] --> pool4["MaxPool2D (2×2)"]
    end
    
    CNN_Stage_4 --> dropout_cnn["Dropout (0.2)"]
    dropout_cnn --> reshape["Reshape (B, W/16, 256×8)"]
    reshape --> lstm1["Bidirectional LSTM1 (256 units)"]
    lstm1 --> dropout_lstm1["Dropout (0.3)"]
    dropout_lstm1 --> lstm2["Bidirectional LSTM2 (256 units)"]
    lstm2 --> dropout_lstm2["Dropout (0.3)"]
    dropout_lstm2 --> attention["Attention"]
    attention --> fc["Linear (512 → num_classes)"]
    fc --> output["Output (B, W/16, num_classes)"]
    
    class conv1,conv2,conv3,conv4,conv1_2,conv2_2,conv3_2,conv4_2 convblock
    class pool1,pool2,pool3,pool4 poolblock
    class bn1,bn2,bn3,bn4,bn1_2,bn2_2,bn3_2,bn4_2 bnblock
    class dropout_cnn,dropout_lstm1,dropout_lstm2 dropoutblock
    class lstm1,lstm2 lstmblock
    class fc fcblock
    class reshape reshapeblock
    class attention fcblock
```

lets assume that our input image has a dimension (3,w,h). The image will be processed through our model accoringly:

```bash
(batch, 3, w, h) → Input RGB image
↓ [Conv1 block + MaxPool]
(batch, 32, w/2, h/2)
↓ [Conv2 block + MaxPool]
(batch, 64, w/4, h/4)
↓ [Conv3 block + MaxPool]
(batch, 128, w/8, h/8)
↓ [Conv4 block + MaxPool]
(batch, 256, w/16, h/16)
↓ [Reshape for LSTM]
(batch, w/16, 256*(h/16)) → Each timestep contains features from a column of the feature map
↓ [LSTM 1 (bidirectional)]
(batch, w/16, 512)   512 = 256*2 (bidirectional)
↓ [LSTM 2 (bidirectional)]
(batch, w/16, 512)
↓ [Attention Layer]
(batch, w/16, 512) 
↓ [Fully Connected Layer]
(batch, w/16, Vocabulary Size) → Per-timestep character predictions for CTC loss
```

### Loss Function Used [CTC(Connectionist Temporal Classification)] 

<img src="images\CTC.png" alt="Description" width="1200" height="300">

After the text detection step, regions, where the text is present, are cropped and sent through convolutional layers to get the features from the image. Later these features are fed to many-to-many LSTM architecture, which outputs softmax probabilities over the vocabulary.  These outputs from different time steps are fed to the CTC decoder to finally get the raw text from images.

the model predicts a character probability distribution for each of the w/16 timesteps. However, the ground truth is just the text (e.g., "HELLO") without specifying which timestep corresponds to which character.

**CTC Loss allows us to align our predictions to the actual ground truth. CTC computes the probability of all possible alignments between the predicted sequence and the ground truth, then maximizes the sum of these probabilities.**

<img src="images\CTC_mapping.png" alt="Description" width="600" height="200">

**Key Components of CTC**

1. Blank Token: CTC introduces a special "blank" token  to handle:

Repeated characters (e.g., "HELLO" vs "HEELLLLOO")
Different sequence lengths (typically, predictions are longer than labels)

(I added the blank token while creating the char_to_idx mapping)

2. Many-to-One Mapping: CTC defines a many-to-one mapping that collapses predictions into ground truth:

First, it merges repeated characters (unless separated by blank)
Then, it removes all blank tokens

For example:

Prediction: "H-EE-L-L-O" → "HELLO"
Prediction: "HHH--EEELLL--OO" → "HELLO"

3. Probability Computation: For each possible alignment path, CTC calculates its probability based on your model's outputs.


our model outputs (batch, w/16, Vocabulary Size) so we have w/16 predictions for each image. so for each image CTC computes the probability of all valid alignments across the w/16 timesteps. **The loss is the negative log of the sum of these probabilities**

### Results

I also trained the model without applying the preprocessing step to check how much improvement in performance can be achieved.

| Metric      | Without Preprocessing | With Preprocessing |
|------------|----------------------|-------------------|
| Train Loss | 0.0211              | 0.0155           |
| Val Loss   | 0.0618              | 0.0486           |
| Accuracy   | 0.9521              | 0.9596           |
| CER        | 0.0160              | 0.0131           |

During training(with preprocessing) the accuracy reached till 0.965 and can be further improved by fine tuning the early stopping patience.

<img src="images\training_metrics.png" alt="training metrics" width="1000" height="700">

<img src="images\predictions.png" alt="prediction with preprocessing" width="1400" height="400">

the details of the training can be found in `/models` folder [here](https://iitbhu365-my.sharepoint.com/:f:/g/personal/saarthak_gupta_mec22_iitbhu365_onmicrosoft_com/EmmfM8DCrjlIr9iSP_KBMu0BbnYZ2szhd7zWK2mjeF8skQ?e=cuNL2A).


### Testing
Now finally I used the trained model to generate transcripts for the testing pages.(These are the page of the book for which no transcripts were provided).

Example of full text generated transcript:
```
aproacion
del reuerendisino padre maestro fre diego miseno gues
uezes prouincial aora defnidor mayor dela prouincia
de castilla del orden de basilio magn doctor de
la iglesia
m p s
e orden y comision de u ahe uisto con
igual cuydado gusto un libro cuotitulo
es principe ean ministros ajustados docu
mentos politicos  morales cuyo author es el muy
r padre andres mendo de la compa�iad iesus re
ctor del de irlandeses de salamanco de
sad
calificador consejo de la
faas
suprema no auer encontra
dd
do proposicion alguna que al sincero sentir
mmaa
de nuestra catholica religion ni honesto proceder
de las christianas costumbres juzgo no solo por de
leytables por sumamente utiles estos documentos
consagrados a erudicion sin
daaa haddaagaa
gular de principes la eleccion
pues como dijo lipsio d moribu aus scientiis aliquid
do pref ad lb
soluticr s usum habet s 2on magis in regno
caa
ciuili com illic mm aliquem ciue
meliorem doctioremue facias hic in o ones el la
car a un principe consumado estudiante en el arte
de gouernar es lo mismo que hazer docto sabio to
do el principado pues de aquel primer mobil depen
den las acciones y de aquella cabeza las influencias
a todo lo restante del cuerpo y en los libros es sin
duda que beben mas syncera y pura la enseanza
en otros medios que como dijo el doctisimo
sada
hablando con los principes sunt apud uos
fateo consilysui copiam asidue faciants utinan co
sily
```
The final transcripts for all the pages is available in `/predictions` folder.
  
## Setup Instructions
To reproduce the results follow the below instructions

### Install Dependencies
Run the following command to install required libraries:

```bash
pip install -r requirements.txt
```

### Configure Tesseract OCR
Ensure **Tesseract OCR** is installed and correctly configured:
- **Windows:** Set the path to `tesseract.exe` in the notebook.

### Running the Notebook
1. Clone this repository:
   ```bash
   git clone https://github.com/sarthakg004/HumanAI-RenAIssance.git
   cd HumanAI-RenAIssance
   ```
2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook experimentation.ipynb
   ```
3. Run Each cell sequentially.

