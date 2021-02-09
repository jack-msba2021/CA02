# Spam eMail Detection using Naive Bayes Classification Algorithm

## Project Overview
This project practices training and predicting with sklearn Naive Bayes while using the Gaussian method for model training. The project example uses email data and attempts to train the model to properly label messages as Spam or Not Spam. The model was trained using 702 emails (equally divided into spam and non spam categories) and tested on 206 emails. Before running the model, the data was cleaned by creating a dictionary for all words found in the files, removing non-alphabetical words, and extracting the 3000 most common words into a final dictionary. Lastly, a label and word frequency matrix was generated. After running the model an accuracy score was printed.

## Requirements
The following can be used to run the Python code for this project:

 - [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb#recent=true)

Make sure your Google Drive/Colab Folder structure is as follows:

 - /content/drive/My Drive/MSBA_Colab_2020/ML_Algorithms/CA02/Data
  
### Installation
The following packages are needed for this project:
```python
import os
import numpy as np
from collections import Counter
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
```

## Project File and Data
You can access the [Project](https://colab.research.google.com/drive/1jX1oEV3FwhmYO7jaUhchNvEUBszmJckG?usp=sharing) through this link.

Download the data here: [Data](https://github.com/ArinB/MSBA-CA01-Spam-Mail-Naibe-Bayes) 

## Trouble Shooting
Make sure to mount your drive before running the model.


## Credits and Acknowledgments
Thank you Professor Arin Brahma of Loyola Marymount University for providing the file template and dataset.

## License
[MIT](https://choosealicense.com/licenses/mit/)

