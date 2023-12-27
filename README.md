 # Question Answering with Transformers

This code demonstrates how to use a pre-trained transformer model for question answering. We will use the Deepset/roberta-base-squad2 model, which is a transformer model fine-tuned on the SQuAD2 dataset.

## Step 1: Import the necessary libraries

```python
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
```

## Step 2: Define the QA input

We define two QA input dictionaries, each containing a question and a context. The context is a long passage of text, and the question is a question about the context.

```python
QA_input = [{'question': 'What are neutrinos?',
             'context': 'Neutrinos are elusive subatomic particles that are constantly streaming through space and matter nearly undetected. Though trillions of neutrinos pass through your body every second, they rarely interact with matter due to their neutral charge and tiny mass. Neutrinos come in three flavors - electron, muon, and tau - each associated with a corresponding lepton particle. They are produced copiously in nuclear reactions within stars, supernovae, and other high-energy astrophysical environments. Neutrinos oscillate between flavors as they travel, implying they have mass, unlike previously theorized. The detection of solar neutrinos in the 1960s provided critical evidence that nuclear fusion powers the Sun. Neutrinos provide a unique window into these energetic phenomena across the universe due to their weak interactions. But their ghostly nature also makes them challenging to study. Determining precise neutrino masses and other properties remains an active area of research in particle physics today. Though abundantly produced in nature, much about neutrinos continues to elude scientists.'},
             {'question': 'What significance do neutrinos have on our planet Earth?',
              'context': 'Neutrinos pass unfelt through the matter of our planet, only rarely leaving a telltale trace of their journey. Though trillions stream through every square inch of Earth each second, these ethereal particles interact so rarely that their direct influence is but a whisper. Produced in the furious furnace of the Sun core, neutrinos zip through its dense plasma unhindered, emerging straight from the heart of the nuclear forge. Detecting these solar neutrinos proved the power source of stars, yet our world remains largely oblivious to the torrent passing through it.'}]