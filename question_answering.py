import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

QA_input = [{'question': 'What are neutrinos?',
             'context': 'Neutrinos are elusive subatomic particles that are constantly streaming through space and matter nearly undetected. Though trillions of neutrinos pass through your body every second, they rarely interact with matter due to their neutral charge and tiny mass. Neutrinos come in three flavors - electron, muon, and tau - each associated with a corresponding lepton particle. They are produced copiously in nuclear reactions within stars, supernovae, and other high-energy astrophysical environments. Neutrinos oscillate between flavors as they travel, implying they have mass, unlike previously theorized. The detection of solar neutrinos in the 1960s provided critical evidence that nuclear fusion powers the Sun. Neutrinos provide a unique window into these energetic phenomena across the universe due to their weak interactions. But their ghostly nature also makes them challenging to study. Determining precise neutrino masses and other properties remains an active area of research in particle physics today. Though abundantly produced in nature, much about neutrinos continues to elude scientists.'},
             {'question': 'What significance do neutrinos have on our planet Earth?',
              'context': 'Neutrinos pass unfelt through the matter of our planet, only rarely leaving a telltale trace of their journey. Though trillions stream through every square inch of Earth each second, these ethereal particles interact so rarely that their direct influence is but a whisper. Produced in the furious furnace of the Sun core, neutrinos zip through its dense plasma unhindered, emerging straight from the heart of the nuclear forge. Detecting these solar neutrinos proved the power source of stars, yet our world remains largely oblivious to the torrent passing through it. On occasion, a neutrino will collide with an atom in our upper atmosphere, creating a shower of particles that reveals where the neutrino began its cosmic trek. And deep underground, in laboratories shielded from other radiation, scientists patiently watch for the next faint flash that announces a neutrinos arrival. While their ghostly presence flows through the rocks, water, and lives of our planet, neutrinos remain aloof, barely noticeable except by those seeking their subtle signs. The Earth itself feels no direct impact from the trillions of neutrinos traversing it every moment. These particles reveal their secrets only to those determined to listen for their whisper-soft footsteps.'}]

model_name = 'deepset/roberta-base-squad2'
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(model)
print(tokenizer)

input0 = tokenizer(QA_input[0]['question'], QA_input[0]['context'], return_tensors="pt")
output0 = model(**input0)

input1 = tokenizer(QA_input[1]['question'], QA_input[1]['context'], return_tensors="pt")
output1 = model(**input1)

print("Input 0", input0)
print("Output 0", output0)
print("Input 1", input1)
print("Output 1", output1)

answer_start_idx = torch.argmax(output0.start_logits)
answer_end_idx = torch.argmax(output0.end_logits)

answer_tokens = input0.input_ids[0, answer_start_idx: answer_end_idx + 1]
answer = tokenizer.decode(answer_tokens)
print("ques: {}\nanswer: {}".format(QA_input[0]['question'], answer))

answer_start_idx = torch.argmax(output1.start_logits)
answer_end_idx = torch.argmax(output1.end_logits)

answer_tokens = input1.input_ids[0, answer_start_idx: answer_end_idx + 1]
answer = tokenizer.decode(answer_tokens)
print("ques: {}\nanswer: {}".format(QA_input[1]['question'], answer))

qa = pipeline('question-answering', model=model_name, tokenizer=model_name)

output_0 = qa(QA_input[0]['question'], QA_input[0]['context'])
print(output_0)

output_1 = qa(QA_input[1]['question'], QA_input[1]['context'])
print(output_1)