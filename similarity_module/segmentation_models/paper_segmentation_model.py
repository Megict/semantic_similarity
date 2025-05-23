import torch
from tqdm import tqdm
import numpy as np

from transformers import RobertaForSequenceClassification, RobertaTokenizer, pipeline

LabtoNum = {'Algo' : 0 , 'Goal' : 1, 'None' : 2, 'Obl' : 3, 'Resl' : 4}
NumToLab = { 0 : 'Algo', 1 : 'Goal', 2 : 'None', 3 : 'Obl', 4 : 'Resl'}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

class text_segment_classifier():
    def __init__(self, model_state_dict_file : str, only_cpu = False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("loading base model...")
        tokenizer = RobertaTokenizer.from_pretrained("mlsa-iai-msu-lab/sci-rus-tiny3")
        model = RobertaForSequenceClassification.from_pretrained("mlsa-iai-msu-lab/sci-rus-tiny3")
        print("reconfiguring classifier head...")
        # настройка модели и загрузка параметров.
        model.config.id2label = NumToLab
        model.config.label2id = LabtoNum
        model.classifier = torch.nn.Sequential(
            torch.nn.Linear(312,149),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=0.1, inplace=False),
            torch.nn.Linear(149,5),
            torch.nn.Softmax(dim = -1)
        )
        print("loading new weights...")
        if only_cpu:
            model.load_state_dict(torch.load(model_state_dict_file, weights_only=True, map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(model_state_dict_file, weights_only=True))
        model = model.eval()
        # настройка пайплайна
        _pipeline = pipeline("text-classification", model=model,tokenizer=tokenizer, device = self.device)
        self.pipeline = _pipeline
        print("done!")

    def predict(self, sentence : str):
        with torch.no_grad():
            inp = self.pipeline.tokenizer(sentence, return_tensors = "pt")
            inp = inp.to(device)
            outputs = self.pipeline.model(**inp)
        res = outputs.logits[0][0]
        pred = int(torch.argmax(res))
        pred_calss = self.pipeline.model.config.id2label[pred]
        return pred_calss

    def mass_predict(self, sentences : list, verbal : bool = False):
        preds = []
        it = tqdm(sentences) if verbal else sentences
        for sentence in it:
            with torch.no_grad():
                inp = self.pipeline.tokenizer(sentence, return_tensors = "pt")
                inp = inp.to(device)
                outputs = self.pipeline.model(**inp)
            res = outputs.logits[0][0]
            pred = int(torch.argmax(res))
            pred_calss = self.pipeline.model.config.id2label[pred]
            preds.append(pred_calss)
        return preds
    
    def mass_predict_alter(self, sentences : list, n_samples = 5, verbal : bool = False):
        preds = ['None' for _ in sentences]
        all_scores = []
        it = tqdm(sentences) if verbal else sentences
        for sentence in it:
            with torch.no_grad():
                inp = self.pipeline.tokenizer(sentence, return_tensors = "pt", truncation=True)
                inp = inp.to(device)
                outputs = self.pipeline.model(**inp)
            scores = outputs.logits[0][0].detach().cpu().numpy()
            # print(scores)
            all_scores.append(scores)
            # pred = int(torch.argmax(res))
        all_scores = np.array(all_scores)
        for class_ in list(self.pipeline.model.config.id2label.keys()):
            for ind in all_scores[:,class_].argsort()[-n_samples:][::-1]:
                preds[ind] = self.pipeline.model.config.id2label[class_]
        return preds