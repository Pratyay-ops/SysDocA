!pip install --upgrade pip
!pip install transformers
!pip install torch
!pip install sentencepiece
!pip install sentence_transformers
!pip install huggingface_hub
!pip install tiktoken
!pip install pypdf
!pip install langchain
!pip install chromadb
!pip install ipywidgets
!pip install --upgrade transformers

from google.colab import drive
drive.mount('/content/drive')

folder_path='/content/drive/MyDrive/pdf'

import langchain
from langchain.document_loaders import PyPDFLoader
import os
pdf_texts={}
data_loader={}

for file in os.listdir(folder_path):
  pdf_text=""
  if file.endswith('.pdf'):
    pdf_path = os.path.join(folder_path, file)
    loader = PyPDFLoader(pdf_path)
    data = loader.load()
    data_loader[pdf_path]=data
    for doc in data:
      pdf_text+=doc.page_content+" "
  pdf_texts[pdf_path]=pdf_text

for pdf_path,pdf_text in pdf_texts.items():
  text=pdf_text.replace("\n","")
  pdf_texts[pdf_path]=text

def chunk_text(text, chunk_size=1024):
  chunks = []
  start = 0
  while start < len(text):
    end = min(start + chunk_size, len(text))
    chunks.append(text[start:end])
    start = end
  return chunks

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("slauw87/bart_summarisation")
model = AutoModelForSeq2SeqLM.from_pretrained("slauw87/bart_summarisation")
model.to(device)
summaries={}

for pdf_path,text in pdf_texts.items():
  inputs = tokenizer.encode(text, truncation=True, max_length=1024, return_tensors="pt")
  summary_ids = model.generate(inputs.to(device), num_beams=4, min_length=100, max_length=200, length_penalty=2.0, early_stopping=True)
  summary = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)
  summaries[pdf_path]=summary

for pdf_path,summary in summaries.items():
  print(f"{pdf_path} : {summary}\n")

from transformers import pipeline
qa_pipeline = pipeline("question-answering", model="facebook/bart-large-cnn")

def find_best_matching_pdf(query):
    best_match = None
    best_score = 0

    for pdf_file, summary in summaries.items():
        answer = qa_pipeline(question=query, context=summary)
        print(answer)
        relevance_score = answer["score"]
        print(f"{pdf_file} score: {relevance_score}")
        if relevance_score > best_score:
            best_match = pdf_file
            best_score = relevance_score

    return best_match

import tiktoken

tokenizer = tiktoken.get_encoding('cl100k_base')

def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

from langchain.text_splitter import RecursiveCharacterTextSplitter                  
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""]
)

from langchain.embeddings import HuggingFaceEmbeddings                         
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

from langchain.chains import RetrievalQA
import langchain
from langchain import HuggingFaceHub
from langchain.vectorstores import Chroma
import os

HUGGINGFACEHUB_API_TOKEN= #api_id#
repo_id = "tiiuae/falcon-7b-instruct"
llm = HuggingFaceHub(huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
                     repo_id=repo_id,
                     model_kwargs={"temperature":0.0, "max_new_tokens":800})
docsearch_list={}
for pdf_path, data in data_loader.items():
  texts = text_splitter.split_documents(data)
  docsearch = Chroma.from_documents(texts, embeddings)
  docsearch_list[pdf_path]=docsearch

def get_answer(query,docsearch):
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())
    answer = qa(query)
    return answer['result']

!pip install nltk
import nltk
import string
from nltk.tokenize import word_tokenize
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
sw = stopwords.words('english')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
def query_refiner(query):
    original_query=query
    original_query=original_query.lower()
    tokens = word_tokenize(original_query)
    tokens = [word for word in tokens if word not in string.punctuation]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    filtered_tokens = [word for word in tokens if word not in sw]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def relevant_texts(docsearch,query):
  relevant_passages=docsearch.similarity_search(query)
  num_passages=len(relevant_passages)
  i=0
  relevant_texts=""
  while i<num_passages-1:
    relevant_texts=relevant_texts+relevant_passages[i].page_content
    i=i+1
  return relevant_texts,relevant_passages

def backtrack_answer(relevant_passage, query):
  matching_pages = []
  matching_text = []
  num_passages=len(relevant_passage)
  i=0
  while i<num_passages-1:
    page_num=relevant_passage[i].metadata['page']
    matching_pages.append(page_num)
    page_text=relevant_passage[i].page_content
    matching_text.append(page_text)
    i=i+1
  return matching_pages,matching_text

import re
import random
import json
import en_core_web_sm
from string import punctuation
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
import torch


class QuestionGenerator:
    def __init__(self, model_dir=None):
        QG_PRETRAINED = 'iarfmoose/t5-base-question-generator'
        self.ANSWER_TOKEN = '<answer>'
        self.CONTEXT_TOKEN = '<context>'
        self.SEQ_LENGTH = 512

        self.device = torch.device('cuda')
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.qg_tokenizer = AutoTokenizer.from_pretrained(QG_PRETRAINED,padding=True)
        self.qg_model = AutoModelForSeq2SeqLM.from_pretrained(QG_PRETRAINED)
        self.qg_model.to(self.device)

        self.qa_evaluator = QAEvaluator(model_dir)


    def generate(self, article, use_evaluator=True, num_questions=None, answer_style='all'):
        qg_inputs, qg_answers = self.generate_qg_inputs(article, answer_style)
        generated_questions = self.generate_questions_from_inputs(qg_inputs, num_questions)
        return generated_questions


    def generate_qg_inputs(self, text, answer_style):
        VALID_ANSWER_STYLES = ['all', 'sentences', 'multiple_choice']

        if answer_style not in VALID_ANSWER_STYLES:
            raise ValueError(
                "Invalid answer style {}. Please choose from {}".format(
                    answer_style,
                    VALID_ANSWER_STYLES
                )
            )

        inputs = []
        answers = []

        if answer_style == 'sentences' or answer_style == 'all':
            segments = self._split_into_segments(text)
            for segment in segments:
                sentences = self._split_text(segment)
                prepped_inputs, prepped_answers = self._prepare_qg_inputs(sentences, segment)
                inputs.extend(prepped_inputs)
                answers.extend(prepped_answers)

        if answer_style == 'multiple_choice' or answer_style == 'all':
            sentences = self._split_text(text)
            prepped_inputs, prepped_answers = self._prepare_qg_inputs_MC(sentences)
            inputs.extend(prepped_inputs)
            answers.extend(prepped_answers)

        return inputs, answers


    def generate_questions_from_inputs(self, qg_inputs, num_questions):
        generated_questions = []
        count = 0
        for qg_input in qg_inputs:
            if count < int(num_questions):
                question = self._generate_question(qg_input)
                question = question.strip()                 #remove trailing spaces
                question = question.strip(punctuation)      #remove trailing questionmarks
                question += "?"                             #add one ?
                if question not in generated_questions:
                    generated_questions.append(question)
                    count += 1
            else:
                return generated_questions
        return generated_questions


    def _split_text(self, text):
        MAX_SENTENCE_LEN = 128
        sentences = re.findall('.*?[.!\?]', text)
        cut_sentences = []
        for sentence in sentences:
            if len(sentence) > MAX_SENTENCE_LEN:
                cut_sentences.extend(re.split('[,;:)]', sentence))
        cut_sentences = [s for s in sentences if len(s.split(" ")) > 5]
        sentences = sentences + cut_sentences
        return list(set([s.strip(" ") for s in sentences]))


    def _split_into_segments(self, text):
        MAX_TOKENS = 490
        paragraphs = text.split('\n')
        tokenized_paragraphs = [self.qg_tokenizer(p)['input_ids'] for p in paragraphs if len(p) > 0]
        segments = []
        while len(tokenized_paragraphs) > 0:
            segment = []
            while len(segment) < MAX_TOKENS and len(tokenized_paragraphs) > 0:
                paragraph = tokenized_paragraphs.pop(0)
                segment.extend(paragraph)
            segments.append(segment)
        return [self.qg_tokenizer.decode(s) for s in segments]


    def _prepare_qg_inputs(self, sentences, text):
        inputs = []
        answers = []
        for sentence in sentences:
            qg_input = '{} {} {} {}'.format(
                self.ANSWER_TOKEN,
                sentence,
                self.CONTEXT_TOKEN,
                text
            )
            inputs.append(qg_input)
            answers.append(sentence)
        return inputs, answers


    def _prepare_qg_inputs_MC(self, sentences):
        spacy_nlp = en_core_web_sm.load()
        docs = list(spacy_nlp.pipe(sentences, disable=['parser']))
        inputs_from_text = []
        answers_from_text = []
        for i in range(len(sentences)):
            entities = docs[i].ents
            if entities:
                for entity in entities:
                    qg_input = '{} {} {} {}'.format(
                        self.ANSWER_TOKEN,
                        entity,
                        self.CONTEXT_TOKEN,
                        sentences[i]
                    )
                    answers = self._get_MC_answers(entity, docs)
                    inputs_from_text.append(qg_input)
                    answers_from_text.append(answers)
        return inputs_from_text, answers_from_text


    def _get_MC_answers(self, correct_answer, docs):
        entities = []
        for doc in docs:
            entities.extend([{'text': e.text, 'label_': e.label_} for e in doc.ents])

        # remove duplicate elements
        entities_json = [json.dumps(kv) for kv in entities]
        pool = set(entities_json)
        num_choices = min(4, len(pool)) - 1  # -1 because we already have the correct answer

        # add the correct answer
        final_choices = []
        correct_label = correct_answer.label_
        final_choices.append({'answer': correct_answer.text, 'correct': True})
        pool.remove(json.dumps({'text': correct_answer.text, 'label_': correct_answer.label_}))

        # find answers with the same NER label
        matches = [e for e in pool if correct_label in e]

        
        if len(matches) < num_choices:
            choices = matches
            pool = pool.difference(set(choices))
            choices.extend(random.sample(list(pool), num_choices - len(choices)))
        else:
            choices = random.sample(matches, num_choices)

        choices = [json.loads(s) for s in choices]
        for choice in choices:
            final_choices.append({'answer': choice['text'], 'correct': False})
        random.shuffle(final_choices)
        return final_choices


    def _generate_question(self, qg_input):
        self.qg_model.eval()
        encoded_input = self._encode_qg_input(qg_input)
        with torch.no_grad():
            output = self.qg_model.generate(input_ids=encoded_input['input_ids'])
        question = self.qg_tokenizer.decode(output[0])
        question = question.replace('pad>', '').replace('</s', '').replace('<unk>','').strip()
        return question


    def _encode_qg_input(self, qg_input):
        return self.qg_tokenizer(
            qg_input,
            padding='longest',
            max_length=self.SEQ_LENGTH,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)


class QAEvaluator:
    def __init__(self, model_dir=None):
        QAE_PRETRAINED = 'iarfmoose/bert-base-cased-qa-evaluator'
        self.SEQ_LENGTH = 512

        self.device = torch.device('cuda')
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.qae_tokenizer = AutoTokenizer.from_pretrained(QAE_PRETRAINED)
        self.qae_model = AutoModelForSequenceClassification.from_pretrained(QAE_PRETRAINED)
        self.qae_model.to(self.device)


    def encode_qa_pairs(self, questions, answers):
        encoded_pairs = []
        for i in range(len(questions)):
            encoded_qa = self._encode_qa(questions[i], answers[i])
            encoded_pairs.append(encoded_qa.to(self.device))
        return encoded_pairs


    def get_scores(self, encoded_qa_pairs):
        scores = {}
        self.qae_model.eval()
        with torch.no_grad():
            for i in range(len(encoded_qa_pairs)):
                scores[i] = self._evaluate_qa(encoded_qa_pairs[i])
        return [k for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)]


    def _encode_qa(self, question, answer):
        if type(answer) is list:
            for a in answer:
                if a['correct']:
                    correct_answer = a['answer']
        else:
            correct_answer = answer
        return self.qae_tokenizer(
            text=question,
            text_pair=correct_answer,
            padding='longest',
            max_length=self.SEQ_LENGTH,
            truncation=True,
            return_tensors="pt"
        )


    def _evaluate_qa(self, encoded_qa_pair):
        output = self.qae_model(**encoded_qa_pair)
        return output[0][0][1]

qg=QuestionGenerator()

def question_gen(relevant_text,query):
  generated_questions_list = qg.generate(relevant_text.replace("\n",''), num_questions=3)
  return generated_questions_list

import ipywidgets as widgets
from IPython.display import display

question_text = widgets.Textarea(
    placeholder='Enter your question...',
    layout=widgets.Layout(width='600px', height='150px')
)

answer_text = widgets.Textarea(
    placeholder='Answer will be displayed here...',
    layout=widgets.Layout(width='600px', height='150px')
)

matches_text = widgets.Textarea(
    placeholder='Matches found will be displayed here...',
    layout=widgets.Layout(width='600px', height='150px')
)
suggested_questions_text = widgets.Textarea(
    placeholder='Suggested question will be displaced here...',
    layout=widgets.Layout(width='600px', height='150px')
)
ask_button = widgets.Button(
    description='Ask',
    button_style='success',
    layout=widgets.Layout(width='100px')
)

next_button = widgets.Button(
    description='Next Question',
    button_style='info',
    layout=widgets.Layout(width='150px')
)

def on_ask_button_clicked(b):
    question = question_text.value.strip()
    best_match=find_best_matching_pdf(question)
    docsearch=docsearch_list[best_match]
    relevant_text,relevant_passage=relevant_texts(docsearch,question)
    if question:
        refined_query = query_refiner(question)
        answer = get_answer(refined_query,docsearch)
        answer_text.value = answer
        answer_text.layout.visibility = 'visible'

        matching_pages, matching_text = backtrack_answer(relevant_passage, refined_query)
        if matching_pages:
            pdf_text=best_match.replace("/content/drive/MyDrive/pdfs/","")
            pdf_page_text= f"Matches found on {pdf_text}"
            pages_text = f"Matches found on pages: {', '.join(map(str, matching_pages))}"
            text_text = "\n\n".join(matching_text)
            matches_text.value = f"{pdf_page_text}\n\nMatches found on pages:\n{pages_text}\n\nMatching text:\n{text_text}"
            matches_text.layout.visibility = 'visible'
        else:
            matches_text.value = "No matches found."
            matches_text.layout.visibility = 'visible'

        generated_questions_list =  question_gen(relevant_text,question)
        suggested_questions_text.value = "\n".join(generated_questions_list)
        suggested_questions_text.layout.visibility = 'visible'
    else:
        question_text.placeholder = 'Please enter a question.'

display(gui_container)
