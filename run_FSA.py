import os
import time
import threading
import argparse
import logging

import math
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin

from datasets import load_dataset
from google import genai


class Logger:
    def __init__(self, filename, formatter):
        self.logger = logging.getLogger(filename)
        self.ch = logging.StreamHandler()
        self.logger.setLevel(logging.INFO)
        self.ch.setFormatter(formatter)
        self.logger.addHandler(self.ch)
        self.fh = logging.FileHandler(filename=filename, mode='w')
        self.logger.setLevel(logging.DEBUG)
        self.fh.setFormatter(formatter)
        self.logger.addHandler(self.fh)

    def debug(self, info):
        self.logger.debug(info)

    def info(self, info):
        self.logger.info(info)

    def warning(self, info):
        self.logger.warning(info)

    def critical(self, info):
        self.logger.critical(info)


lock = threading.Lock()
correct = {0: 0, 1: 0, 2: 0}


def sample_subset(dataset, size):
    r'''
    sample a subset of a dataset without replacement.
    :param dataset: Dataset
    :param size: number of samples per class
    :return: sampled subset of dataset
    '''
    idx_list = []
    for c in range(3):
        sample_list = np.random.choice(np.where(np.array(dataset['label']) == c)[0], size, replace=False).tolist()
        idx_list.extend(sample_list)
    # print(idx_list)
    return dataset.select(idx_list)


def load_data(dataset_name, cache_dir='./data', seed=42):
    assert dataset_name in ['fiqa', 'twitter']
    np.random.seed(seed)
    if dataset_name == 'fiqa':
        dataset = load_dataset(path="pauri32/fiqa-2018", cache_dir=cache_dir)
        train = dataset['train'].rename_column(original_column_name='sentence', new_column_name='text')
        train = sample_subset(train, 30)
        test = dataset['test'].rename_column(original_column_name='sentence', new_column_name='text')
        sentiments = {0: "positive", 1: "neutral", 2: "negative"}
        return train, test, sentiments

    elif dataset_name == 'twitter':
        dataset = load_dataset(path="zeroshot/twitter-financial-news-sentiment", cache_dir=cache_dir)
        train = dataset['train']
        train = sample_subset(train, 30)
        test = dataset['validation']
        test = sample_subset(test, 197)
        sentiments = {0: "negative", 1: "positive", 2: "neutral"}
        return train, test, sentiments
    else:
        print('Invalid dataset name. Please choose among [fiqa, twitter]')
        return None


def get_exemplar(train, shots, method='random'):
    num_exemplars = 3 * shots
    if method == 'random':
        idx_list = np.random.choice(len(train), size=num_exemplars, replace=False)
        return train.select(idx_list)

    elif method == 'distance':
        client = genai.Client(api_key="YOUR_API_KEY")
        embedding_list = []
        for i in range(len(train)):
            text = train[i]['text']
            not_done = 1
            while not_done:
                try:
                    response = client.models.embed_content(model="text-embedding-004", contents=text)
                    embedding = response.embeddings[0].values
                    embedding_list.append(embedding)
                    not_done = 0
                except Exception as e:
                    print('Index {:4d}'.format(i), f'| text: {text}', f'| {e}')

        embedding_matrix = np.array(embedding_list)

        first_index = np.random.choice(len(train))
        idx_list = [first_index]
        for _ in range(1, num_exemplars):
            dist_to_selected = np.sum(np.linalg.norm(embedding_matrix[idx_list] - embedding_matrix[:, None], axis=2), axis=1)
            next_index = np.argmax(dist_to_selected)
            idx_list.append(next_index)

        return train.select(idx_list)

    elif method == 'subclustering':
        client = genai.Client(api_key="YOUR_API_KEY")
        embedding_dict = {0: [], 1: [], 2: []}
        for i in range(len(train)):
            text = train[i]['text']
            label = train[i]['label']
            not_done = 1
            while not_done:
                try:
                    response = client.models.embed_content(model="text-embedding-004", contents=text)
                    embedding = response.embeddings[0].values
                    embedding_dict[label].append(embedding)
                    not_done = 0
                except Exception as e:
                    print('Index {:4d}'.format(i), f'| text: {text}', f'| {e}')
        idx_list = []
        for c in range(3):
            embedding_matrix = np.array(embedding_dict[c])
            kmeans = KMeans(n_clusters=shots, random_state=42, n_init=10)
            kmeans.fit(embedding_matrix)
            closest_indices = pairwise_distances_argmin(kmeans.cluster_centers_, embedding_matrix) + c * 30
            idx_list.extend(closest_indices)
        return train.select(idx_list)


def get_output(test, i, client, model_name, sentiments, prefix, logger):
    text = test[i]['text']
    label = sentiments[test[i]['label']]
    prompt = prefix + f'\nInput: {text} \nAnswer: '
    not_done = 1
    while not_done:
        time.sleep(np.random.rand()/2)
        try:
            response = client.models.generate_content(model=model_name, contents=prompt)
            prediction = response.text.lower().strip()

            if prediction in ['positive', 'negative', 'neutral']:
                not_done = 0
                global correct
                with lock:
                    correct[test[i]['label']] += int(label == prediction)
                print('Index {:4d}'.format(i),
                      '| label: {:10s}'.format(label),
                      '| prediction: {:10s}'.format(prediction),
                      )
                log_info = ''.join(['Index {:4d}'.format(i),
                                    '| label: {:10s}'.format(label),
                                    '| prediction: {:10s}'.format(prediction),])
                logger.info(log_info)
            else:
                print(prediction)
        except Exception as e:
            print('Index {:4d}'.format(i), f'| text: {text}', f'| {e}')
            time.sleep(0.5)


def run(platform, model_name, dataset_name, shots, batch_size, selective_method):
    global correct
    correct = {0: 0, 1: 0, 2: 0}

    if not os.path.exists('./log'):
        os.mkdir('./log')

    filename = f'log/{dataset_name}_{model_name}_{shots}_{selective_method}.log'
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger = Logger(filename, formatter)

    if platform == 'gemini':
        client = genai.Client(api_key="YOUR_API_KEY")
    else:
        client = None

    train, test, sentiments = load_data(dataset_name)
    num_batch = math.ceil(len(test) / batch_size)

    if shots == 0:
        prefix = ('Instruction: You are a financial sentiment analyst. '
                  'What is the sentiment of the last news? '
                  'You MUST choose an answer from (positive/negative/neutral) with no extra words.')
    else:
        exemplars = get_exemplar(train, shots, method=selective_method)
        exemplar_text = ''.join(['\nInput: {:s} \nAnswer: {:s}'.format(e['text'], sentiments[e['label']]) for e in exemplars])

        prefix = ('Instruction: You are a financial sentiment analyst. '
                  'According to the following exemplars, what is the sentiment of the last news? '
                  'You MUST choose an answer from (positive/negative/neutral) with no extra words.')
        prefix = prefix + exemplar_text

    t0 = time.time()
    for batch_idx in range(num_batch):
        time.sleep(0.8)
        if batch_idx < num_batch - 1:
            idx_list = list(range(batch_idx * batch_size, (batch_idx + 1) * batch_size))
        else:
            idx_list = list(range(batch_idx * batch_size, len(test)))
        threads = []
        for idx in idx_list:
            thread = threading.Thread(target=get_output,
                                      args=(test, idx, client, model_name, sentiments, prefix, logger))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

    t1 = time.time()
    accuracy = sum(correct.values()) / len(test)
    log_info = ''.join(['Accuracy: {:.4f}'.format(accuracy),
                        f' | {correct}',
                        ' | time: {:.4f} seconds'.format(t1 - t0)])
    logger.info(log_info)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Financial sentiment analysis through in-context learning')
    parser.add_argument('--platform', type=str, default='gemini', help='Choose platform: [gemini]')
    parser.add_argument('--model_name', type=str, default='gemini-1.5-flash', help='Choose model_name')
    parser.add_argument('--dataset_name', type=str, default='fiqa', help='Choose dataset: [fiqa, twitter]')
    parser.add_argument('--shots', type=int, default=2, help='Number of shots (default: 2)')
    parser.add_argument('--selective_method', type=str, default='random', help='selective method: [random, distance, subclustering]')
    parser.add_argument('--batch_size', type=int, default=15, help='Batch size (default: 15)')
    args = parser.parse_args()

    platform = args.platform
    assert platform in ['gemini']

    model_name = args.model_name
    dataset_name = args.dataset_name
    batch_size = args.batch_size
    shots = args.shots
    assert shots in [0, 1, 2, 3, 5]
    selective_method = args.selective_method
    assert selective_method in ['random', 'distance', 'subclustering']
    run(platform, model_name, dataset_name, shots, batch_size, selective_method)
