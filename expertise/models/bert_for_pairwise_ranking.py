import numpy as np
import torch

import transformers

from expertise import utils
from expertise.evaluators.mean_avg_precision import eval_map
from expertise.evaluators.hits_at_k import eval_hits_at_k
import pandas as pd
import ipdb


def attention_mask(input_tensors, pad_val=0):
    return torch.where(
        input_tensors != 0,
        torch.ones_like(input_tensors),
        torch.zeros_like(input_tensors))

class BPRTripletsDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, tokenizer, max_seq_length=250):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        self.tokenizer=tokenizer

        self.max_seq_length = max_seq_length

        self.triplets_frame = pd.read_csv(
            csv_file,
            header=None,
            names=['source', 'positive', 'negative'])

    def __len__(self):
        return len(self.triplets_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.triplets_frame.iloc[idx, :]

        input_ids_A = utils.fixedwidth(
            self.tokenizer.encode(sample[0]),
            self.max_seq_length,
            pad_val=self.tokenizer.vocab['[PAD]'])

        input_ids_B = utils.fixedwidth(
            self.tokenizer.encode(sample[1]),
            self.max_seq_length,
            pad_val=self.tokenizer.vocab['[PAD]'])

        input_ids_C = utils.fixedwidth(
            self.tokenizer.encode(sample[2]),
            self.max_seq_length,
            pad_val=self.tokenizer.vocab['[PAD]'])

        input_tensor_A = torch.tensor(input_ids_A, dtype=torch.long)
        input_tensor_B = torch.tensor(input_ids_B, dtype=torch.long)
        input_tensor_C = torch.tensor(input_ids_C, dtype=torch.long)

        return input_tensor_A, input_tensor_B, input_tensor_C

class BertForBayesianPairwiseRanking(transformers.BertPreTrainedModel):
    def __init__(self, config):
        super(BertForBayesianPairwiseRanking, self).__init__(config)

        self.use_cuda = False
        if torch.cuda.is_available():
            device_number = torch.cuda.current_device()
            device = torch.device(device_number)
            self.use_cuda = True
            self.device = device

        self.bert = transformers.BertModel(config)
        # self.pos_cos_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        # self.neg_cos_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

        if self.use_cuda:
            self.bert = self.bert.to(self.device)
            # self.pos_cos_similarity = self.pos_cos_similarity.to(self.device)
            # self.neg_cos_similarity = self.neg_cos_similarity.to(self.device)

        self.init_weights()

    def forward(
        self,
        source_inputs,
        positive_inputs,
        negative_inputs=None):

        if self.use_cuda:
            source_inputs = source_inputs.to(self.device)
            positive_inputs = positive_inputs.to(self.device)

            if negative_inputs is not None:
                negative_inputs = negative_inputs.to(self.device)

        # TODO:
        # See notes on BertModel output. Despite the name "pooled" output,
        # pooled_output = outputs[1] is apparently the CLS vector,
        # which isn't a good semantic summary.
        # Instead, should average or pool across all word vectors.
        # (or maybe: should average or pool across all keyword vectors?)
        source_bert_outputs = self.bert(
            source_inputs,
            attention_mask=attention_mask(source_inputs))

        positive_bert_outputs = self.bert(
            positive_inputs,
            attention_mask=attention_mask(positive_inputs))

        # outputs[0] is the list of hidden states
        source_avg = torch.mean(source_bert_outputs[0], dim=1)
        positive_avg = torch.mean(positive_bert_outputs[0], dim=1)

        if self.use_cuda:
            source_avg = source_avg.to(self.device)
            positive_avg = positive_avg.to(self.device)

        # positive_scores = self.pos_cos_similarity(source_avg, positive_avg)
        positive_scores = utils.row_wise_dot(source_avg, positive_avg)

        if negative_inputs is not None:

            if self.use_cuda:
                targets = torch.ones((source_inputs.shape[0], 1), device=self.device)
            else:
                targets = torch.ones((source_inputs.shape[0]), 1)

            negative_bert_outputs = self.bert(
                negative_inputs,
                attention_mask=attention_mask(negative_inputs))

            negative_avg = torch.mean(negative_bert_outputs[0], dim=1)
            # negative_scores = self.neg_cos_similarity(source_avg, negative_avg)
            negative_scores = utils.row_wise_dot(source_avg, negative_avg)

            loss_fct = torch.nn.BCEWithLogitsLoss()
            ipdb.set_trace(context=30)
            loss = loss_fct(
                positive_scores-negative_scores,
                targets)

            return (loss, positive_scores, negative_scores)

        return positive_scores


def train(config):
    '''
    training loop
    '''
    for train_subdir in ['dev_scores', 'dev_predictions']:
        train_subdir_path = os.path.join(config.train_dir, train_subdir)
        if not os.path.exists(train_subdir_path):
            os.mkdir(train_subdir_path)

    vocab_path = os.path.join(
        config.kp_setup_dir, 'textrank_vocab.pkl')
    vocab = utils.load_pkl(vocab_path)

    torch.manual_seed(config.random_seed)

    train_samples_path = os.path.join(
        config.setup_dir, 'train_samples.jsonl')

    dev_samples_path = os.path.join(
        config.setup_dir, 'dev_samples.jsonl')

    print('reading train samples from ', train_samples_path)
    batcher = Batcher(input_file=train_samples_path)
    batcher_dev = Batcher(input_file=dev_samples_path)

    model = centroid_scibert.Model(config, vocab)
    if config.use_cuda:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.l2penalty)

    # Stats
    best_map = 0
    sum_loss = 0.0

    # a lookup table of torch.Tensor objects, keyed by user/paper ID.
    bert_lookup = utils.load_pkl(os.path.join(config.kp_setup_dir, 'bert_lookup.pkl'))

    print('Begin Training')

    # Training loop
    for counter, batch in enumerate(batcher.batches(batch_size=config.batch_size)):

        batch_source = []
        batch_pos = []
        batch_neg = []

        for data in batch:
            batch_source.append(bert_lookup[data['source_id']])
            batch_pos.append(bert_lookup[data['positive_id']])
            batch_neg.append(bert_lookup[data['negative_id']])

        print('num_batches: {}'.format(counter))
        optimizer.zero_grad()

        loss_parameters = (
            torch.stack(batch_source),
            torch.stack(batch_pos),
            torch.stack(batch_neg)
        )

        loss = model.compute_loss(*loss_parameters)
        loss.backward()

        # torch.nn.utils.clip_grad_norm(model.parameters(), config.clip)
        optimizer.step()

        # Question: is this if block just for monitoring?
        if counter % 100 == 0:

            this_loss = loss.cpu().data.numpy()
            sum_loss += this_loss

            print('Processed {} batches, Loss of batch {}: {}. Average loss: {}'.format(
                counter, counter, this_loss, sum_loss / (counter / 100)))

        if counter % config.eval_every == 0:

            # is this reset needed?
            batcher_dev.reset()

            predictions = centroid_scibert.generate_predictions(config, model, batcher_dev, bert_lookup)

            prediction_filename = config.train_save(predictions,
                'dev_predictions/dev.predictions.{}.jsonl'.format(counter))

            print('prediction filename', prediction_filename)
            map_score = float(centroid_scibert.eval_map_file(prediction_filename))
            hits_at_1 = float(centroid_scibert.eval_hits_at_k_file(prediction_filename, 1))
            hits_at_3 = float(centroid_scibert.eval_hits_at_k_file(prediction_filename, 3))
            hits_at_5 = float(centroid_scibert.eval_hits_at_k_file(prediction_filename, 5))
            hits_at_10 = float(centroid_scibert.eval_hits_at_k_file(prediction_filename, 10))

            score_lines = [
                [config.name, counter, text, data] for text, data in [
                    ('MAP', map_score),
                    ('Hits@1', hits_at_1),
                    ('Hits@3', hits_at_3),
                    ('Hits@5', hits_at_5),
                    ('Hits@10', hits_at_10)
                ]
            ]
            config.train_save(score_lines, 'dev_scores/dev.scores.{}.tsv'.format(counter))

            if map_score > best_map:
                best_map = map_score

                best_model_path = os.path.join(
                    config.train_dir, 'model_{}_{}.torch'.format(config.name, 'best'))

                torch.save(model, best_model_path)
                config.best_model_path = best_model_path
                config.best_map_score = best_map
                config.hits_at_1 = hits_at_1
                config.hits_at_3 = hits_at_3
                config.hits_at_5 = hits_at_5
                config.hits_at_10 = hits_at_10
                config.save_config()

                config.train_save(score_lines, 'dev.scores.best.tsv')

        if counter == config.num_minibatches:
            break


class Model(torch.nn.Module):
    def __init__(self, config, vocab):
        super(Model, self).__init__()

        self.config = config
        self.vocab = vocab

        # Keyword embeddings
        self.linear_layer = torch.nn.Linear(config.bert_dim, config.embedding_dim)

        # Vector of ones (used for loss)
        if self.config.use_cuda:
            self.ones = torch.ones(config.batch_size, 1).cuda()
        else:
            self.ones = torch.ones(config.batch_size, 1)

        self._bce_loss = torch.nn.BCEWithLogitsLoss()

    def compute_loss(self, batch_source, batch_pos, batch_neg):
        """ Compute the loss (BPR) for a batch of examples
        :param batch_source: a batch of source keyphrase indices (list of lists)
        :param batch_pos: True aliases of the Mentions
        :param batch_neg: False aliases of the Mentions
        """

        batch_size = len(batch_source)

        avg_source = torch.mean(batch_source, dim=1)
        avg_pos = torch.mean(batch_pos, dim=1)
        avg_neg = torch.mean(batch_neg, dim=1)

        # B by dim
        source_embed = self.embed(avg_source)
        # B by dim
        pos_embed = self.embed(avg_pos)
        # B by dim
        neg_embed = self.embed(avg_neg)

        loss = self._bce_loss(
            utils.row_wise_dot(source_embed, pos_embed )
            - utils.row_wise_dot(source_embed, neg_embed ),
            self.ones[:batch_size])
        return loss

    def score_pair(self, source, target, source_len, target_len):
        """

        :param source: Batchsize by Max_String_Length
        :param target: Batchsize by Max_String_Length
        :return: Batchsize by 1
        """
        source_embed = self.embed_dev(source, source_len)
        target_embed = self.embed_dev(target, target_len)
        scores = utils.row_wise_dot(source_embed, target_embed)
        return scores

    def embed(self, vector_batch):
        """
        :param vector_batch: torch.Tensor - Batch_size by bert_embedding_dim
        """

        if self.config.use_cuda:
            vector_batch = vector_batch.cuda()

        # do a linear transformation of the scibert vector into the centroid dimension
        # B x bert_embedding_dim
        try:
            embeddings = self.linear_layer(vector_batch)
        except AttributeError as e:
            ipdb.set_trace()
            raise e

        return embeddings

    def embed_dev(self, vector_batch, print_embed=False, batch_size=None):
        """
        :param keyword_lists: Batch_size by max_num_keywords
        """
        return self.embed(vector_batch)

    def score_dev_test_batch(self,
        batch_queries,
        batch_targets,
        batch_size
        ):

        avg_queries = torch.mean(batch_queries, dim=1)
        avg_targets = torch.mean(batch_targets, dim=1)

        if batch_size == self.config.dev_batch_size:
            source_embed = self.embed_dev(avg_queries)
            target_embed = self.embed_dev(avg_targets)
        else:
            source_embed = self.embed_dev(avg_queries, batch_size=batch_size)
            target_embed = self.embed_dev(avg_targets, batch_size=batch_size)

        scores = utils.row_wise_dot(source_embed, target_embed)

        # what is this?
        scores[scores != scores] = 0

        return scores


def generate_predictions(config, model, batcher, bert_lookup):
    """
    Use the model to make predictions on the data in the batcher

    :param model: Model to use to score reviewer-paper pairs
    :param batcher: Batcher containing data to evaluate (a DevTestBatcher)
    :return:
    """

    for idx, batch in enumerate(batcher.batches(batch_size=config.dev_batch_size)):
        if idx % 100 == 0:
            print('Predicted {} batches'.format(idx))

        batch_queries = []
        batch_query_lengths = []
        batch_query_ids = []
        batch_targets = []
        batch_target_lengths = []
        batch_target_ids = []
        batch_labels = []
        batch_size = len(batch)

        for data in batch:
            # append a positive sample
            batch_queries.append(bert_lookup[data['source_id']])
            batch_query_lengths.append(data['source_length'])
            batch_query_ids.append(data['source_id'])
            batch_targets.append(bert_lookup[data['positive_id']])
            batch_target_lengths.append(data['positive_length'])
            batch_target_ids.append(data['positive_id'])
            batch_labels.append(1)

            # append a negative sample
            batch_queries.append(bert_lookup[data['source_id']])
            batch_query_lengths.append(data['source_length'])
            batch_query_ids.append(data['source_id'])
            batch_targets.append(bert_lookup[data['negative_id']])
            batch_target_lengths.append(data['negative_length'])
            batch_target_ids.append(data['negative_id'])
            batch_labels.append(0)

        scores = model.score_dev_test_batch(
            torch.stack(batch_queries),
            torch.stack(batch_targets),
            np.asarray(batch_size)
        )

        if type(batch_labels) is not list:
            batch_labels = batch_labels.tolist()

        if type(scores) is not list:
            scores = list(scores.cpu().data.numpy().squeeze())

        for source, source_id, target, target_id, label, score in zip(
            batch_queries,
            batch_query_ids,
            batch_targets,
            batch_target_ids,
            batch_labels,
            scores
            ):

            # temporarily commenting out "source" and "target" because I think they are not needed.
            prediction = {
                # 'source': source,
                'source_id': source_id,
                # 'target': target,
                'target_id': target_id,
                'label': label,
                'score': float(score)
            }

            yield prediction

def load_jsonl(filename):

    labels_by_forum = defaultdict(dict)
    scores_by_forum = defaultdict(dict)

    for data in utils.jsonl_reader(filename):
        forum = data['source_id']
        reviewer = data['target_id']
        label = data['label']
        score = data['score']
        labels_by_forum[forum][reviewer] = label
        scores_by_forum[forum][reviewer] = score


    result_labels = []
    result_scores = []

    for forum, labels_by_reviewer in labels_by_forum.items():
        scores_by_reviewer = scores_by_forum[forum]

        reviewer_scores = list(scores_by_reviewer.items())
        reviewer_labels = list(labels_by_reviewer.items())

        sorted_labels = [label for _, label in sorted(reviewer_labels)]
        sorted_scores = [score for _, score in sorted(reviewer_scores)]

        result_labels.append(sorted_labels)
        result_scores.append(sorted_scores)

    return result_labels, result_scores

def eval_map_file(filename):
    list_of_list_of_labels, list_of_list_of_scores = utils.load_labels(filename)
    return eval_map(list_of_list_of_labels, list_of_list_of_scores)

def eval_hits_at_k_file(filename, k=2, oracle=False):
    list_of_list_of_labels,list_of_list_of_scores = utils.load_labels(filename)
    return eval_hits_at_k(list_of_list_of_labels, list_of_list_of_scores, k=k,oracle=oracle)

