# Copyright (c) 2016 Thomas Kipf
# Copyright (C) 2017 Sarah Parisot <s.parisot@imperial.ac.uk>, Sofia Ira Ktena <ira.ktena@imperial.ac.uk>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


from __future__ import division
from __future__ import print_function

import os
import shutil
import time

import numpy as np

import random

import torch
import sklearn.metrics
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from environment_setup import PROJECT_ROOT_DIR
from geometric_dataset import create_pytorch_dataset, ABIDEDataset
from metrics import masked_accuracy, MaskedAUC
from model import GCN, Deep_GCN, MLP

# masked_auc_obj = MaskedAUC()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
counter = 0


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


# def normalize_adj(adj):
#     """Symmetrically normalize adjacency matrix."""
#     rowsum = np.array(adj.sum(1))
#     d_inv_sqrt = np.power(rowsum, -0.5).flatten()
#     d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
#     return adj.dot(d_inv_sqrt).transpose().dot(d_inv_sqrt)


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = features.sum(1)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    features = r_mat_inv.dot(features)
    return features


def get_train_test_masks(labels, idx_train, idx_val, idx_test):
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])
    return train_mask, val_mask, test_mask


def create_criterion(dataset):
    # Cross entropy error
    labels = dataset.y[dataset.train_idx]
    label_count_1 = (labels == 1).sum()
    label_count_0 = (labels == 0).sum()
    max_count = torch.max(label_count_1, label_count_0)
    weights = [max_count/label_count_0, max_count/label_count_1]  # Inverse of the count.
    return torch.nn.CrossEntropyLoss(weight=torch.as_tensor(weights, dtype=torch.float).to(DEVICE))

def accuracy(outputs, labels, is_gcn=True):
    accuracy = masked_accuracy(outputs, labels)
    return accuracy


def scale_lr(optimizer, lr):
    """
    Scale the learning rate of the optimizer
    :return: in-place update of parameters
    """
    lr = lr * 0.1
    print(f"Updating the learning rate to {lr}")
    optimizer = optimizer
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr, 0.1


def create_logger(logdir='logs', delete_older=True):
    global counter
    if os.path.exists(os.path.join(PROJECT_ROOT_DIR, logdir)) and delete_older and counter == 0:
        print("deleting older")
        shutil.rmtree(os.path.join(PROJECT_ROOT_DIR, logdir))
    os.makedirs(os.path.join(PROJECT_ROOT_DIR, logdir), exist_ok=True)
    logger_train = SummaryWriter(os.path.join(PROJECT_ROOT_DIR, logdir, f'train{counter}'))
    logger_val = SummaryWriter(os.path.join(PROJECT_ROOT_DIR, logdir, f'val{counter}'))
    counter = counter + 1
    return logger_train, logger_val


def create_model(params, input_dim):
    model_type = params['model']
    hidden1 = int(params['hidden'])
    depth = int(params['depth'])
    dropout = float(params['dropout'])
    degree = int(params['max_degree'])
    if model_type == 'dense':
        return MLP(input_dim=input_dim, hidden1=hidden1, dropout=dropout), False
    elif depth != 0:
        return Deep_GCN(input_dim=input_dim, hidden1=hidden1, depth=depth, dropout=dropout, degree=degree), True
    else:
        return GCN(input_dim=input_dim, hidden1=hidden1, dropout=dropout, degree=degree), True


def run_training(adj, features, labels, idx_train, idx_val, idx_test, params):
    # Set random seed
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])

    learning_rate = float(params['lrate'])
    epochs = int(params['epochs'])
    early_stopping = int(params['early_stopping'])
    weight_decay = float(params['decay'])
    # Some preprocessing
    features = torch.as_tensor(preprocess_features(features), dtype=torch.float)
    # Settings
    # dataset = create_pytorch_dataset(final_graph=adj, x_data=features, y_data=labels)
    dataset = ABIDEDataset(final_graph=adj, x_data=features, y_data=labels)
    dataset = dataset[0]
    # Create test, val and train masked variables
    train_mask, val_mask, test_mask = get_train_test_masks(labels, idx_train, idx_val, idx_test)
    dataset.train_idx = torch.tensor(train_mask, dtype=torch.bool)
    dataset.val_idx = torch.tensor(val_mask, dtype=torch.bool)
    dataset.test_idx = torch.tensor(test_mask, dtype=torch.bool)
    loss_criterion = create_criterion(dataset)
    model, is_gcn = create_model(params=params, input_dim=features.size(1))
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=20)
    cost_val = []
    # Train model
    dataset = dataset.to(DEVICE)
    best_val_acc = 0
    train_logger, val_logger = create_logger()
    for epoch in range(epochs):
        model.train()
        t = time.time()
        optimizer.zero_grad()  # Clear gradients.
        # Construct feed dictionary
        pred = forward_pass(dataset, model)
        loss = loss_criterion(pred[dataset.train_idx],
                         dataset.y[dataset.train_idx])  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer.step()
        # Training step
        # one_hot = torch.nn.functional.one_hot(dataset.y)
        train_acc = accuracy(pred[dataset.train_idx], labels=dataset.y[dataset.train_idx], is_gcn=is_gcn)
        pred = torch.softmax(pred, dim=1)[:, 1]
        train_auc = sklearn.metrics.roc_auc_score(dataset.y[dataset.train_idx].cpu().numpy(),
                                                  pred[dataset.train_idx].cpu().detach().numpy())

        # Validation
        with torch.no_grad():
            model.eval()
            pred = forward_pass(dataset, model)
            cost = loss_criterion(pred[dataset.val_idx],
                             dataset.y[dataset.val_idx])
            cost_val.append(cost.item())
            val_acc = accuracy(pred[dataset.val_idx], labels=dataset.y[dataset.val_idx], is_gcn=is_gcn)
            # one_hot = torch.nn.functional.one_hot(dataset.y)
            pred = torch.softmax(pred, dim=1)[:, 1]
            val_auc = sklearn.metrics.roc_auc_score(dataset.y[dataset.val_idx].cpu().numpy(),
                                                    pred[dataset.val_idx].cpu().detach().numpy())

        # scheduler.step(cost.item())
        log_results(cost, epoch, loss, t, train_acc, train_auc, train_logger, val_acc, val_auc, val_logger)
        # Save model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save()

    print("Optimization Finished!")

    # Testing
    # load the best model weights
    model.load()
    with torch.no_grad():
        model.eval()
        pred = forward_pass(dataset, model)
        test_cost = loss_criterion(pred[dataset.test_idx], dataset.y[dataset.test_idx])
        test_acc = accuracy(pred[dataset.test_idx], labels=dataset.y[dataset.test_idx], is_gcn=is_gcn)
        # one_hot = torch.nn.functional.one_hot(dataset.y)
        pred = torch.softmax(pred, dim=1)[:, 1]
        test_auc = sklearn.metrics.roc_auc_score(dataset.y[dataset.test_idx].cpu().numpy(),
                                                 pred[dataset.test_idx].cpu().detach().numpy())

    print("Test set results:", "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc),
          "auc=", "{:.5f}".format(test_auc))

    return test_acc, test_auc


def log_results(cost, epoch, loss, t, train_acc, train_auc, train_logger, val_acc, val_auc, val_logger):
    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()),
          "train_acc=", "{:.5f}".format(train_acc), "train_auc=", "{:.5f}".format(train_auc), "val_loss=",
          "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(val_acc), "val_auc=", "{:.5f}".format(val_auc), "time=",
          "{:.5f}".format(time.time() - t))
    # Plot the results on Tensorbord
    train_logger.add_scalar("acc", train_acc, epoch)
    train_logger.add_scalar("auc", train_auc, epoch)
    train_logger.add_scalar("loss", loss.item(), epoch)
    val_logger.add_scalar("acc", val_acc, epoch)
    val_logger.add_scalar("auc", val_auc, epoch)
    val_logger.add_scalar("loss", cost.item(), epoch)


def forward_pass(dataset, model):
    if isinstance(model, GCN):
        pred = model.predict(dataset.x, dataset.edge_index, dataset.edge_attr)
    else:
        pred = model.predict(dataset.x)
    return pred
