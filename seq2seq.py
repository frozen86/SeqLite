import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from masked_cross_entropy import *
from preprocess import *
from parameter import *
import time

# # Training


def train(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):

    batch_size = BATCH_SIZE
    clip = CLIP

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0  # Added onto for each word

    # Run through encoder
    encoder_outputs, encoder_hidden = encoder(
        input_batches, input_lengths, None)

    # Initialize decoder input
    decoder_input = torch.LongTensor([SOS_index] * batch_size)

    # Use last (forward) hidden state from encoder
    # encoder_hidden size: num_layers * num_directions(=2), batch, hidden_size
    # decoder_hidden size: num_layers, batch, hidden_size
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Find the max length
    max_target_length = max(target_lengths)

    # Initialize decoder output
    all_decoder_outputs = torch.zeros(
        max_target_length, batch_size, decoder.output_size)

    # Move new Variables to CUDA
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        all_decoder_outputs = all_decoder_outputs.cuda()

    # Run through decoder one time step at a time
    for t in range(max_target_length):
        decoder_output, decoder_hidden, decoder_attn = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )

        all_decoder_outputs[t] = decoder_output
        decoder_input = target_batches[t]  # Next input is current target

    # Loss calculation and backpropagation
    # loss_cal = nn.BCELoss()
    # loss = loss_cal(all_decoder_outputs, target_batches)
    # print("target:", target_batches.size())
    # print("output:", all_decoder_outputs.size())
    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
        target_batches.transpose(0, 1).contiguous(),  # -> batch x seq
        target_lengths
    )
    loss.backward()

    # Clip gradient norms
    ec = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    dc = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Update parameters with optimizers
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item(), ec, dc


# # Evaluating the network

# def evaluate(input_seq, max_length=MAX_LENGTH):
def evaluate(input_batches, input_lengths, input_lang, output_lang, encoder, decoder, max_length=MAX_LENGTH):

    # Set to not-training mode to disable dropout
    encoder.train(False)
    decoder.train(False)

    # Run through encoder
    encoder_outputs, encoder_hidden = encoder(
        input_batches, input_lengths, None)

    # Inference only, no back propagation
    with torch.no_grad():
        # Initialize decoder input
        decoder_input = torch.LongTensor([SOS_index])
        # Use last (forward) hidden state from encoder
        decoder_hidden = encoder_hidden[:decoder.n_layers]

    if USE_CUDA:
        decoder_input = decoder_input.cuda()

    # Store output words and attention states
    output_sindices = []
    decoder_attentions = torch.zeros(max_length + 1, max_length + 1)

    # Run through decoder
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attn = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        decoder_attentions[di, :decoder_attn.size(
            2)] += decoder_attn.squeeze(0).squeeze(0).cpu().data

        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        # Extract number from pytorch variable
        ni = ni.item()
        output_sindices.append(ni)
        if ni == EOS_index:
            break
        # Next input is chosen word
        decoder_input = torch.LongTensor([ni])
        if USE_CUDA:
            decoder_input = decoder_input.cuda()

    # Set back to training mode
    encoder.train(True)
    decoder.train(True)

    return output_sindices, decoder_attentions[:di + 1, :len(encoder_outputs)]


def evaluate_and_show_attention(input_sentence, input_length, input_lang, output_lang,
                                target_batches, encoder, decoder, epoch):
    sindices, attentions = evaluate(
        input_sentence, input_length, input_lang, output_lang, encoder, decoder)

    input_sentence = indices_to_sentence(input_lang, input_sentence)
    output_sentence = indices_to_sentence(output_lang, sindices)
    target_sentence = indices_to_sentence(output_lang, target_batches)

    print_summary = 'Evaluation:'+'\n'
    print_summary += ' in/src:' + input_sentence + '\n'
    print_summary += ' out:' + output_sentence + '\n'
    if target_sentence is not None:
        print_summary += ' tgt:' + target_sentence + '\n'
    show_attention(input_sentence, output_sentence, attentions, epoch)
    return input_sentence, output_sentence, target_sentence


def show_attention(input_sentence, output_sentence, attentions, epoch):
    # Set up figure with colorbar
    # print(attentions)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' '), rotation=90)
    ax.set_yticklabels([''] + output_sentence.split(' '))

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.savefig(PLOT_PATH + '/epoch-%d.png' % epoch)
    fig.savefig(PLOT_PATH + '/last.png')
    # plt.show(block=True)
    # plt.close()
