import time
import os
import queue
from torch import optim
# from multiprocessing import Process

import matplotlib.pyplot as plt
from encoder import *
from decoder import *
from seq2seq import *
from parameter import *
from preprocess import *


def init_dir():
    default_dir = os.path.exists(DATA_PATH)
    if not default_dir:
        os.makedirs(DATA_PATH)
    default_dir = os.path.exists(MODEL_PATH)
    if not default_dir:
        os.makedirs(MODEL_PATH)
    default_dir = os.path.exists(PLOT_PATH)
    if not default_dir:
        os.makedirs(PLOT_PATH)


def save_model(encoder, decoder, in_lang, out_lang, epoch):
    model = {
        'encoder': encoder,
        'decoder': decoder,
        'in_lang': in_lang,
        'out_lang': out_lang,
    }
    torch.save(model,
               MODEL_PATH + '/%s-%s-%d.'
               % (in_lang.name, out_lang.name, epoch))
    torch.save(model,
               MODEL_PATH + '/last_model')


def load_model(model_path):
    model = torch.load(model_path)
    return model['encoder'], model['decoder'], model['in_lang'], model['out_lang']


# Keep track of time elapsed and running averages
def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)  # put ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show(block=False)
    # plt.close()


# All things start from
def seqlite(train, msg_que=None):   # msg_que would be assigned, if use ui
    if train:
        train_model(msg_que)
    else:
        use_model(msg_que)


# Print msg, and put it to queue.
def put_msg(msg_que, msg):
    print(msg)
    try:
        msg_que.put(msg)
    except:
        pass


# Train model
def train_model(msg_que, load=True):
    # Init necessary directories
    init_dir()

    # Init Corpus
    in_corpus = Corpus(IN_LANG)
    msg = "Corpus %s: %d sentences." % (in_corpus.name, in_corpus.n_lines)
    put_msg(msg_que, msg)

    out_corpus = Corpus(OUT_LANG)
    msg = "Corpus %s: %d sentences." % (out_corpus.name, out_corpus.n_lines)
    put_msg(msg_que, msg)

    # Init Lang
    in_lang = Lang(IN_LANG)
    out_lang = Lang(OUT_LANG)

    # Filter corpus by sentlen
    if FILTER_LEN:
        n_lines1, n_lines2 = filter_sentlen(in_corpus, out_corpus)
        msg = "filter_sentlen: filtered to %d and %d sentences" % (n_lines1, n_lines2)
        put_msg(msg_que, msg)

    # Lang load Corpus
    in_lang.load_corpus(in_corpus)
    msg = 'Lang ' + in_lang.name + ': Loading corpus...'
    put_msg(msg_que, msg)

    out_lang.load_corpus(out_corpus)
    msg = 'Lang ' + out_lang.name + ': Loading corpus...'
    put_msg(msg_que, msg)

    # Lang trim its vocab
    if TRIM_VOCAB:
        vocab_size0, vocab_size1 = trim_vocab(in_lang, MIN_COUNT)
        msg = 'Lang ' + in_lang.name + ': vocab trimed %s / %s (%.4f)' % (vocab_size1, vocab_size0, vocab_size1 / vocab_size0)
        put_msg(msg_que, msg)
        vocab_size0, vocab_size1 = trim_vocab(out_lang, MIN_COUNT)
        msg = 'Lang ' + out_lang.name + ': vocab trimed %s / %s (%.4f)' % (vocab_size1, vocab_size0, vocab_size1 / vocab_size0)
        put_msg(msg_que, msg)

    # Filter corpus by unknow words
    if FILTER_UNK:
        n_lines1, n_lines2 = filter_unkword(in_corpus, out_corpus, in_lang, out_lang)
        msg = 'filter_unkword: filtered to %d and %d sentences' % (n_lines1, n_lines2)
        put_msg(msg_que, msg)

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    # Initialize models
    encoder = EncoderRNN(in_lang.n_words, HIDDEN_SIZE,
                         N_LAYERS, dropout=DROPOUT)
    decoder = LuongAttnDecoderRNN(
        ATTN_MODEL, HIDDEN_SIZE, out_lang.n_words, N_LAYERS, dropout=DROPOUT)

    # Initialize optimizers and criterion
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
    decoder_optimizer = optim.Adam(
        decoder.parameters(), lr=LEARNING_RATE * DECODER_LEARNING_RATIO)
    criterion = nn.CrossEntropyLoss()

    # Move models to GPU
    if USE_CUDA:
        encoder.cuda()
        decoder.cuda()

    eca = 0
    dca = 0
    epoch = 0

    # Load last model and continue
    if load:
        try:
            encoder, decoder, in_lang, out_lang = load_model(MODEL_PATH + '/last_model')
        except:
            put_msg(msg_que, 'Fail loading model! Please check the model file in ' + MODEL_PATH)

    put_msg(msg_que, 'Training start...\n')
    while epoch < N_EPOCHS:
        epoch += 1
        # Get training data for this cycle
        input_batches, input_lengths, target_batches, target_lengths = random_batch(
            in_corpus, out_corpus, in_lang, out_lang, BATCH_SIZE)

        # Run the train function
        loss, ec, dc = train(
            input_batches, input_lengths, target_batches, target_lengths,
            encoder, decoder,
            encoder_optimizer, decoder_optimizer, criterion
        )

        # if epoch % 25 == 0:
        # print(loss)
        # print(input_batches)
        # Keep track of loss
        print_loss_total += loss
        plot_loss_total += loss
        eca += ec
        dca += dc

        # job.record(epoch, loss)
        if epoch % PRINT_EVERY == 0:
            print_loss_avg = print_loss_total / PRINT_EVERY
            print_loss_total = 0
            print_summary = '%s (%d %d%%) %.4f' % (time_since(
                start, epoch / N_EPOCHS), epoch, epoch / N_EPOCHS * 100, print_loss_avg)
            put_msg(msg_que, print_summary)

        if epoch % EVALUATE_EVERY == 0:
            save_model(encoder, decoder, in_lang, out_lang, epoch)
            # batch size = 1, only 1 sample
            input_batches, input_lengths, target_batches, _ = random_batch(
                in_corpus, out_corpus, in_lang, out_lang, 1)
            input_sentence, output_sentence, target_sentence = evaluate_and_show_attention(
                input_batches, input_lengths, in_lang, out_lang, target_batches, encoder, decoder, epoch)

            print_summary = 'Evaluation:' + '\n'
            print_summary += ' in/src:' + input_sentence + '\n'
            print_summary += ' out:' + output_sentence + '\n'
            if target_sentence is not None:
                print_summary += ' tgt:' + target_sentence + '\n'
            put_msg(msg_que, print_summary)

        if epoch % PLOT_EVERY == 0:
            plot_loss_avg = plot_loss_total / PLOT_EVERY
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    show_plot(plot_losses)


# Use model
def use_model(in_que, out_que):
    # Restore the model
    try:
        encoder, decoder, in_lang, out_lang = load_model(MODEL_PATH + '/last_model')
    except:
        put_msg(out_que, 'Fail loading model! Please check the model file in ' + MODEL_PATH)
        return

    # Get input and run
    in_line = ''
    out_line = ''
    if in_que and not in_que.empty():
        in_line = in_que.get()
    else:
        in_line = input('input: ')

    if in_line == '':
        pass
    else:
        in_line, length = to_batch(in_lang, in_line)
        _, out_line, _ = evaluate_and_show_attention(in_line, length, in_lang, out_lang, '', encoder, decoder, 0)
        put_msg(out_que, out_line)


if __name__ == "__main__":
    seqlite(train=TRAIN)
