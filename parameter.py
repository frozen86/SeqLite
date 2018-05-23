# Train model
# train model: TRAIN = True; use model: TRAIN = False
TRAIN = True

# Configure preprocess
FILTER_LEN = False   # Filter corpus by sentence length
FILTER_UNK = False  # Filter corpus by unknown word
TRIM_VOCAB = True   # Trim rare words from vocab
# Configure preprocess
MIN_LENGTH = 3
MAX_LENGTH = 25
MIN_COUNT = 5

# Configure language
IN_LANG = 'eng'
OUT_LANG = 'fra'

# Configure language functional token, DONT MAKE CHANGE
PAD_index = 0
SOS_index = 1
EOS_index = 2
UNK_index = 3
PAD_word = '[PAD]'
SOS_word = '[SOS]'
EOS_word = '[EOS]'
UNK_word = '[UNK]'

# Configure encoder and decoder
RNN = 'GRU'
# RNN = 'LSTM'
ATTN_MODEL = 'dot'
HIDDEN_SIZE = 500
N_LAYERS = 2
DROPOUT = 0.1
BATCH_SIZE = 10
USE_CUDA = True

# Configure training/optimization
CLIP = 50.0
TEACHER_FORCING_RATIO = 0.5
LEARNING_RATE = 0.0001
DECODER_LEARNING_RATIO = 5.0
N_EPOCHS = 50000

# Configure display
PLOT_EVERY = 20
PRINT_EVERY = 100
EVALUATE_EVERY = 500

# Configure Path
DATA_PATH = 'data'
MODEL_PATH = 'save/model'
PLOT_PATH = 'save/plot'
