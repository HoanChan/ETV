import os

DIR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = r'F:/data' if os.name == 'nt' else '/media/hoanchan/USB-BACKUP/data'

TEST_IDS = 400  # Number of tables to test
NUM_PROCESSES = os.cpu_count()  # One threads per logic core

# Vocabulary files
STRUCTURE_VOCAB_FILE = os.path.join(DIR_ROOT, 'src', 'data', 'structure_vocab.txt')
CELL_VOCAB_FILE = os.path.join(DIR_ROOT, 'src', 'data', 'cell_vocab.txt')

# PubTabNet dataset paths
PUBTABNET_ROOT = os.path.join(DATA_ROOT, 'pubtabnet')
PUBTABNET_TEST_IMAGE_ROOT = os.path.join(PUBTABNET_ROOT, 'test')
PUBTABNET_TEST_JSON = os.path.join(PUBTABNET_ROOT, 'test.jsonl')
# VitabSet dataset paths
VITABSET_ROOT = os.path.join(DATA_ROOT, 'vitabset')
VITABSET_TRAIN_IMAGE_ROOT = os.path.join(VITABSET_ROOT, 'train')
VITABSET_VAL_IMAGE_ROOT = os.path.join(VITABSET_ROOT, 'val')
VITABSET_TEST_IMAGE_ROOT = os.path.join(VITABSET_ROOT, 'test')
VITABSET_TRAIN_JSON = os.path.join(VITABSET_ROOT, 'train.bz2')
VITABSET_VAL_JSON = os.path.join(VITABSET_ROOT, 'val.bz2')
VITABSET_TEST_JSON = os.path.join(VITABSET_ROOT, 'test.bz2')