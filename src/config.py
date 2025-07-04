import os

# DUMP xem ở đây: https://dumps.wikimedia.org/other/enterprise_html/runs
DIR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = r'F:/data' if os.name == 'nt' else '/media/hoanchan/USB-BACKUP/data'

TEST_IDS = 400  # Number of tables to test
NUM_PROCESSES = os.cpu_count()  # One threads per logic core

# Vocabulary files
STRUCTURE_VOCAB_FILE = os.path.join(DIR_ROOT, 'data', 'structure_vocab.txt')
CELL_VOCAB_FILE = os.path.join(DIR_ROOT, 'data', 'cell_vocab.txt')

# Processing options
PUBTABNET_IMAGE_ROOT = os.path.join(DATA_ROOT, 'pubtabnet')
PUBTABNET_TEST_JSON = os.path.join(PUBTABNET_IMAGE_ROOT, 'test.jsonl')

VITABSET_IMAGE_ROOT = os.path.join(DATA_ROOT, 'vitabset')
VITABSET_TRAIN_JSON = os.path.join(VITABSET_IMAGE_ROOT, 'train.jsonl')
VITABSET_VAL_JSON = os.path.join(VITABSET_IMAGE_ROOT, 'val.jsonl')
VITABSET_TEST_JSON = os.path.join(VITABSET_IMAGE_ROOT, 'test.jsonl')