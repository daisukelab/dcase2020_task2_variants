from image_common import *
import warnings
warnings.filterwarnings("ignore")


DATA_ROOT = Path('/data/task2/dev') # set your data folder here
SAVE_TO = Path('./dev_data')

types = [t.name for t in sorted(DATA_ROOT.glob('*'))]
print('Machine types:', types)

df = pd.DataFrame()
df['file'] = sorted(DATA_ROOT.glob('*/*/*.wav'))
df['type'] = df.file.map(lambda f: f.parent.parent.name)
df['split'] = df.file.map(lambda f: f.parent.name)

print('=== DCASE 2020 Challenge Task 2 Data Preprocessing LEVEL 1 ===')

for t in types:
    for split in ['train', 'test']:
        type_df = df[df['type'] == t][df.split == split].reset_index()
        # convert first file, we want to know the shape
        mels = get_log_mel_spectrogram(type_df.file[0])
        # create big bucket to keep all the data
        all_mels = np.zeros((len(type_df), mels.shape[0], mels.shape[1]))
        filename = f'dc2020t2l1-{t}-{split}.npy'
        # convert all files
        print(f'{filename}: making {len(type_df)} log mel spectrogram data as shape {all_mels.shape}')
        all_mels[0, :, :] = mels
        print(type_df.file[0])
        for i, f in enumerate(type_df.file[1:]):
            print(f)
            all_mels[i, :, :] = get_log_mel_spectrogram(f)
        np.save(SAVE_TO/filename, all_mels)

print('done!')
