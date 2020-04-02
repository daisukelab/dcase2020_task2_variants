from dlcliche.utils import *
from common import *


# set location of your copy here
DATA_ROOT = Path('/data/task2/add_dev_data')

types = [t.name for t in sorted(DATA_ROOT.glob('*')) if t.is_dir()]
print(types)

df = pd.DataFrame()
df['file'] = sorted(DATA_ROOT.glob('*/*/*.wav'))
df['type'] = df.file.map(lambda f: f.parent.parent.name)
df['split'] = df.file.map(lambda f: f.parent.name)


def get_wave_info(filename):
    wav, sampling_rate = file_load(filename)
    print(filename)
    return wav.shape[0], sampling_rate, wav.shape[-1]/sampling_rate


df['infos'] = df.file.apply(lambda f: get_wave_info(f))

df['frames'] = df.infos.map(lambda x: x[0])
df['sampling_rate'] = df.infos.map(lambda x: x[1])
df['sec'] = df.infos.map(lambda x: x[2])

del df['infos']
df.set_index('file').to_csv('additional_file_info.csv')
