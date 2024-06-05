# VALLE v2

## Installation requirement 

Set up your environemnt as in Amphion README (e.g. by running `sh env.sh`).

espeak-ng is required to run G2p. To install it, you could refer to: 
https://github.com/espeak-ng/espeak-ng/blob/master/docs/guide.md

For Linux, it should be `sudo apt-get install espeak-ng`.
For Windows, refer to the above link.
If you do not have sudo privilege, you could build the library by following the last section of this readme.

## Inference pretrained VALLE models
We provide our pretrained VALLE model that is trained on 45k hours MLS dataset.
You could r

## The model files
File `valle_ar.py` and `valle_nar.py` in this folder are models files, these files can be run directly via `python -m models.tts.valle_gpt_simple.valle_ar` (or `python -m models.tts.valle_gpt_simple.valle_nar`, and will invoke a test which overfits it to a single example.

## Preparing dataset and dataloader
Write your own dataloader for your dataset. 
You can reference the `__getitem__` method in `models/tts/valle_gpt_simple/mls_dataset.py`
It should return a dict of a 1-dimensional tensor 'speech', which is a 24kHz speech; and a 1-dimensional tensor of 'phone', which is the phoneme sequence of the speech.

As long as your dataset returns this in `__getitem__`, it should work.

## Training AR with MLS dataset (Large Dataset)
```bash
sh ./egs/tts/valle_gpt_simple/train_ar_mls.sh
```
## Training NAR with MLS dataset (Large Dataset)
```bash
sh ./egs/tts/valle_gpt_simple/train_nar_mls.sh
```
# Practice with your own dataset (`dev-clean` in [LibriTTS](https://www.openslr.org/60) for example) 

In this part we will use the 'dev-clean' part of the LibriTTS dataset as training data for our AR model. 

**The section with [TODO] mark needs to be modified according to your environment.⭐⭐⭐**

You need to fill in the square brackets such as `[YourDatadir]`
## [Download `dev-clean` Dataset](https://www.openslr.org/60)

You can practice with smaller datasets using `egs/tts/valle_gpt_simple/train_ar_libritts.sh` 

Here we gonna use smaller dataset in **LibriTTS** `dev-clean.tar.gz` 1.2 GB as example.

To download dataset dev-clean use this command:

```bash
cd /[YourWorkspace]/data/
weget https://openslr.magicdatatech.com/resources/60/dev-clean.tar.gz
```

If your download speed is slow ,we recommand you use your own VPN download from US mirror `https://us.openslr.org/resources/60/dev-clean.tar.gz` and upload it to your server.

## [Prepare Dataloader](../../../models/tts/valle_gpt_simple/libritts_dataset.py) (Create Your Dataset File)

Make file `libritts_dataset.py`

**[TODO] Add your datadir to the dict dataset2dir**

```py
        ######## add data dir to dataset2dir ##########
        self.dataset2dir = {
            'dev-clean' : '/[YourWorkspace]/data/LibriTTS',
        }
```

Initialize metadata_cache and transcripts_cache and filter duration between 3s and 15s:

You can try 3s to 10s for less memory.

```py
    def __init__(self, args):
        # set dataframe clumn name
        book_col_name = ["ID", "Original_text", "Normalized_text", "Aligned_or_not", "Start_time", "End_time", "Signal_to_noise_ratio"]
        trans_col_name = ["ID", "Original_text", "Normalized_text","Dir_path","Duration"]
        self.metadata_cache = pd.DataFrame(columns = book_col_name)
        self.trans_cache = pd.DataFrame(columns = trans_col_name)

        ###### load metadata and transcripts #####
        for dataset_name in self.dataset_list:
            # get [book,transcripts,audio] files list
            self.book_files_list = self.get_metadata_files(self.dataset2dir[dataset_name])
            self.trans_files_list = self.get_trans_files(self.dataset2dir[dataset_name])

            
            ## create metadata_cache (book.tsv file is not filtered, some file is not exist, but contain Duration and Signal_to_noise_ratio)
            for book_path in self.book_files_list:
                tmp_cache = pd.read_csv(book_path, sep = '\t', names=book_col_name, quoting=3)
                self.metadata_cache = pd.concat([self.metadata_cache, tmp_cache], ignore_index = True)
            self.metadata_cache.set_index('ID', inplace=True)

            ## create transcripts (the trans.tsv file)
            for trans_path in self.trans_files_list:
                tmp_cache = pd.read_csv(trans_path, sep = '\t', names=trans_col_name, quoting=3)
                tmp_cache['Dir_path'] = os.path.dirname(trans_path)
                self.trans_cache = pd.concat([self.trans_cache, tmp_cache], ignore_index = True)
            self.trans_cache.set_index('ID', inplace=True)

            ## calc duration
            self.trans_cache['Duration'] = self.metadata_cache.End_time[self.trans_cache.index]-self.metadata_cache.Start_time[self.trans_cache.index]
            ## add fullpath
            # self.trans_cache['Full_path'] = os.path.join(self.dataset2dir[dataset_name],self.trans_cache['ID'])

        # filter_by_duration: filter_out files with duration < 3.0 or > 15.0
        print(f"Filtering files with duration between 3.0 and 15.0 seconds")
        print(f"Before filtering: {len(self.trans_cache)}")
        self.trans_cache = self.trans_cache[(self.trans_cache['Duration'] >= 3.0) & (self.trans_cache['Duration'] <= 15.0)]
        print(f"After filtering: {len(self.trans_cache)}")
```

Function getitem returns phone and speech sequence
```py
    def __getitem__(self, idx):
        # Get the file rel path
        file_dir_path = self.trans_cache['Dir_path'].iloc[idx]
        # Get uid
        uid = self.trans_cache.index[idx]
        # Get the file name from cache uid
        file_name = uid + '.wav'
        # Get the full file path
        full_file_path = os.path.join(file_dir_path, file_name)

        # get phone
        phone = self.trans_cache['Normalized_text'][uid]
        phone = phonemizer_g2p(phone, 'en')[1]
        # load speech
        speech, _ = librosa.load(full_file_path, sr=SAMPLE_RATE)
```

**You can test dataset.py by run `python -m models.tts.valle_gpt_simple.libritts_dataset`**

[TODO] If you want to test, don't forget change your workspace dir:

```py
def test():
    from utils.util import load_config
    cfg = load_config('/[YourWorkspace]/AmphionVALLEv2/egs/tts/valle_gpt_simple/exp_ar_libritts.json')
    dataset = VALLEDataset(cfg.trans_exp)
    metadata_cache = dataset.metadata_cache
    trans_cache = dataset.trans_cache
    print(trans_cache.head(10))
    # print(dataset.book_files_list)
    breakpoint()
```

run `python -m models.tts.valle_gpt_simple.libritts_dataset`

if you get a dataframe of transcripts with dir_path and Duration your dataset.py is correct 😊

## [Add your Dataloader to the Trainer](../../../models/tts/valle_gpt_simple/valle_ar_trainer.py)

Here we add our dataset class `libritts` add to the Trainer: `AmphionVALLEv2/models/tts/valle_gpt_simple/valle_ar_trainer.py`

```py
    ##########add your own dataloader to the trainer#############
    def _build_dataloader(self):
        from torch.utils.data import ConcatDataset, DataLoader
        if self.cfg.train.dataset.name == 'emilia':
            from .emilia_dataset import EmiliaDataset as VALLEDataset
            train_dataset = VALLEDataset()
        elif self.cfg.train.dataset.name == 'mls':
            from .mls_dataset import VALLEDataset as VALLEDataset
            train_dataset = VALLEDataset(self.cfg.trans_exp, resample_to_24k=True)
        elif self.cfg.train.dataset.name == 'libritts':
            from .libritts_dataset import VALLEDataset as VALLEDataset
            train_dataset = VALLEDataset()
```

## [Edit Training Config](exp_ar_libritts.json)

You can change the experiment settings in the `/AmphionVALLEv2/egs/tts/valle_gpt_simple/exp_ar_libritts.json` such as the learning rate, optimizer and the dataset.

**[TODO]: Set your log directory**

```json
"log_dir": "/[YourWorkspace]/AmphionVALLEv2/ckpt/valle_gpt_simple",
```

Here we choose `libritts` dataset we added and set `use_dynamic_dataset` false.

Config `use_dynamic_dataset` is used to solve the problem of inconsistent sequence length and improve gpu utilization, here we set it to false for simplicity.

```json
"dataset": {
          "use_mock_dataset": false,
          "use_dynamic_batchsize": false,
          "name": "libritts"
        },
```

Set our dataset list as `dev-clean` , `cache_dir` and `noise_dir` and `content_model_path` are not used, so don't worry about setting them.
```json
    "trans_exp": {
      "dataset_list":["dev-clean"],
      "cache_dir": "/mnt/workspace/lizhekai/data/exp_cache/tts/",
      "use_speaker": false,
      "use_noise": false,
      "noise_dir": "/mnt/workspace/lizhekai/data/noise/audiocaps",
      "content_model_path": "/mnt/petrelfs/hehaorui/data/pretrained_ckpts/mhubert"
    },
```

**Set smaller batch_size if you are out of memory😢😢**

I used batch_size=3 to successfully run on a single card, if you'r out of memory, try smaller.

```json
        "batch_size": 3,
        "max_tokens": 11000,
        "max_sentences": 64,
        "random_seed": 0
```

## [Prepare Training Script File](train_ar_libritts.sh)

To run the trainer you need to create experiment shell script `/AmphionVALLEv2/egs/tts/valle_gpt_simple/train_ar_libritts.sh`

**[TODO]: Set your experiment directory and work directory (fill the brackets with your own path)**

```bash
######## Build Experiment Environment ###########
exp_dir="/[YourWorkspace]/AmphionVALLEv2/egs/tts/valle_gpt_simple"
echo exp_dir
work_dir="/[YourWorkspace]/AmphionVALLEv2/"
echo work_dir
```

**Set the config file name.**

Lets fill in with the `exp_ar_libritts.json` file here

```bash
######## Set Config File Dir ##############
if [ -z "$exp_config" ]; then
    exp_config="${exp_dir}"/exp_ar_libritts.json
fi
echo "Exprimental Configuration File: $exp_config"
```

**And set the experiment name:**

```bash
######## Set the experiment name ##########
exp_name="ar_libritts_dev_clean"
```

## Training AR with dev-clean dataset 

Before the training, let's set the **CUDA_VISIBLE_DEVICE=0** in [train_ar_libritts.sh](train_ar_libritts.sh), for better debuging.

Please modify it according to your envirnment

```sh
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port $port "${work_dir}"/bins/tts/train.py --config $exp_config --exp_name $exp_name --log_level debug 
```

**[TODO]: Run the command to Train AR on `dev-clean` dataset**

```sh
sh egs/tts/valle_gpt_simple/train_ar_libritts.sh
```
Hope your code is running🏃🏃🏃🏃🏃

## RuntimeError: language "cmn" is not supported by the espeak backend
If you met the error`RuntimeError: language "cmn" is not supported by the espeak backend`

Comment this line out in [g2p.py](../../../utils/g2p/g2p.py)

```py
# 创建支持中文的 Espeak 后端
phonemizer_zh = EspeakBackend('cmn', preserve_punctuation=False, with_stress=False, language_switch="remove-flags")
# phonemizer_zh.separator = separator
```

# Run with Slurm 
首先通过 `sinfo`指令查看空闲的节点，在下面指定节点时需要指定一个空闲的节点。
首先将 `/nfsmnt/{yourname}/AmphionVALLEv2/egs/tts/valle_gpt_simple/train_ar_libritts.sh` 路径下的前八行改为这个样子：
```
#!/usr/bin/env bash
#SBATCH --job-name=train-valle-ar            # Job name
#SBATCH --output result.out         ## filename of the output
#SBATCH --nodes=1                   ## Run all processes on a single node
#SBATCH -w, --nodelist=node03             ## Request the pointed node 	
#SBATCH --ntasks=8                  ## number of processes = 20
#SBATCH --cpus-per-task=1           ## Number of CPU cores allocated to each process
#SBATCH --partition=Project         ## Partition name: Project or Debug (Debug is default)
```

## espeak-ng Installation

1. 浏览器搜索 `espeak-ng`，找到 GitHub 上的库，然后输入：
    ```sh
    git clone https://github.com/espeak-ng/espeak-ng.git
    ```

2. 下载完后执行：
    ```sh
    sh autogen.sh
    ./configure --prefix=你的路径/espeak
    ```

3. 接着输入：
    ```sh
    make
    make install
    ```

4. 修改 `.bashrc` 文件，加入这一行：
    ```sh
    export PATH=$PATH:/nfsmnt/{yourname}/espeak/bin
    ```

5. 然后更新：
    ```sh
    source .bashrc
    ```

## Modify g2p.py

修改 `utils` 下的 `g2p/g2p.py` 文件，在 `import` 下面加入这两行：

```python
_ESPEAK_LIBRARY = '/nfsmnt/{yourname}/espeak/lib/libespeak-ng.so.1.1.51'
EspeakBackend.set_library(_ESPEAK_LIBRARY)
```

## JSON Configuration

修改 `/nfsmnt/{yourname}/AmphionVALLEv2/egs/tts/valle_gpt_simple/exp_ar_libritts.json` 文件：

- 将 13 行的 `numworker` 改为 `4`
- 40 行的 `batchsize` 改为 `1`

## Modify Dataset Script

修改 `/nfsmnt/{yourname}/AmphionVALLEv2/models/tts/valle_gpt_simple/libritts_dataset.py` 下的 87 行，从 `25` 改为 `10`：

```python
self.trans_cache = self.trans_cache[(self.trans_cache['Duration'] >= 3.0) & (self.trans_cache['Duration'] <= 10.0)]
```

You can save the above content as a `.md` file. If you need any further modifications, let me know!

## Resume from the exist checkpoint

在 `/nfsmnt/qiujunwen/AmphionVALLEv2/egs/tts/valle_gpt_simple/train_ar_libritts.sh` 文件里的最后面的train model 改为：
```
echo "Experimental Name: $exp_name"
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port $port "${work_dir}"/bins/tts/train.py --config $exp_config --exp_name $exp_name --log_level debug \
    --resume \
    --resume_type "resume" \
    --resume_from_ckpt_path "/nfsmnt/qiujunwen/AmphionVALLEv2/ckpt/valle_gpt_simple/ar_libritts_dev_clean/checkpoint/epoch-0029_step-0092000_loss-0.348073"
```
ckpt_path 更改成你跑出来最后一个epoch的checkpoint文件夹路径。
