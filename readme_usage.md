# DCASE 2020 baseline (Ver.1.0.0)




## Usage

### 1. unzip dataset

ZENODOからdev_data.zipとeval_data.zipをダウンロードしてください。  
ダウンロード後、dcase2020_baselineディレクトリで7zを利用して展開してください。


- ./dcase2020_baseline
    - /dev_data
    - /eval_data
    - /00_train.py
    - /01_test.py
    - /common.py
    - /readme.md

ディレクトリ構成が上記の構成になっていることを確認してください。


### 2. baseline system
- `00_train.py` , 音響データを利用してmodelを作成します。
- `01_test.py`  , modelを利用して異常検出が実行されます。

### 3.Application arguments

| Argument                    |                                   | Description                                                  |
| --------------------------- | --------------------------------- | ------------------------------------------------------------ |
| `-h`                        | `--help`                          | Application help.                                            |
| `-v`                        | `--version`                       | Show application version.                                    |
| `-e`                        | `--eval`                          | run mode Evaluation                                          |                     
| `-d`                        | `--dev`                           | run mode Development                                         |
```
$ python3.6 00_train.py [option]
```

コマンドライン引数でDevelopmentかEvaluationを指定して実行してください。

00_train.pyを実行すると　**model/**　に機種ごとのmodelが作成されます。

01_test.pyを実行すると　**result/** に個体ごとのanomaly_score_csvが作成されます。  
Development実行時のみAUC,pAUCのresult.csvが作成されます。

パラメーターを変更する場合は、baseline.yamlを編集してください

- model/ :  
	Training results are located.  
- result/ :  
	.csv file (default = result.csv) is located.  
	In the file, all result AUCs and pAUCs are written.

### 4. sample result
```  
ToyCar		
id	    AUC	        pAUC
1	    0.956937229	0.925837321
2	    0.990857143	0.97281884
3	    0.967105121	0.922939424
4	    0.998609164	0.992679813
Average	    0.978377165	0.953568849
		
ToyConveyor		
id	    AUC	        pAUC
1	    0.9999875	0.999934211
2	    0.994355634	0.972831727
3	    0.998096212	0.989980062
Average	    0.997479782	0.987582
		
fan		
id	    AUC	        pAUC
0	    0.592457002	0.498900815
2	    0.932423398	0.81762205
4           0.803390805	0.63415003
6	    0.928171745	0.78932789
Average	    0.814110738	0.685000196
		
pump		
id	    AUC	        pAUC
0	    0.771818182	0.813029076
2	    0.632522523	0.64817449
4	    0.9211	0.717894737
6	    0.840784314	0.7249742
Average	    0.791556255	0.726018126

  ...
```

## Dependency

We develop the source code on Ubuntu 16.04 LTS and 18.04 LTS.
In addition, we checked performing on **Ubuntu 16.04 LTS**, **18.04 LTS**, **Cent OS 7**, and **Windows 10**.

### Software packages
- p7zip-full
- Python == 3.6.5
- FFmpeg

### Python packages
- Keras                         == 2.1.5
- Keras-Applications            == 1.0.2
- Keras-Preprocessing           == 1.0.1
- matplotlib                    == 3.0.3
- numpy                         == 1.15.4
- PyYAML                        == 3.13
- scikit-learn                  == 0.20.2
- librosa                       == 0.6.0
- audioread                     == 2.1.5 (more)
- setuptools                    == 41.0.0
- tensorflow                    == 1.15.0