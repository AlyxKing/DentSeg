# **DentSeg / フレキシブルなU-Netでの歯科セグメンテーション **


## **説明**

このプロジェクトは、フレキシブルなU-NetアーキテクチャのPyTorch実装を使用して、歯科X線画像のセグメンテーションを紹介します。["Half U-Net: A Simplified U-Net Architecture for Medical Image Segmentation"](https://www.frontiersin.org/articles/10.3389/fninf.2022.911679/full) の論文で詳述されているように、Half U-Netモードで動作する機能を導入します。さらに、この構造のバリエーションを組み込み、["GhostNet: More Features from Cheap Operations"](https://paperswithcode.com/method/ghost-module) と "GhostNetV2: Enhance Cheap Operation with Long-Range Attention" からのゴーストモジュールv2のコンセプトを統合して、最小限の計算要件で追加の特徴層を作成します。


## **特徴**



* **フレキシブルなU-Netアーキテクチャ**: 効率的な計算を実現しつつ、セグメンテーション性能を維持するためにHalf U-Netモードに調整可能。
* **ゴーストモジュールの統合**: 「安価な操作」を利用して追加の特徴層を生成し、モデルの能力を維持しつつ計算コストを節約。
* **設定可能なチャネル**: Half U-Net論文で提案されている方法論に従って、U-Net全体でチャネルを固定するオプションを提供。


## **データセット**

トレーニングに使用された歯科X線画像データセットは、[Humans in the Loop Dental x-ray imagery](https://www.kaggle.com/datasets/humansintheloop/teeth-segmentation-on-dental-x-ray-images) から入手されました。

データセットの簡易版[ アーカイブ](https://chat.openai.com/c/dentseg_dataset.tar.gz) が提供されています。


## **インストール**


### **ソースから**



1. ローカルマシンにリポジトリをクローンする：


```
git clone https://github.com/alyxking/dentseg.git

```



1. プロジェクトディレクトリに移動する：


```
cd dentseg

```



1. `requirements.txt` で詳細に説明されている必要な依存関係がインストールされていることを確認する。または、以下のようにコンテナ環境から実行する。


### **コンテナから**



1. Dockerイメージをプルする


```
docker pull ghcr.io/alyxking/dentseg:tag

```



1. データセットアーカイブを抽出する
2. コンテナを実行する（DATASET_HOST_PATH と SRC_HOST_PATH を設定）


```
docker run --rm --gpus all \
  -v "DATASET_HOST_PATH:/app/dataset" \
  -v "SRC_HOST_PATH:/app/src" \
  -p "80:80" \
  -p "8888:8888" \
  --name dentseg dentseg \
  jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
```



## **使用方法**

DentSegモデルは、データセットとトレーニング要件に合わせてさまざまなパラメータで設定して実行できます。デフォルト値と利用可能なオプションは以下の通りです。


### **設定パラメータ**

モデルを設定して実行するには、以下のコマンドライン引数を使用できます：



* **<code>--run_name</code>** (デフォルト: <code>DentSeg7</code>)：実行の名前を設定します。
* **<code>--epochs</code>** (デフォルト: <code>10</code>)：トレーニングのエポック数を指定します。
* **<code>--batch_size</code>** (デフォルト: <code>25</code>)：トレーニングのバッチサイズを設定します。
* **<code>--image_size</code>** (デフォルト: <code>256</code>)：入力画像のサイズを定義します。
* **<code>--dataset_path</code>** (デフォルト: <code>/app/dataset</code>)：データセットへのパスを提供します。パスには画像とアノテーションの両方が含まれるべきです。
* **<code>--device</code>** (デフォルト: <code>cuda:0</code>)：トレーニングに使用するデバイスを選択します（例：<code>'cuda:0'</code>）。
* **<code>--lossfunc</code>** (デフォルト: <code>DICEBCE</code>)：トレーニングに使用する損失関数を選択します。上記の利用可能な損失関数を参照してください。
* **<code>--evalfunc</code>** (デフォルト: <code>IOU</code>)：モデルの性能を評価するための評価関数を選択します。
* **<code>--lr</code>** (デフォルト: <code>1e-4</code>)：学習率を設定します。
* **<code>--in_c</code>** (デフォルト: <code>1</code>)：入力チャネルの数を指定します。
* **<code>--out_c</code>** (デフォルト: <code>1</code>)：出力チャネルの数を指定します。
* **<code>--flat</code>**：Half U-Net統一チャネル幅を選択するためにこのフラグを使用します。このフラグがない場合、標準のU-Netチャネルは各ダウンステップで倍増します。


### **モデルローディング設定パラメータ**

これらの追加パラメータでモデル設定をさらにカスタマイズできます：



* **<code>--load_model</code>**：トレーニングプロセスを開始する前に、事前に訓練されたモデルをロードするためにこのフラグを使用します。これはON/OFFフラグです；コマンドに含めてONにします。
* **<code>--model_name</code>** (デフォルト: <code>None</code>)：事前に訓練されたモデルをロードしており、それが<code>run_name</code>と異なる名前の場合、この引数でモデルの名前を指定します。これは、以前に保存されたモデル状態からトレーニングを続けたり、事前に訓練されたモデルを評価したりする場合に特に便利です。
* **<code>--eval</code>**：このフラグはモデルを評価モードのみに設定します。トレーニングはスキップされ、データセットでのテストのみが実行されます。
* **<code>--full_model</code>**：モデルのロード方法を制御するためのフラグです。デフォルトでは、モデルローディングメカニズムは状態辞書を期待しています。保存されたモデルがより複雑な構造（例：オプティマイザ、スケジューラ）を含む場合、このフラグを使用して完全なモデルをロードします。これはON/OFFフラグです；コマンドに含めてONにします。


#### **例コマンド**

CLIからトレーニング/評価を実行するには、以下の例のようなコマンドを使用します：


```
python dentsegdataset.py --run_name DentSeg5 --epochs 200 --batch_size 4 --image_size 512 --dataset_path /path/to/the/dataset --device cuda:0 --lossfunc DICEBCE --evalfunc IOU --lr 0.0001 --in_c 3 --out_c 3 --flat
```



#### **利用可能な損失関数**

モデルは、トレーニングと評価のためのさまざまな損失関数をサポートしています：



* **BCE:** 二値交差エントロピー損失
* **IOU:** 交差オーバーユニオン損失
* **DICE:** ダイス損失
* **DICEBCE:** ダイスとBCE損失の組み合わせ
* **FOCAL:** フォーカル損失
* **TVERSKY:** テバースキー損失
* **FOCALTVERSKY:** フォーカルテバースキー損失
* **DISCLOSS:** 判別損失（複数インスタンスのセグメンテーションのみ）


### **コンテナからの実行：**

含まれているipynbノートブックから実行するか、dockerコマンドに設定コマンドを追加するか、コンテナ内の端末に入ります。


## **ライセンス**

このプロジェクトはGNU General Public License v3.0の下でライセンスされています - 詳細については、[LICENSE](https://chat.openai.com/c/LICENSE)ファイルを参照してください。


## **謝辞**

このプロジェクトは、Half U-NetアーキテクチャとGhostNet技術に関する論文のアーキテクチャに基づいています。トレーニングは、Humans in the Loopの歯科X線画像データセットで実施されました。


## **参考文献**

```
@dataset{HumansInTheLoop2023,
  author       = {Humans In The Loop},
  title        = {{Teeth Segmentation on dental X-ray images}},
  year         = 2023,
  publisher    = {Kaggle},
  version      = {1},
  doi          = {10.34740/KAGGLE/DSV/5884500},
  url          = {https://www.kaggle.com/datasets/humansintheloop/teeth-segmentation-on-dental-xray-images}
}
```
```
@article{LuHaoran2022,
  author       = {Lu Haoran and She Yifei and Tie Jun and Xu Shengzhou},
  title        = {{Half-UNet: A Simplified U-Net Architecture for Medical Image Segmentation}},
  journal      = {Frontiers in Neuroinformatics},
  volume       = {16},
  year         = 2022,
  doi          = {10.3389/fninf.2022.911679},
  url          = {https://www.frontiersin.org/articles/10.3389/fninf.2022.911679},
  issn         = {1662-5196}
}
```
```
@article{Han2020GhostNet,
  author       = {Kai Han and Yunhe Wang and Qi Tian and Jianyuan Guo and Chunjing Xu and Chang Xu},
  title        = {{GhostNet: More Features from Cheap Operations}},
  year         = 2020,
  eprint       = {1911.11907},
  archivePrefix= {arXiv},
  primaryClass = {cs.CV},
  url          = {https://arxiv.org/abs/1911.11907}
}
