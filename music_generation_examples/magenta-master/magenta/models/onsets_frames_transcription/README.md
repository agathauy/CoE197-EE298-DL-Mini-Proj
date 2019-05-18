## Onsets and Frames: Dual-Objective Piano Transcription

State of the art piano transcription, including velocity estimation.

For original model details, see our paper on arXiv,
[Onsets and Frames: Dual-Objective Piano Transcription](https://goo.gl/magenta/onsets-frames-paper), and the accompanying [blog post](https://g.co/magenta/onsets-frames).

We have since made improvements to the model and released a new [training dataset](https://g.co/magenta/maestro-dataset). These are detailed in our paper, [Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset
](https://goo.gl/magenta/maestro-paper), and blog post, [The MAESTRO Dataset and Wave2Midi2Wave
](https://g.co/magenta/maestro-wave2midi2wave).

The code in this directory corresponds to the latest version from the MAESTRO paper. For code corresponding to the Onsets and Frames paper, please browse the repository at commit [9885adef](https://github.com/tensorflow/magenta/tree/9885adef56d134763a89de5584f7aa18ca7d53b6). Note that we can only provide support for the code at HEAD.

You may also be interested in a [PyTorch Onsets and Frames](https://github.com/jongwook/onsets-and-frames) implementation by [Jong Wook Kim](https://github.com/jongwook) (not supported by the Magenta team).

Finally, we have also open sourced the [align_fine](/magenta/music/alignment) tool for high performance fine alignment of sequences that are already coarsely aligned, as described in the "Fine Alignment" section of the Appendix in the [MAESTRO paper](https://goo.gl/magenta/maestro-paper).

## JavaScript App

The easiest way to try out the model is with our web app: [Piano Scribe](https://goo.gl/magenta/piano-scribe). You can try transcribing audio files right in your browser without installing any software. You can read more about it on our blog post, [Piano Transcription in the Browser with Onsets and Frames](http://g.co/magenta/oaf-js).

## Colab Notebook

We also provide an [Onsets and Frames Colab Notebook](https://goo.gl/magenta/onsets-frames-colab).

## Transcription Script

If you would like to run transcription locally, you can use the transcribe
script. First, set up your [Magenta environment](/README.md).

Next, download our pre-trained
[checkpoint](https://storage.googleapis.com/magentadata/models/onsets_frames_transcription/maestro_checkpoint.zip),
which is trained on the [MAESTRO dataset](https://g.co/magenta/maestro-dataset).

After unzipping that checkpoint, you can run the following command:

```bash
MODEL_DIR=<path to directory containing checkpoint>
onsets_frames_transcription_transcribe \
  --model_dir="${CHECKPOINT_DIR}" \
  <piano_recording1.wav, piano_recording2.wav, ...>
```

## Train your own

If you would like to train the model yourself, first set up your [Magenta environment](/README.md).

### MAESTRO Dataset

If you plan on using the default dataset creation setup, you can also just download a pre-generated copy of the TFRecord files that will be generated by the steps below: [onsets_frames_dataset_maestro_v1.0.0.zip](https://storage.googleapis.com/magentadata/models/onsets_frames_transcription/onsets_frames_dataset_maestro_v1.0.0.zip). If you modify any of the steps or want to use custom code, you will need to do the following steps.

For training and evaluation, we will use the [MAESTRO](https://g.co/magenta/maestro-dataset) dataset. These steps will process the raw dataset into training examples containing 20-second chunks of audio/MIDI and validation/test examples containing full pieces.

Our dataset creation tool is written using Apache Beam. These instructions will cover how to run it using Google Cloud Dataflow, but you could run it with any platform that supports Beam. Unfortunately, Apache Beam does not currently support Python 3, so you'll need to use Python 2 here.

To prepare the dataset, do the following:

1. Set up Google Cloud Dataflow. The quickest way to do this is described in [this guide](https://cloud.google.com/dataflow/docs/quickstarts/quickstart-python).
1. Run the following command:

```
BUCKET=bucket_name
PROJECT=project_name
MAGENTA_SETUP_PATH=/path/to/magenta/setup.py

PIPELINE_OPTIONS=\
"--runner=DataflowRunner,"\
"--project=${PROJECT},"\
"--temp_location=gs://${BUCKET}/tmp,"\
"--setup_file=${MAGENTA_SETUP_PATH}"

onsets_frames_transcription_create_dataset_maestro \
  --output_directory=gs://${BUCKET}/datagen \
  --pipeline_options="${PIPELINE_OPTIONS}" \
  --alsologtostderr
```

Depending on your setup, this could take up to a couple hours to run (on Google Cloud, this may cost around $20). Once it completes, you should have about 19 GB of files in the `output_directory`.

You could also train using Google Cloud, but these instructions assume you have downloaded the generated TFRecord files to your local machine.

### MAPS Dataset (Optional)

Training and evaluation will happen on the MAESTRO dataset. If you would also like to evaluate (or even train) on the MAPS dataset, follow these steps.

First, you'll need to download a copy of the
[MAPS Database](http://www.tsi.telecom-paristech.fr/aao/en/2010/07/08/maps-database-a-piano-database-for-multipitch-estimation-and-automatic-transcription-of-music/).
Unzip the MAPS zip files after you've downloaded them.

Next, you'll need to create TFRecord files that contain the relevant data from MAPS by running the following command:

```bash
MAPS_DIR=<path to directory containing unzipped MAPS dataset>
OUTPUT_DIR=<path where the output TFRecord files should be stored>

onsets_frames_transcription_create_dataset_maps \
  --input_dir="${MAPS_DIR}" \
  --output_dir="${OUTPUT_DIR}"
```

### Training

Now can train your own transcription model using the training TFRecord file generated during dataset creation.

Note that if you have the `audio_transform` hparam set to true (which it is by default), you will need to have the [sox](http://sox.sourceforge.net/) binary installed on your system.

Note that if you run a training or an eval job on a platform other than an NVIDIA GPU, you will need to add the argument `--hparams=use_cudnn=false` when running that job. This will use a cuDNN-compatible ops that can run on the CPU.

```bash
TRAIN_EXAMPLES=<path to training tfrecord(s) generated during dataset creation>
RUN_DIR=<path where checkpoints and summary events should be saved>

onsets_frames_transcription_train \
  --examples_path="${TRAIN_EXAMPLES}" \
  --run_dir="${RUN_DIR}" \
  --mode='train'
```

You can also run an eval job during training to check metrics:

```bash
TEST_EXAMPLES=<path to eval tfrecord(s) generated during dataset creation>
MODEL_DIR=<path where checkpoints should be loaded>
OUTPUT_DIR=$MODEL_DIR/eval

onsets_frames_transcription_infer \
  --examples_path="${TEST_EXAMPLES}" \
  --model_dir="${MODEL_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --hparams="use_cudnn=false" \
  --eval_loop
```

During training, you can check on progress using TensorBoard:

```bash
tensorboard --logdir="${RUN_DIR}"
```

### Inference

To get final performance metrics for the model, run the `onsets_frames_transcription_infer` script.

```bash
MODEL_DIR=<path where checkpoints should be loaded>
TEST_EXAMPLES=<path to eval tfrecord(s) generated during dataset creation>
OUTPUT_DIR=<path where output should be saved>

onsets_frames_transcription_infer \
  --model_dir="${CHECKPOINT_DIR}" \
  --examples_path="${TEST_EXAMPLES}" \
  --output_dir="${RUN_DIR}"
```

You can check on the metrics resulting from inference using TensorBoard:

```bash
tensorboard --logdir="${RUN_DIR}"
```