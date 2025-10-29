import os
import pandas as pd
import tensorflow as tf
import librosa
import numpy as np
import string


class TextTransform:
    """Mapira karaktere u intove i obrnuto"""
    def __init__(self, offset=0):
        alphabet = string.ascii_lowercase + " ,"
        self.offset = offset
        self.char_map = {char: idx + offset for idx, char in enumerate(alphabet)}
        self.index_map = {idx + offset: char for idx, char in enumerate(alphabet)}

    def text_to_int(self, text):
        return [self.char_map[char] for char in text if char in self.char_map]

    def int_to_text(self, labels):
        return ''.join([self.index_map[label] for label in labels if label in self.index_map])


class MelSpectrogramDataset:
    """Pravi Mel Spektrograme i enkodira tekst iz TSV metapodataka."""
    def __init__(self, tsv_files, audio_dir, text_transform):
        self.metadata = self._load_and_filter_metadata(tsv_files, audio_dir)
        self.audio_dir = audio_dir
        self.text_transform = text_transform

    def _load_and_filter_metadata(self, tsv_files, audio_dir):
        all_metadata = []
        for tsv_file in tsv_files:
            df = pd.read_csv(tsv_file, delimiter='\t')
            df = df[df['path'].apply(lambda x: os.path.exists(os.path.join(audio_dir, x)))]
            all_metadata.append(df)
        return pd.concat(all_metadata, ignore_index=True)

    def generator(self):
        for _, row in self.metadata.iterrows():
            audio_path = os.path.join(self.audio_dir, row['path'])
            label_text = row['sentence'].lower()

            label = tf.convert_to_tensor(self.text_transform.text_to_int(label_text), dtype=tf.int32)

            y, sr = librosa.load(audio_path, sr=None)
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
            S_dB = librosa.power_to_db(S, ref=np.max)
            S_min, S_max = S_dB.min(), S_dB.max()
            if S_max > S_min:
                S_dB = (S_dB - S_min) / (S_max - S_min)
            else:
                S_dB = np.zeros_like(S_dB)
            spectrogram = tf.convert_to_tensor(S_dB, dtype=tf.float32)
            spectrogram = tf.expand_dims(spectrogram, axis=-1)

            input_length = tf.shape(spectrogram)[1]
            label_length = tf.size(label)

            yield spectrogram, label, input_length, label_length

def serialize_example(spectrogram, label, input_length, label_length):
    spec_serialized = tf.io.serialize_tensor(spectrogram).numpy()
    label_serialized = tf.io.serialize_tensor(label).numpy()
    feature = {
        'spectrogram': tf.train.Feature(bytes_list=tf.train.BytesList(value=[spec_serialized])),
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_serialized])),
        'input_length': tf.train.Feature(int64_list=tf.train.Int64List(value=[input_length])),
        'label_length': tf.train.Feature(int64_list=tf.train.Int64List(value=[label_length])),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def write_tfrecord(dataset, tfrecord_path):
    """Upisuje Dataset u TFRecord fajl za brzo ucitavanje"""
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for spectrogram, label, input_length, label_length in dataset.generator():
            spec_np = spectrogram.numpy()
            spec_np = np.nan_to_num(spec_np, nan=0.0, posinf=0.0, neginf=0.0)
            label_np = label.numpy()
            input_length_np = int(input_length.numpy())
            label_length_np = int(label_length.numpy())

            example = serialize_example(spec_np, label_np, input_length_np, label_length_np)
            writer.write(example)
    print(f"TFRecord written to {tfrecord_path}")

def parse_example(serialized_example, max_width=1200, max_label_length=200):
    feature_description = {
        'spectrogram': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
        'input_length': tf.io.FixedLenFeature([], tf.int64),
        'label_length': tf.io.FixedLenFeature([], tf.int64)
    }

    example = tf.io.parse_single_example(serialized_example, feature_description)

    spectrogram = tf.io.parse_tensor(example['spectrogram'], out_type=tf.float32)
    spectrogram = tf.where(tf.math.is_finite(spectrogram), spectrogram, 0.0)
    label = tf.io.parse_tensor(example['label'], out_type=tf.int32)
    label = tf.where(label >= 1, label - 1, label)
    orig_input_length = example['input_length']
    label_length = example['label_length']

    spectrogram = tf.image.resize_with_pad(spectrogram, 128, max_width)
    spectrogram = tf.where(tf.math.is_finite(spectrogram), spectrogram, 0.0)
    spectrogram = tf.clip_by_value(spectrogram, 0.0, 1.0)
    input_length = tf.minimum(orig_input_length, tf.cast(max_width, tf.int64))
    label = tf.pad(label, [[0, max_label_length - tf.shape(label)[0]]], constant_values=-1)

    return spectrogram, label, input_length, label_length


def load_tfrecord_dataset(tfrecord_path, batch_size=16, shuffle=True, augment=False):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    mapper = parse_example_train if augment else parse_example
    dataset = dataset.map(mapper, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.filter(
        lambda spec, label, in_len, lab_len: tf.greater_equal(in_len, lab_len)
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset