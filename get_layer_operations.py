#encoding:utf-8
import tensorflow as tf
import numpy as np
import os
import sys
import argparse
from tensorflow.python.platform import gfile


def load_model(model):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))

def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files)==0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files)>1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups())>=2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    #print("123",meta_file, ckpt_file)
    return meta_file, ckpt_file
def display_nodes(nodes):
    for i, node in enumerate(nodes):
        print('%d %s %s' % (i, node.name, node.op))
        [print(u'└─── %d ─ %s' % (i, n)) for i, n in enumerate(node.input)]
        # if i>2000:
        #     break
def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('model_dir', type=str,
        help='Directory containing the metagraph (.meta) file and the checkpoint (ckpt) file containing model parameters or .pb model =')
    parser.add_argument('output_file', type=str,
        help='Filename for the exported graphdef protobuf layer operations')
    return parser.parse_args(argv)

def main(args):
    with tf.Graph().as_default():

        with tf.Session() as sess:
            load_model(args.model_dir)
            all_oprations = tf.get_default_graph().get_operations()

            with open(args.output_file,'a') as fw:
                for i in range(len(all_oprations)):
                    fw.write(str(all_oprations[i].name))
                    fw.write('\n')
            # #print(all_oprations)

    # graph = tf.GraphDef()
    # with tf.gfile.Open('/home/biao/test/resnettime_test/models/facenet/20171123-175923.pb', 'rb') as f:
    #     data = f.read()
    #     graph.ParseFromString(data)
    #
    # display_nodes(graph.node)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
