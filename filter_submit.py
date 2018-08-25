import os
import sys
import argparse


FLAGS = None
DEFAULT_SCORE = 0.05


def filter_submit(input_path, output_path, score):
    with open(input_path, 'r') as f_in:
        with open(output_path, 'w') as f_out:
            first_line = f_in.readline()
            f_out.write('ImageId,PredictionString\n')
            for line in f_in.readlines():
                line = line.strip()
                img_id, rest_of_line = line.split(',')
                print(img_id)
                boxes = rest_of_line.split(' ') if rest_of_line else []
                assert len(boxes) % 6 == 0
                box_strings = []
                for i in range(0, len(boxes), 6):
                    # 1 record: [label, score, x_min, y_min, x_max, y_max]
                    b = boxes[i:i+6]
                    if float(b[1]) >= score:
                        box_strings.append(' '.join(b))
                f_out.write('{},'.format(img_id))
                if box_strings:
                    f_out.write(' '.join(box_strings))
                f_out.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--score', type=float, default=DEFAULT_SCORE,
        help='score/confidence threshold (default {:f})'.format(DEFAULT_SCORE))
    parser.add_argument('input', help='input file name')
    parser.add_argument('output', help='output file name')

    FLAGS = parser.parse_args()

    filter_submit(FLAGS.input, FLAGS.output, FLAGS.score)
