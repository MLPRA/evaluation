import argparse
import matplotlib.pyplot as plot

import chainer

from chainercv.links import SSD300
from chainercv import utils
from chainercv.visualizations import vis_bbox


def main():
    chainer.config.train = False

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--pretrained_model')
    parser.add_argument('image')
    parser.add_argument('--label_names', help='The path to the file with label names')
    args = parser.parse_args()

    with open(args.label_names, 'r') as f:
        label_names = f.read().splitlines()

    model = SSD300(
        n_fg_class=len(label_names),
        pretrained_model=args.pretrained_model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    img = utils.read_image(args.image, color=True)
    model.nms_thresh = 0.45
    model.score_thresh = 0.3
    bboxes, labels, scores = model.predict([img])
    print((bboxes, labels, scores))
    bbox, label, score = bboxes[0], labels[0], scores[0]

    vis_bbox(img, bbox, label, score, label_names=label_names)
    plot.show()


if __name__ == '__main__':
    main()
