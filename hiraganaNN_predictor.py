# coding:utf-8
import numpy as np
import cv2
import six.moves.cPickle as pickle
import chainer
import chainer.functions as F
import argparse
from dataArgs import DataArgumentation


IMGSIZE = 64

# unicodeの順番で対応付け
unicode2number = {}
import csv
train_f = open('./hiragana_unicode.csv', 'rb')
train_reader = csv.reader(train_f)
train_row = train_reader
hiragana_unicode_list = []
counter = 0
for row in train_reader:
    for e in row:
        unicode2number[e] = counter
        counter = counter + 1


parser = argparse.ArgumentParser(description='READMODEL')
parser.add_argument('--img', '-i', default=0, help=' --sample.png')
parser.add_argument('--model', '-m', default=0, help=' --model23')
args = parser.parse_args()

model = pickle.load(open(args.model, 'rb'))


def forward(x_data, train=False):
    x = chainer.Variable(x_data, volatile=not train)
    h = F.max_pooling_2d(F.relu(model.bn1(model.conv1(x))), 2)
    h = F.max_pooling_2d(F.relu(model.bn2(model.conv2(h))), 2)
    h = F.max_pooling_2d(F.relu(model.conv3(h)), 2)
    h = F.dropout(F.relu(model.fl4(h)), train=train)
    y = model.fl5(h)
    return y.data


src = cv2.imread(args.img, 0)
# 余白がついていなくて、文字が大きすぎると誤認識の原因になっている。zero paddingなどで適切な文字サイズにしないといけない
# 本来は文字高さを取ってから適切なpaddingがしたいが、今はとりあえずこのまま
src = cv2.copyMakeBorder(
    src, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
src = cv2.resize(src, (IMGSIZE, IMGSIZE))

dargs = DataArgumentation(src)

# xtest = []
result = None
for x in xrange(0, 14):
    dst = dargs.argumentation([2, 3])
    # ret, dst = cv2.threshold(dst,
    #                          23,
    #                          255,
    #                          cv2.THRESH_BINARY)
    ret, dst = cv2.threshold(
        dst, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('ARGUMENTATED', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    xtest = np.array(dst).astype(np.float32).reshape(
        (1, 1, IMGSIZE, IMGSIZE)) / 255
    if result is None:
        result = forward(xtest)
    else:
        result = result + forward(xtest)

    tmp = np.argmax(forward(xtest))
    for strunicode, number in unicode2number.iteritems():
        if number == tmp:
            hiragana = unichr(int(strunicode, 16))
            print '候補　ニューロン番号:{0}, Unicode:{1}, ひらがな:{2}'.format(number, strunicode, hiragana.encode('utf_8'))


# print result
predict = np.argmax(result)

for strunicode, number in unicode2number.iteritems():
    if number == predict:
        hiragana = unichr(int(strunicode, 16))
        print '**最終判断　ニューロン番号:{0}, Unicode:{1}, ひらがな:{2}**'.format(number, strunicode, hiragana.encode('utf_8'))
# print unicode2number
