# coding:utf-8
import six.moves.cPickle as pickle
import six
import chainer
from chainer import cuda
import chainer.functions as F
from chainer import optimizers
import cv2
import numpy as np
from dataArgs import DataArgumentation

# NN設定
batchsize = 90
n_epoch = 200
IMGSIZE = 64

# データセット読み込み
x_train = []
x_test = []
y_train = []
y_test = []

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

import os
files = os.listdir('./')
for file in files:
    if len(file) == 4:
        # 平仮名ディレクトリ
        _unicode = file
        imgs = os.listdir('./' + _unicode + '/')
        counter = 0
        for img in imgs:
            if img.find('.png') > -1:
                if len(imgs) - counter != 1:
                    src = cv2.imread('./' + _unicode + '/' + img, 0)
                    src = cv2.resize(src, (IMGSIZE, IMGSIZE))
                    dargs = DataArgumentation(src)
                    src = cv2.bitwise_not(src)
                    x_train.append(src)
                    y_train.append(unicode2number[_unicode])
                    for x in xrange(1, 10):
                        dst = dargs.argumentation([2, 3])
                        ret, dst = cv2.threshold(
                            dst, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        x_train.append(dst)
                        y_train.append(unicode2number[_unicode])
                        # cv2.imshow('ARGUMENTATED', dst)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                else:
                    src = cv2.imread('./' + _unicode + '/' + img, 0)
                    src = cv2.resize(src, (IMGSIZE, IMGSIZE))
                    dargs = DataArgumentation(
                        cv2.bitwise_not(cv2.bitwise_not(src)))

                    for x in xrange(1, 5):
                        dst = dargs.argumentation([2, 3])
                        ret, dst = cv2.threshold(
                            dst, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        x_test.append(dst)
                        y_test.append(unicode2number[_unicode])
                counter = counter + 1


# 読み込んだデータを0~1に正規化，numpy.arrayに変換
x_train = np.array(x_train).astype(np.float32).reshape(
    (len(x_train), 1, IMGSIZE, IMGSIZE)) / 255
y_train = np.array(y_train).astype(np.int32)
x_test = np.array(x_test).astype(np.float32).reshape(
    (len(x_test), 1, IMGSIZE, IMGSIZE)) / 255
y_test = np.array(y_test).astype(np.int32)
N = len(y_train)
N_test = len(y_test)

# モデルの設定
model = chainer.FunctionSet(conv1=F.Convolution2D(1, 8, 3),
                            bn1=F.BatchNormalization(8),
                            conv2=F.Convolution2D(8, 16, 3, pad=1),
                            bn2=F.BatchNormalization(16),
                            conv3=F.Convolution2D(16, 16, 3, pad=1),
                            fl4=F.Linear(1024, 512),
                            fl5=F.Linear(512, 83))


def forward(x_data, y_data, train=True):
    x, t = chainer.Variable(x_data, volatile=not train), chainer.Variable(
        y_data, volatile=not train)
    h = F.max_pooling_2d(F.relu(model.bn1(model.conv1(x))), 2)
    h = F.max_pooling_2d(F.relu(model.bn2(model.conv2(h))), 2)
    h = F.max_pooling_2d(F.relu(model.conv3(h)), 2)
    h = F.dropout(F.relu(model.fl4(h)), train=train)
    y = model.fl5(h)
    return F.softmax_cross_entropy(y, t), F.accuracy(y, t)


# オプティマイザーの設定
optimizer = optimizers.Adam()
optimizer.setup(model.collect_parameters())


# 学習のループ
for epoch in six.moves.range(1, n_epoch + 1):
    print('epoch', epoch)
    # 学習
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, N, batchsize):
        x_batch = x_train[perm[i:i + batchsize]]
        y_batch = y_train[perm[i:i + batchsize]]
        optimizer.zero_grads()
        loss, acc = forward(x_batch, y_batch)
        loss.backward()
        optimizer.update()
        sum_loss += float(cuda.to_cpu(loss.data)) * len(y_batch)
        sum_accuracy += float(cuda.to_cpu(acc.data)) * len(y_batch)
        # 動作確認のため定期的に出力
        if i % 10000 == 0:
            print 'COMPUTING...'

    print('train mean loss={}, accuracy={}'.format(
        sum_loss / len(y_train), sum_accuracy / len(y_train)))

    # テスト
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, N_test, batchsize):
        x_batch = x_test[i:i + batchsize]
        y_batch = y_test[i:i + batchsize]

        loss, acc = forward(x_batch, y_batch, train=False)
        sum_loss += float(cuda.to_cpu(loss.data)) * len(y_batch)
        sum_accuracy += float(cuda.to_cpu(acc.data)) * len(y_batch)

    print 'test mean loss={}, accuracy={}'.format(sum_loss / len(y_test), sum_accuracy / len(y_test))

    # モデルの保存
    pickle.dump(model, open('model' + str(epoch), 'wb'), -1)
