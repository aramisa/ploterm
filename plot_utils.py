#!/usr/bin/env python
# -*- coding: utf-8 -*-
# PYTHON_ARGCOMPLETE_OK
import numpy as np
import json


def showurl(u):
    import cv2
    cv2.namedWindow('tt', 1)
    _, crop, url = u.split('@')
    fname = '/data/db_imas/' + url.split('//')[-1]
    im = cv2.imread(fname)
    x, y, w, h = map(int, crop.replace(':', '+').split('+'))
    im = cv2.rectangle(im, (x, y), (x + w, y + h),
                       color=(255, 0, 0))
    cv2.imshow('tt', im)
    print u
    cv2.waitKey(0)
    cv2.destroyWindow('tt')
    cv2.waitKey(10)


def collage_triplets(data1, data2, data3,
                     mean_image=0, dotransp=True):
    if len(data3.shape) == 1 or data3.shape[1] == 1:
        # this are the labels for contrastive
        data3_ = np.ones(data1.shape, dtype='float32')
        for i in range(data3_.shape[0]):
            data3_[i, :, :, :] = ((data3[i] * 255) -
                                  mean_image.transpose(2, 0, 1))
        data3 = data3_
    if dotransp:
        data1 = [data1[p].transpose(1, 2, 0) for p in range(len(data1))]
        data2 = [data2[p].transpose(1, 2, 0) for p in range(len(data2))]
        data3 = [data3[p].transpose(1, 2, 0) for p in range(len(data3))]
    sep = 2
    imside = (data1[0].shape[0], data1[0].shape[1])
    per_row = int(1900 / (imside[0] + sep))
    num_rows = int(np.ceil(len(data1) / float(per_row)))
    newim = np.zeros(((imside[0] + sep) * 3 * num_rows + sep,
                     per_row * (imside[1] + sep) + sep, 3), dtype='uint8')
    i = 0
    for row in range(num_rows):
        starty = (imside[0] + sep) * 3 * row + sep
        for col in range(per_row):
            startx = (imside[1] + sep) * col + sep
            # ref
            newim[starty:starty + imside[0],
                  startx:startx + imside[1],
                  :] = (data1[i] + mean_image).astype('uint8')
            # pos
            newim[starty + imside[0] + sep:starty + 2 * imside[0] + sep,
                  startx:startx + imside[1],
                  :] = (data2[i] + mean_image).astype('uint8')
            # neg / label
            newim[starty + 2 * imside[0] + 2 * sep:starty +
                  3 * imside[0] + 2 * sep,
                  startx:startx + imside[1],
                  :] = (data3[i] + mean_image).astype('uint8')
            i += 1
            if i >= len(data1):
                break
        if i >= len(data1):
            break
    return newim


def collage_imas(data, mean_image=0, dotransp=True):
    if dotransp:
        data = [data[p].transpose(1, 2, 0) for p in range(len(data))]
    nimages = len(data)
    sep = 2
    shape = (int(np.ceil(nimages / np.ceil(np.sqrt(nimages)))),
             int(np.ceil(np.sqrt(nimages))))
    imside = (data[0].shape[0], data[0].shape[1])
    newim = np.zeros((shape[0] * (sep + imside[0]),
                      shape[1] * (sep + imside[1]), 3),
                     dtype='uint8')
    p = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            if p >= len(data):
                break
            newim[i * (sep + imside[0]):(i * (sep + imside[0])) + imside[0],
                  j * (sep + imside[1]):(j * (sep + imside[1])) + imside[1],
                  :] = (data[p] + mean_image).astype('uint8')
            p += 1
    return newim


def plot_progress(graph_queue, losses, max_iters, test_step):
    import matplotlib.pyplot as plt
    # losses = losses.keys()
    losses = losses[:]  # copying data
    losses.insert(0, 'all')
    nlosses = len(losses)
    accuracy_te = {}
    loss_tr = {}
    loss_te = {}
    for loss in losses:
        accuracy_te[loss] = np.nan * np.ones(max_iters)
        loss_tr[loss] = np.nan * np.ones(max_iters)
        loss_te[loss] = np.nan * np.ones(max_iters)

    gs = {}
    plt.ion()
    gs['fig_losstr'] = plt.figure()
    gs['fig_losste'] = plt.figure()
    gs['fig_accte'] = plt.figure()
    gs['lim_x'] = 1000
    subplot_shape = (np.ceil(nlosses / np.ceil(np.sqrt(nlosses))),
                     np.ceil(np.sqrt(nlosses)))
    for i in range(nlosses):
        loss = losses[i]
        gs['ax_losstr_' + loss] = gs['fig_losstr'].add_subplot(
            subplot_shape[0], subplot_shape[1], i + 1)
        gs['ax_losstr_' + loss].set_title(loss)
        gs['ax_losste_' + loss] = gs['fig_losste'].add_subplot(
            subplot_shape[0], subplot_shape[1], i + 1)
        gs['ax_losste_' + loss].set_title(loss)
        gs['ax_accte_' + loss] = gs['fig_accte'].add_subplot(
            subplot_shape[0], subplot_shape[1], i + 1)
        gs['ax_accte_' + loss].set_title(loss)
    for loss in losses:
        gs['h_losstr_' + loss], = gs['ax_losstr_' + loss].plot([], [])
        gs['h_losste_' + loss], = gs['ax_losste_' + loss].plot([], [])
        gs['h_accte_' + loss], = gs['ax_accte_' + loss].plot([], [])
        gs['ax_losstr_' + loss].set_autoscaley_on(True)
        gs['ax_losste_' + loss].set_autoscaley_on(True)
        gs['ax_accte_' + loss].set_autoscaley_on(True)
        gs['ax_losstr_' + loss].set_xlim(0, gs['lim_x'])
        gs['ax_losste_' + loss].set_xlim(0, gs['lim_x'])
        gs['ax_accte_' + loss].set_xlim(0, gs['lim_x'])
        gs['ax_losstr_' + loss].grid()
        gs['ax_losste_' + loss].grid()
        gs['ax_accte_' + loss].grid()

    while True:
        # get data from queue
        try:
            batch_plot = graph_queue.get(timeout=600)
        except:
            print 'graph: not getting new data to plot.'
            continue
        if batch_plot[0] == 'train':
            phase, step_loss_tr, prevstep, curstep = batch_plot
            # update internal variables
            for loss in step_loss_tr:
                loss_tr[loss][prevstep:curstep] = step_loss_tr[loss]
            if curstep + 100 > gs['lim_x']:
                gs['lim_x'] += 1000
                for loss in losses:
                    gs['ax_losste_' + loss].set_xlim(0, gs['lim_x'])
                    gs['ax_losstr_' + loss].set_xlim(0, gs['lim_x'])
                    gs['ax_accte_' + loss].set_xlim(0, gs['lim_x'])
            for loss in losses:
                gs['h_losstr_' + loss].set_xdata(range(len(loss_tr[loss])))
                gs['h_losstr_' + loss].set_ydata(loss_tr[loss])
                # update plot
                gs['ax_losstr_' + loss].relim()
                gs['ax_losstr_' + loss].autoscale_view()
            gs['fig_losstr'].canvas.draw()
            gs['fig_losstr'].canvas.flush_events()
        else:
            phase, step_acc_te, step_loss_te, test_iter = batch_plot
            # update internal variables
            for loss in step_acc_te:
                accuracy_te[loss][test_iter] = step_acc_te[loss]
            for loss in step_loss_te:
                loss_te[loss][test_iter] = step_loss_te[loss]
            for loss in losses:
                gs['h_accte_' + loss].set_xdata(test_step *
                                                np.arange(len(loss_te[loss])))
                gs['h_accte_' + loss].set_ydata(accuracy_te[loss])
                gs['h_losste_' + loss].set_xdata(test_step *
                                                 np.arange(len(loss_te[loss])))
                gs['h_losste_' + loss].set_ydata(loss_te[loss])
                # update plot
                gs['ax_accte_' + loss].relim()
                gs['ax_accte_' + loss].autoscale_view()
                gs['ax_losste_' + loss].relim()
                gs['ax_losste_' + loss].autoscale_view()
            gs['fig_losste'].canvas.draw()
            gs['fig_losste'].canvas.flush_events()
            gs['fig_accte'].canvas.draw()
            gs['fig_accte'].canvas.flush_events()


def plot_conv1_filters(net, stop=True, conv1name=None, shell=False):
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg')
    print 'plotting conv1'
    plt.figure(num=101, figsize=(10, 10))
    plt.clf()
    # set display defaults
    # plt.rcParams['figure.figsize'] = (10, 10)        # large images
    # don't interpolate: show square pixels
    # plt.rcParams['image.interpolation'] = 'nearest'
    # use grayscale output rather than a (potentially misleading) color heatmap
    # plt.rcParams['image.cmap'] = 'gray'
    if conv1name is None:
        conv1name = net.params.keys()[0]
    data = net.params[conv1name][0].data.transpose(0, 2, 3, 1)
    data = data[:, :, :, [2, 1, 0]]  # switch RGB to BGR
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))

    # add some space between filters
    # don't pad the last dimension (if there is one)
    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1)) +
               ((0, 0),) * (data.ndim - 3))
    # pad with ones (white)
    data = np.pad(data, padding, mode='constant', constant_values=1)

    # tile the filters into an image
    data = data.reshape((n, n) +
                        data.shape[1:]).transpose((0, 2, 1, 3) +
                                                  tuple(range(4,
                                                              data.ndim + 1)))
    data = data.reshape((n * data.shape[1],
                         n * data.shape[3]) + data.shape[4:])
    plt.axis('off')  # normalize data for display
    plt.imshow(data, interpolation='nearest', cmap='gray')
    plt.savefig('/tmp/conv1plot.png')
    if stop:
        plt.show()
    else:
        plt.ion()
        plt.show()
        plt.show()
        plt.show()
    # plt.imshow(data)
    # plt.savefig('/tmp/conv1plot.png')
    # plt.show()


def reduce_data(data, W):
    assert len(data.shape) or sum([x != 1 for x in data.shape[1:]]) == 0
    I = int(np.ceil(data.shape[0] / float(W)))
    # find good regime of splits
    ranges = [I for i in range(W)]
    idx = 0
    while sum(ranges) > data.shape[0]:
        ranges[idx] -= 1
        idx += 1
    dataW = np.array([np.mean(data[sum(ranges[:i]):sum(ranges[:i + 1])])
                      for i in range(W)])
    return dataW[~np.isnan(dataW)]


def simple_histogram(data):
    labels = set(data)
    minlab = min(labels)
    maxlab = max(labels)
    map_labels = {x: i for i, x in enumerate(range(int(minlab),
                                                   int(maxlab) + 1))}
    counts = np.zeros((len(map_labels),))
    for x in data:
        counts[map_labels[int(x)]] += 1
    return counts, len(map_labels)


def double_histogram(data1, data2, W=100, normed=True):
    # decide bins
    _, bins1 = np.histogram(data1)
    _, bins2 = np.histogram(data2)
    minb = min(bins1[0], bins2[0])
    maxb = max(bins1[-1], bins2[-1])
    bins = np.arange(minb, maxb, (maxb - minb) / float(W))
    return (np.histogram(data1, bins=bins, density=normed)[0],
            np.histogram(data2, bins=bins, density=normed)[0],
            bins)


def heatmap(data, H=20, W=40, cmap='jet', color_maps24=None):
    import cv2
    assert len(data.shape) == 2
    if color_maps24 is None:
        color_maps24 = json.load(open('colormaps.json'))
        color_maps24 = {k: np.array(x) for k, x in color_maps24.iteritems()}
    dataW = cv2.resize(data, (W, H * 2))
    vmin = np.min(dataW)
    vmax = np.max(dataW)
    ncolor = len(color_maps24[cmap])
    dataWint = np.floor((ncolor - 1) * (dataW - vmin) /
                        float(vmax - vmin)).astype(dtype=int)
    Ccm = color_maps24[cmap][dataWint]
    evenCcm = Ccm[0::2, :]
    oddCcm = Ccm[1::2, :]
    C = np.array([u'\033[38%s\033[48%s▄\033[0m' % (oddCcm[j, i], evenCcm[j, i])
                  for j in range(evenCcm.shape[0])
                  for i in range(Ccm.shape[1])]).reshape(H, W)
    for c in C:
        print u''.join(c)
    return C, dataW, color_maps24


def view_filters(net, H=11, W=11, conv1name=None, uncenter=127, unscale=255):
    if conv1name is None:
        conv1name = net.params.keys()[0]
    data = net.params[conv1name][0]
    filters = [view_filter(data[i, :, :, [2, 1, 0]],
                           uncenter=uncenter, unscale=unscale,
                           paint=False)[0]
               for i in range(data.shape[0])]
    # n = int(np.ceil(np.sqrt(len(filters))))
    # for i in range(n):
    #     for j in range(n):
    for C in filters:
        for c in C:
            print u''.join(c)


def view_filter(data, H=20, W=40, paint=True, uncenter=127, unscale=255):
    import cv2
    assert len(data.shape) == 3
    if data.shape[0] == 3:
        dataW = data.transpose((1, 2, 0))
    else:
        dataW = data
    dataW = cv2.resize(dataW, (W, H * 2))
    dataW = (dataW * unscale + uncenter).astype(int)
    C = np.array(
        [[u'\033[38%s\033[48%s▄\033[0m' %
          (';2;%d;%d;%dm' % tuple(reversed(list(dataW[oddrow + 1, col, :]))),
           ';2;%d;%d;%dm' % tuple(reversed(list(dataW[oddrow, col, :]))))
          for col in range(dataW.shape[1])]
         for oddrow in range(0, dataW.shape[0], 2)])
    if paint:
        for c in C:
            print u''.join(c)
    return C, dataW


def ascii_graph_multiple(data1, data2, W=200, H=20, minData=None, maxData=None,
                         color='basic', paint=True, grid=True, bins=None,
                         xticks=False):
    color_schemas = {'basic':
                     {
                         'background': '\x1b[2;23;40m',
                         'data1.b': '\x1b[1;31;40m',
                         'data2.b': '\x1b[1;34;40m',
                         'data1.2': '\x1b[0;31;104m',
                         'data2.1': '\x1b[0;34;101m',
                         'data1.2s': '\x1b[0;31;101m',
                         'data2.1s': '\x1b[0;34;104m',
                     },
                     }
    data1 = data1[~np.isnan(data1)]
    data2 = data2[~np.isnan(data2)]
    if (len(data1) == 0 or len(data1) == 1 or
       len(data2) == 0 or len(data2) == 1):
        return np.array(['0'], dtype='U20'), None, ['']
    assert data1.shape == data2.shape
    if data1.shape[0] > W:
        data1W = reduce_data(data1, W)
        data2W = reduce_data(data2, W)
    else:
        data1W = data1
        data2W = data2
    W = data1W.shape[0]
    fullbar_num = 9
    CMAP = {8: u'▇', 7: u'▆', 6: u'▅', 5: u'▄',
            4: u'▃', 3: u'▂', 2: u'▁', 1: u'_', 9: u'█'}
    h = len(CMAP) - 1

    C1 = np.array([0 for x in range(H * W)], dtype='uint8').reshape(H, W)
    C2 = np.array([0 for x in range(H * W)], dtype='uint8').reshape(H, W)

    if maxData is None:
        maxData = max(data1W.max(), data2W.max())
    if minData is None:
        minData = min(data1W.min(), data2W.min())
    if maxData == minData:
        # flat histogram
        if maxData == 0:
            # flat histogram at 0
            maxData = 0.1
        else:
            maxData = maxData + 0.1
            minData = 0
    # do two individual plots
    for C, dataW in zip([C1, C2], [data1W, data2W]):
        for n in range(W):
            P = min(maxData, max(minData, dataW[n]))
            R = H * ((P - minData) / float(maxData - minData))
            E = int(np.floor(R))
            F = R - E
            f = np.floor(h * F)
            for x in range(E):
                C[x, n] = fullbar_num
            if F != 0:
                C[E, n] = f + 1
    # decide who goes in front
    c2front = 0
    c1front = 0
    for n in range(W):
        for x in range(H):
            if C1[x, n] == 0 or C2[x, n] == 0:
                continue
            elif C1[x, n] == fullbar_num and C2[x, n] == fullbar_num:
                continue
            elif C1[x, n] == fullbar_num and C2[x, n] != fullbar_num:
                c2front += 1
            elif C2[x, n] == fullbar_num and C1[x, n] != fullbar_num:
                c1front += 1
            elif C2[x, n] > C1[x, n]:
                c2front += 1
            else:
                c1front += 1

    c_end = '\x1b[0m'
    c_bg = color_schemas[color]['background']
    if c1front > c2front:
        CF = C1
        CB = C2
        c_df = color_schemas[color]['data1.b']
        c_db = color_schemas[color]['data2.b']
        c_dfb = color_schemas[color]['data1.2']
        c_dbf = color_schemas[color]['data1.2s']
    else:
        CF = C2
        CB = C1
        c_df = color_schemas[color]['data2.b']
        c_db = color_schemas[color]['data1.b']
        c_dfb = color_schemas[color]['data2.1']
        c_dbf = color_schemas[color]['data2.1s']
        # join plots
    C = np.array([u'' for x in range(H * W)], dtype='U20').reshape(H, W)
    for n in range(W):
        for x in range(H):
            if CF[x, n] == 0 and CB[x, n] == 0:
                C[x, n] = c_bg + (u'_' if grid else ' ') + c_end
            elif CF[x, n] == fullbar_num and CB[x, n] == 0:
                C[x, n] = c_df + CMAP[CF[x, n]] + c_end
            elif CF[x, n] == fullbar_num and CB[x, n] != 0:
                C[x, n] = c_dbf + CMAP[CB[x, n]] + c_end
            elif CB[x, n] == fullbar_num and CF[x, n] == 0:
                C[x, n] = c_db + CMAP[CB[x, n]] + c_end
            elif CF[x, n] < fullbar_num and CB[x, n] == fullbar_num:
                C[x, n] = c_dfb + CMAP[CF[x, n]] + c_end
            elif CF[x, n] >= CB[x, n]:
                C[x, n] = c_df + CMAP[CF[x, n]] + c_end
            elif CF[x, n] < CB[x, n]:
                C[x, n] = c_db + CMAP[CB[x, n]] + c_end
    C = np.flipud(C)
    yaxis = (['%.2f' % maxData + '|'] +
             ['|' if y % 2 == 0
              else '%.2f' % (maxData - (y + 1) *
                             (maxData - minData) / H) +
              '|' for y in range(H - 2)] + ['%.2f' % minData + '|'])
    tail = None
    if H == 1:
        tail = '|%.2f' % minData
    ys = str(max([len(x) for x in yaxis]))
    formatted_plot = []
    for ya, x in zip(yaxis, C):
        fmt = '{:>' + ys + '}'
        formatted_plot.append(fmt.format(ya) + ''.join(x) +
                              (tail if tail is not None else ''))
        if paint:
            print formatted_plot[-1]
    if xticks:
        lntick = 9 + 1  # len(str(max_x)) + 3
        nticks = W / lntick - 1
        lntick = W / nticks - 1  # recompute real length
        lnleft = W - nticks * (lntick + 1)
        lnleft = nticks - lnleft
        if bins is None:
            xtickvals = np.linspace(0, len(data1), nticks + 1)
        else:
            xtickvals = np.array(bins)[::bins.shape[0] / nticks]
            # np.array([bins[i] for i in range(0, W, nticks + 1)])
        xs1 = '{: >' + str(lntick + 1) + '.3g}|'
        xs = '{: >' + str(lntick) + '.3g}|'
        xticks = ('{: >' + str(int(ys) - 1) + '.1f}|').format(xtickvals[0])
        for i, xt in enumerate(xtickvals[1:]):
            if lnleft > 0:
                xticks += xs.format(xt)
                lnleft -= 1
            else:
                xticks += xs1.format(xt)
        formatted_plot.append(xticks)
        if paint:
            print xticks
    return C, (data1W, data2W), formatted_plot


def ascii_graph_simple(data, W=200, H=20, minData=None, maxData=None,
                       color=None, paint=True, xticks=False):
    color_schemas = {'basic':
                     {
                         'odd': '',
                         'even': ''
                     },
                     'stripes':
                     {
                         'odd': ';33;40',
                         'even': ';32;40'
                     }
                     }
    data = data[~np.isnan(data)]
    # not used for now # min_x = 0
    max_x = len(data)
    if len(data) == 0 or len(data) == 1:
        return np.array(['0'], dtype='U20'), None, ['']
    if data.shape[0] > W:
        dataW = reduce_data(data, W)
    else:
        dataW = data
    W = dataW.shape[0]
    fullbar = u'█'
    CMAP = {7: u'▇', 6: u'▆', 5: u'▅', 4: u'▄',
            3: u'▃', 2: u'▂', 1: u'▁'}
    h = len(CMAP)
    icolodd = ('' if color is None
               else u'\x1b[0' + color_schemas[color]['odd'] + 'm')
    icoleven = ('' if color is None
                else u'\x1b[0' + color_schemas[color]['even'] + 'm')
    iend = '' if color is None else u'\x1b[0m'
    C = np.array([(icolodd if x % 2 == 1 else icoleven) + ' ' + iend
                  for x in range(H * W)], dtype='U20').reshape(H, W)
    if maxData is None:
        maxData = dataW.max()
    if minData is None:
        minData = dataW.min()
    if maxData == minData:
        # flat histogram
        if maxData == 0:
            # flat histogram at 0
            maxData = 0.1
        else:
            maxData = maxData + 0.1
            minData = 0
    for n in range(W):
        P = min(maxData, max(minData, dataW[n]))
        R = H * ((P - minData) / float(maxData - minData))
        E = int(np.floor(R))
        F = R - E
        f = np.floor(h * F)
        for x in range(E):
            C[x, n] = (icolodd if x % 2 == 1 else icoleven) + fullbar + iend
        if F != 0:
            C[E, n] = (icolodd if E % 2 == 1
                       else icoleven) + CMAP[f + 1] + iend
    C = np.flipud(C)
    yaxis = (['%.2f' % maxData + '|'] +
             ['|' if y % 2 == 0 else '%.2f' % (maxData - (y + 1) *
                                               (maxData - minData) / H) +
              '|' for y in range(H - 2)] + ['%.2f' % minData + '|'])
    tail = None
    if H == 1:
        tail = '|%.2f' % minData
    ys = str(max([len(x) for x in yaxis]))
    formatted_plot = []
    for ya, x in zip(yaxis, C):
        fmt = '{:>' + ys + '}'
        formatted_plot.append(fmt.format(ya) + ''.join(x) +
                              (tail if tail is not None else ''))
        if paint:
            print formatted_plot[-1]
    if xticks:
        lntick = len(str(max_x)) + 3
        nticks = W / max(1, lntick - 1)
        lntick = W / max(1, nticks - 1)  # recompute real length
        lnleft = W - nticks * (lntick + 1)
        lnleft = nticks - lnleft
        xtickvals = np.linspace(0, max_x, nticks + 1)
        xs1 = '{: >' + str(lntick + 1) + '.1f}|'
        xs = '{: >' + str(lntick) + '.1f}|'
        xticks = (int(ys) - 2) * u'¯' + u'Ō|'
        for i, xt in enumerate(xtickvals[1:]):
            if lnleft > 0:
                xticks += xs.format(xt)
                lnleft -= 1
            else:
                xticks += xs1.format(xt)
        formatted_plot.append(xticks)
        print xticks
    return C, dataW, formatted_plot


def ascii_graph(data, W=200, H=20, minData=None, maxData=None, color=None):
    # maybe useless because:
    # stackoverflow.com/questions/20295646/python-ascii-plots-in-terminal#20411508
    def adjust_point(h, H, P, minD, maxD):
        P = min(maxD, max(minD, P))
        R = H * ((P - minD) / float(maxD - minD))
        E = int(np.floor(R))
        F = R - E
        f = np.floor(h * F)
        return E, F, f

    color_schemas = {'basic':
                     {
                         'odd': '',
                         'even': '',
                         'oddN': '7',
                         'evenN': '7'
                     },
                     'stripes':
                     {
                         'odd': '0;33;40',
                         'even': '0;32;40',
                         'oddN': '0;30;43',
                         'evenN': '0;30;42'
                     }
                     }
    if data.shape[0] > W:
        dataW = reduce_data(data, W)
    else:
        dataW = data
    W = dataW.shape[0]
    fullbar = u'█'
    CMAP_pos = {7: u'▇', 6: u'▆', 5: u'▅', 4: u'▄',
                3: u'▃', 2: u'▂', 1: u'▁'}
    CMAP_neg = {7: u'▁', 6: u'▂', 5: u'▃', 4: u'▄',
                3: u'▅', 2: u'▆', 1: u'▇'}
    icolodd = ('' if color is None
               else u'\x1b[' + color_schemas[color]['odd'] + 'm')
    icoleven = ('' if color is None
                else u'\x1b[' + color_schemas[color]['even'] + 'm')
    icoloddN = (u'\x1b[' + color_schemas['basic']['evenN'] + 'm'
                if color is None
                else u'\x1b[' + color_schemas[color]['oddN'] + 'm')
    icolevenN = (u'\x1b[' + color_schemas['basic']['evenN'] + 'm'
                 if color is None
                 else u'\x1b[' + color_schemas[color]['evenN'] + 'm')
    iend = '' if color is None else u'\x1b[0m'
    h = len(CMAP_pos)
    if maxData is None:
        maxData = dataW.max()
    if minData is None:
        minData = dataW.min()
    if minData < 0 and maxData > 0:
        R = H * ((0 - minData) / float(maxData - minData))
        E = int(np.floor(R))
        Hn = E
        Hp = H - Hn
        Cp = np.array([(icolodd if x % 2 == 1 else icoleven) + ' ' + iend
                       for x in range(Hp * W)], dtype='<U20').reshape(Hp, W)
        Cn = np.array([(icolodd if x % 2 == 1 else icoleven) + ' ' + iend
                       for x in range(Hn * W)], dtype='<U20').reshape(Hn, W)
    elif minData < 0 and maxData < 0:
        Cn = np.array([(icolodd if x % 2 == 1 else icoleven) + ' ' + iend
                       for x in range(H * W)], dtype='<U20').reshape(H, W)
        Cp = None
    else:
        Cp = np.array([(icolodd if x % 2 == 1 else icoleven) + ' ' + iend
                       for x in range(H * W)], dtype='<U20').reshape(H, W)
        Cn = None
    for n in range(W):
        if dataW[n] < 0:
            E, F, f = adjust_point(h, Hn, np.abs(dataW[n]),
                                   np.abs(min(0, maxData)),
                                   np.abs(minData))
            for x in range(E):
                Cn[x, n] = (icolodd if x % 2 == 1
                            else icoleven) + fullbar + iend
            if F != 0:
                Cn[E, n] = (icoloddN if E % 2 == 1
                            else icolevenN) + CMAP_neg[f + 1] + u'\x1b[0m'
        else:
            E, F, f = adjust_point(h, Hp, dataW[n], max(0, minData), maxData)
            for x in range(E):
                Cp[x, n] = (icolodd if x % 2 == 1
                            else icoleven) + fullbar + iend
            if F != 0:
                Cp[E, n] = (icolodd if E % 2 == 1
                            else icoleven) + CMAP_pos[f + 1] + iend

    if Cp is not None and Cn is not None:
        yaxisp = (['%.2f' % maxData + '|'] +
                  ['|' if y % 2 == 0 else '%.2f' % (maxData - (y + 1) *
                                                    (maxData - 0) / Hp) +
                   '|' for y in range(Hp - 2)] + ['0.00' + '|'])
        yaxisn = ['|' if y % 2 == 0 else '%.2f' % (y * minData / Hn) +
                  '|' for y in range(Hn - 1)] + ['%.2f' % minData + '|']
        yaxis = yaxisp + yaxisn
    else:
        yaxis = (['%.2f' % maxData + '|'] +
                 ['|' if y % 2 == 0
                  else '%.2f' % (maxData - (y + 1) *
                                 (maxData - minData) / H) +
                  '|' for y in range(H - 2)] + ['%.2f' % minData + '|'])
    if Cn is not None and Cp is not None:
        C = np.vstack((np.flipud(Cp), Cn))
    elif Cn is None:
        C = np.flipud(Cp)
    else:
        C = Cn
    ys = str(max([len(x) for x in yaxis]))
    formatted_plot = []
    for ya, x in zip(yaxis, C):
        fmt = '{:>' + ys + '}'
        formatted_plot.append(fmt.format(ya) + ''.join(x))
        print formatted_plot[-1]
    return C, dataW, formatted_plot


if __name__ == "__main__":
    import sys
    import argcomplete
    import argparse
    sys.path.insert(1, 'external/caffe/python')
    import caffe
    parser = argparse.ArgumentParser()
    parser.add_argument('-weights', help='Model weights')
    parser.add_argument('-deploy', help='Model deploy prototxt')
    parser.add_argument('-conv1',
                        help='Plot figure with conv1 filters',
                        action='store_true', default=False)
    parser.add_argument('-conv1name',
                        help="""Name of the conv1 layer,
                        otherwise the first layer of the network is used""",
                        default=None)
    parser.add_argument('-gpu', help='CUDA device to use', type=int, default=0)
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    assert args.weights is not None and args.deploy is not None

    # load Net
    # device_use = args.gpu
    # caffe.set_device(device_use)
    # caffe.set_mode_gpu()
    net = caffe.Net(args.deploy, args.weights, caffe.TEST)
    if args.conv1 is True:
        plot_conv1_filters(net, conv1name=args.conv1name)
        # view_filters(net, H=11, W=11)
