import os.path
import datetime
import cv2
import numpy as np
#from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity
from src.utils import metrics
from src.utils import preprocess
import tensorflow as tf

def train(model, ims, real_input_flag, configs, itr, ims_reverse=None):
    
    ims = ims[:, :configs.total_length]
    ims_list = np.split(ims, configs.n_gpu)
    cost = model.train(ims_list, configs.lr, real_input_flag)

    flag = 1

    if configs.reverse_img:
        ims_rev = np.split(ims_reverse, configs.n_gpu)
        cost += model.train(ims_rev, configs.lr, real_input_flag)
        flag += 1

    if configs.reverse_input:
        ims_rev = np.split(ims[:, ::-1], configs.n_gpu)
        cost += model.train(ims_rev, configs.lr, real_input_flag)
        flag += 1
        if configs.reverse_img:
            ims_rev = np.split(ims_reverse[:, ::-1], configs.n_gpu)
            cost += model.train(ims_rev, configs.lr, real_input_flag)
            flag += 1

    cost = cost / flag

    if itr % configs.display_interval == 0:
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'itr: ' + str(itr))
        print('training loss: ' + str(cost))

def graph_test(self, inputs, real_input_flag, configs):
    feed_dict = {self.x[i]: inputs[i] for i in range(1)}
    feed_dict.update({real_input_flag: real_input_flag})
    gen_ims, debug = self.sess.run((self.pred_seq, self.debug), feed_dict)
    return gen_ims, debug

def test(model, test_input_handle, configs, save_name):
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'test...')
    test_input_handle.begin(do_shuffle=False)
    res_path = os.path.join(configs.gen_frm_dir, str(save_name))
    os.mkdir(res_path)
    avg_mse = 0
    batch_id = 0
    img_mse, ssim, psnr, fmae, sharp = [], [], [], [], []

    for i in range(configs.total_length - configs.input_length):
        img_mse.append(0)
        ssim.append(0)
        psnr.append(0)
        fmae.append(0)
        sharp.append(0)

    if configs.img_height > 0:
        height = configs.img_height
    else:
        height = configs.img_width

    real_input_flag = np.zeros(
        (configs.batch_size,
         configs.total_length - configs.input_length - 1,
         configs.img_width // configs.patch_size,
         height // configs.patch_size,
         configs.img_depth // configs.patch_size,
         configs.patch_size ** 3 * configs.img_channel)) #(4, 18, 16, 16, 16, 8): tensor
    f_flag=True
    while not test_input_handle.no_batch_left():
        batch_id = batch_id + 1

        test_ims = test_input_handle.get_batch()
        if batch_id%48==0:
            print('output case {}'.format(batch_id))
        else:
            test_input_handle.next()
            continue
        test_ims = test_ims[:, :configs.total_length]
        if len(test_ims.shape) > 4:
            test_dat = preprocess.reshape_patch(test_ims, configs.patch_size)
        else:
            test_dat = test_ims
        test_dat = np.split(test_dat, configs.n_gpu)

        img_gen, debug = model.test(test_dat, real_input_flag)

        img_gen = np.concatenate(img_gen)
        if len(img_gen.shape) > 3:
            img_gen = preprocess.reshape_patch_back(img_gen, configs.patch_size)
        # MSE per frame
        for i in range(configs.total_length - configs.input_length):
            x = test_ims[:, i + configs.input_length, :, :, :]
            x = test_ims[:, i + configs.input_length, :, :, :, :]
            x = x[:configs.batch_size * configs.n_gpu]
            x = x - np.where(x > 10000, np.floor_divide(x, 10000) * 10000, np.zeros_like(x))
            #gx = img_gen[:, i, :, :, :]
            gx = img_gen[:, i, :, :, :, :]
            fmae[i] += metrics.batch_mae_frame_float(gx, x)
            gx = np.maximum(gx, 0)
            gx = np.minimum(gx, 1)
            mse = np.square(x - gx).sum()
            img_mse[i] += mse
            avg_mse += mse
            real_frm = np.uint8(x * 255) #255でいいのか？
            pred_frm = np.uint8(gx * 255)
            psnr[i] += metrics.batch_psnr(pred_frm, real_frm)

        if batch_id%48==0:
            path = os.path.join(res_path, str(batch_id))
            os.mkdir(path)
            if len(debug) != 0:
                np.save(os.path.join(path, "f.npy"), debug)

            np.save(os.path.join(path, "gt.npy"), test_ims)
            for i in range(configs.total_length):
                name = 'gt' + str(i + 1) + '.png'
                file_name = os.path.join(path, name)
                #img_gt = np.uint8(test_ims[0, i, :, :, :] * 255)
                #print("test_ims.shape=",test_ims.shape)
                img_gt = np.uint8(test_ims[0, i, :, :, 15, :] * 255)
                if configs.img_channel == 2:
                    #img_gt = img_gt[:, :, :1]
                    img_gt = img_gt[ :, :, :1]
                cv2.imwrite(file_name, img_gt)

            np.save(os.path.join(path, "pd.npy"), img_gen)
            for i in range(configs.total_length - configs.input_length):
                name = 'pd' + str(i + 1 + configs.input_length) + '.png'
                file_name = os.path.join(path, name)
                img_pd = img_gen[0, i, :, :, 15, :]
                if configs.img_channel == 2:

                    img_pd = img_pd[ :, :, :1]
                    #print("img_pd.shape after=", img_pd.shape)

                img_pd = np.maximum(img_pd, 0)
                img_pd = np.minimum(img_pd, 1)
                img_pd = np.uint8(img_pd * 255)
                cv2.imwrite(file_name, img_pd)
        if f_flag:
            test_input_handle.next()


    avg_mse = avg_mse / (batch_id * configs.batch_size * configs.n_gpu)
    print('mse per seq: ' + str(avg_mse))
    for i in range(configs.total_length - configs.input_length):
        print(img_mse[i] / (batch_id * configs.batch_size * configs.n_gpu))
    psnr = np.asarray(psnr, dtype=np.float32) / batch_id
    fmae = np.asarray(fmae, dtype=np.float32) / batch_id
    sharp = np.asarray(sharp, dtype=np.float32) / (configs.batch_size * batch_id)

    print('psnr per frame: ' + str(np.mean(psnr)))
    for i in range(configs.total_length - configs.input_length):
        print(psnr[i])
    print('fmae per frame: ' + str(np.mean(fmae)))
    for i in range(configs.total_length - configs.input_length):
        print(fmae[i])
    print('sharpness per frame: ' + str(np.mean(sharp)))
    for i in range(configs.total_length - configs.input_length):
        print(sharp[i])
