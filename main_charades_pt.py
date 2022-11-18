import gc
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
from ctrl_model_pt import CTRL_Model
from dataset_charades_pt import TrainingDataset, TestingDataset, TestingDataset_sent, TestingDataset_vis
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim
from six.moves import xrange
import time
import pickle
import numpy as np
import operator
import time

def ctrl_collate_fn(batch):
    batch_size = len(batch)
    v_feat = torch.zeros(batch_size, 4096*3)
    s_feat = torch.zeros(batch_size, 4800)
    offset = torch.zeros(batch_size, 2)
    vp_feat = torch.zeros(batch_size, 100)
    tp_feat = torch.zeros(batch_size, 100)
    for i, item in enumerate(batch):
        v_feat[i] = torch.tensor(item[0])
        s_feat[i] = torch.tensor(item[1])
        offset[i][0] = torch.tensor(item[2][0])
        offset[i][1] = torch.tensor(item[2][1])
        vp_feat[i] = torch.tensor(item[3])
        tp_feat[i] = torch.tensor(item[4])
    return [v_feat, s_feat, offset, vp_feat, tp_feat]

def ctrl_collate_fn_test(batch):
    batch_size  = len(batch)
    v_feat = torch.zeros(batch_size, 4096*3)
    s_feat = torch.zeros(batch_size, 4800)
    for i, item in enumerate(batch):
        v_feat[i] = torch.tensor(item[0])
        s_feat[i] = torch.tensor(item[1])
    return [v_feat, s_feat]

def nms_temporal(x1,x2,s, overlap):
    pick = []
    assert len(x1)==len(s)
    assert len(x2)==len(s)
    if len(x1)==0:
        return pick

    #x1 = [b[0] for b in boxes]
    #x2 = [b[1] for b in boxes]
    #s = [b[-1] for b in boxes]
    union = list(map(operator.sub, x2, x1)) # union = x2-x1
    I = [i[0] for i in sorted(enumerate(s), key=lambda x:x[1])] # sort and get index

    while len(I)>0:
        i = I[-1]
        pick.append(i)

        xx1 = [max(x1[i],x1[j]) for j in I[:-1]]
        xx2 = [min(x2[i],x2[j]) for j in I[:-1]]
        inter = [max(0.0, k2-k1) for k1, k2 in zip(xx1, xx2)]
        o = [inter[u]/(union[i] + union[I[u]] - inter[u]) for u in range(len(I)-1)]
        I_new = []
        for j in range(len(o)):
            if o[j] <=overlap:
                I_new.append(I[j])
        I = I_new
    return pick

def calculate_IoU(i0,i1):
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
    iou = 1.0*(inter[1]-inter[0])/(union[1]-union[0])
    return iou

'''
compute recall at certain IoU
'''
def compute_IoU_recall_top_n_forreg(top_n, iou_thresh, sentence_image_mat, sentence_image_reg_mat, sclips, iclips):
    correct_num = 0.0
    for k in range(sentence_image_mat.shape[0]):
        gt = sclips[k]
        gt_movie, tim = gt.split('_')
        gt_start = float(tim.split("-")[0])
        gt_end = float(tim.split("-")[1])
        cad = iclips[k]
        cad_movie = cad.split('_')[0]
        #print gt +" "+str(gt_start)+" "+str(gt_end)
        if gt_movie == cad_movie:
            sim_v = [v for v in sentence_image_mat[k]]
            starts = [s for s in sentence_image_reg_mat[k,:,0]]
            ends = [e for e in sentence_image_reg_mat[k,:,1]]
            picks = nms_temporal(starts, ends, sim_v, iou_thresh-0.05)
            #sim_argsort=np.argsort(sim_v)[::-1][0:top_n]
            if top_n < len(picks) : picks = picks[0:top_n]
            for index in picks:
                pred_start = sentence_image_reg_mat[k, index, 0]
                pred_end = sentence_image_reg_mat[k, index, 1]
                iou = calculate_IoU((gt_start, gt_end),(pred_start, pred_end))
                if iou>=iou_thresh:
                    correct_num+=1
                    break
    return correct_num

'''
vcmr evaluation
'''
test_batch_size = 448
@torch.no_grad()
def inference(test_q_loader, test_p_loader, model, iter_step, test_result_output):
    model.eval()
    IoU_thresh = [0.1, 0.5, 0.7]
    all_correct_num_100 = [0.0] * 5
    all_correct_num_10 = [0.0] * 5
    all_correct_num_1 = [0.0] * 5
    all_retrievd = 0.0

    t_len = len(test_q_loader.dataset)
    t_feat_mat = torch.zeros([t_len, 1024])
    t_proto_sim_mat = torch.zeros([t_len, 100])
    gt_name = []

    for t_idx, t_batch in tqdm(enumerate(test_q_loader), total=len(test_q_loader), desc='query embedding computing'):
        t_input = t_batch[1].cuda()
        t_proto, t_feat = model.forward_vec_t(t_input)
        t_feat_mat[t_idx*test_batch_size:min((t_idx+1)*test_batch_size, t_len), :] = t_feat
        t_proto_sim_mat[t_idx*test_batch_size:min((t_idx+1)*test_batch_size, t_len), :] = t_proto
        gt_name.extend(list(t_batch[0]))
        # for t_ins in list(t_batch[0]):
        #     gt_name.append(t_ins)

    v_len = len(test_p_loader.dataset)
    # v_feat_mat = torch.zeros([v_len, 1024])
    # v_proto_sim_mat = torch.zeros([v_len, 100])
    cand_name = []
    for v_idx, v_batch in tqdm(enumerate(test_p_loader), total=len(test_p_loader), desc='visual embedding computing'):
        v_input = v_batch[1].cuda()
        v_proto, v_feat = model.forward_vec_v(v_input)
        np.save('/data2/wangyan/data/charades_test_v_tmp/{}.npy'.format(v_idx), v_feat.cpu().detach().numpy())
        np.save('/data2/wangyan/data/charades_test_v_proto_tmp/{}.npy'.format(v_idx), v_proto.cpu().detach().numpy())
        # v_feat_mat[v_idx * test_batch_size:min((v_idx+1)*test_batch_size, v_len), :] = v_feat
        # v_proto_sim_mat[v_idx * test_batch_size:min((v_idx + 1) * test_batch_size, v_len), :] = v_proto
        cand_name.extend(list(v_batch[0]))
        # for v_ins in list(v_batch[0]):
        #     cand_name.append(v_ins)


    # proto_sim_mat = torch.zeros([t_len, v_len])
    proto_sim_top_mat =torch.zeros([t_len, 100])
    start_time = time.time()
    for k in tqdm(range(t_len), total=t_len, desc='prototype similarity computing:'):
        # for t in range(v_len):
        #     coarse_metric = model.forward_coarse(v_proto[t], t_proto[k])
        #     proto_sim_mat[k, t] = coarse_metric
        proto_k = torch.zeros([v_len])
        for v_idx in range(len(test_p_loader)):
            v_proto = np.load('/data2/wangyan/data/charades_test_v_proto_tmp/{}.npy'.format(v_idx))
            coarse_metric = model.forward_coarse(torch.from_numpy(v_proto).cuda(), t_proto_sim_mat[k].cuda().unsqueeze(0).repeat([len(v_proto), 1]))
            proto_k[v_idx * test_batch_size: v_idx * test_batch_size+len(v_proto)] = coarse_metric
            # for s_idx in range(len(v_proto)):
            #     coarse_metric = model.forward_coarse(torch.from_numpy(v_proto[s_idx]).cuda().unsqueeze(0), t_proto[k].unsqueeze(0))
            #     proto_sim_mat[k, v_idx*test_batch_size+s_idx] = coarse_metric
        _, proto_sim_top_mat[k, :] = torch.topk(proto_k, k=100, dim=0, largest=True)
    sentence_image_mat = torch.zeros([t_len, 100])
    sentence_image_reg_mat = np.zeros([t_len, 100, 2])
    for k in tqdm(range(t_len), total=t_len, desc='fine-grained similarity computing'):
        for rank in range(100):
            # 取值的时候用proto_sim_top_mat[k, rank]
            # 存值的时候用rank
            cand_idx = int(proto_sim_top_mat[k, rank].item())
            start = float(cand_name[cand_idx].split('.npy')[0].split("_")[1].split('-')[0])
            end = float(cand_name[cand_idx].split("-")[1].split(".npy")[0])
            v_feat_mat = np.load('/data2/wangyan/data/charades_test_v_tmp/{}.npy'.format(cand_idx // test_batch_size))
            outputs = model.forward_fine(torch.from_numpy(v_feat_mat[cand_idx % test_batch_size]).unsqueeze(0).cuda(), t_feat_mat[k].cuda().unsqueeze(0))
            # outputs = model(torch.from_numpy(featmap).cuda(device), torch.from_numpy(sent_vec).cuda(device))
            outputs = torch.reshape(outputs, [3])
            sentence_image_mat[k, rank] = outputs[0]
            reg_end = end + outputs[2]
            reg_start = start + outputs[1]
            sentence_image_reg_mat[k, rank, 0] = reg_start
            sentence_image_reg_mat[k, rank, 1] = reg_end
    print('TIME: {}'.format(time.time()-start_time))
    iclips = [b for b in cand_name]
    sclips = [b for b in gt_name]

    # calculate Recall@m, IoU=n
    for k in range(len(IoU_thresh)):
        IoU = IoU_thresh[k]
        correct_num_100 = compute_IoU_recall_top_n_forreg(100, IoU, sentence_image_mat, sentence_image_reg_mat,
                                                         sclips, iclips)
        correct_num_10 = compute_IoU_recall_top_n_forreg(10, IoU, sentence_image_mat, sentence_image_reg_mat, sclips,
                                                        iclips)
        correct_num_1 = compute_IoU_recall_top_n_forreg(1, IoU, sentence_image_mat, sentence_image_reg_mat, sclips,
                                                        iclips)
        print(" IoU=" + str(IoU) + ", R@100: " + str(correct_num_100 / len(sclips)) + "; IoU=" + str(
            IoU) + ", R@10: " + str(correct_num_10 / len(sclips)) + "; IoU=" + str(IoU) + ", R@1: " + str(
            correct_num_1 / len(sclips)))

        all_correct_num_100[k] += correct_num_100
        all_correct_num_10[k] += correct_num_10
        all_correct_num_1[k] += correct_num_1
    all_retrievd += len(sclips)

    for k in range(len(IoU_thresh)):
        print(" IoU=" + str(IoU_thresh[k]) + ", R@100: " + str(all_correct_num_100[k] / all_retrievd) + "; IoU=" + str(
            IoU_thresh[k]) + ", R@10: " + str(all_correct_num_10[k] / all_retrievd) + "; IoU=" + str(
            IoU_thresh[k]) + ", R@1: " + str(all_correct_num_1[k] / all_retrievd))
        # test_result_output = open(test_result_path, "a")
        # with open(test_result_path, "a") as test_result_output:
        test_result_output.write("Step " + str(iter_step) + ": IoU=" + str(IoU_thresh[k]) + ", R@100: " + str(
            all_correct_num_100[k] / all_retrievd) + "; IoU=" + str(IoU_thresh[k]) + ", R@10: " + str(
            all_correct_num_10[k] / all_retrievd) + "; IoU=" + str(IoU_thresh[k]) + ", R@1: " + str(
            all_correct_num_1[k] / all_retrievd) + "\n")
        # test_result_output.close()


    del t_feat_mat, t_proto_sim_mat, gt_name, cand_name, \
        proto_sim_top_mat, sentence_image_reg_mat, sentence_image_mat, iclips, sclips
    gc.collect()

'''
evaluate the model
'''
def do_eval_slidingclips(test_set, model, iter_step, test_result_output):
    IoU_thresh = [0.1, 0.3, 0.5, 0.7, 0.9]
    all_correct_num_10 = [0.0] * 5
    all_correct_num_5 = [0.0] * 5
    all_correct_num_1 = [0.0] * 5
    all_retrievd = 0.0

    start_time = time.time()

    for movie_name in test_set.movie_names:
        # movie_length = movie_length_info[movie_name.split(".")[0]]
        # print("Test movie: " + movie_name + "....loading movie data")
        movie_clip_featmaps, movie_clip_sentences = test_set.load_movie_slidingclip(movie_name, 16)
        # print("sentences: " + str(len(movie_clip_sentences)))
        # print("clips: " + str(len(movie_clip_featmaps)))

        sentence_image_mat = np.zeros([len(movie_clip_sentences), len(movie_clip_featmaps)])
        sentence_image_reg_mat = np.zeros([len(movie_clip_sentences), len(movie_clip_featmaps), 2])
        for k in range(len(movie_clip_sentences)):
            # sentence_clip_name=movie_clip_sentences[k][0]
            # start=float(sentence_clip_name.split("_")[1])
            # end=float(sentence_clip_name.split("_")[2].split("_")[0])

            sent_vec = movie_clip_sentences[k][1]
            sent_vec = np.reshape(sent_vec, [1, sent_vec.shape[0]])
            for t in range(len(movie_clip_featmaps)):
                featmap = movie_clip_featmaps[t][1]
                visual_clip_name = movie_clip_featmaps[t][0]
                start = float(visual_clip_name.split(".npy")[0].split("_")[1].split("-")[0])
                end = float(visual_clip_name.split(".npy")[0].split("_")[1].split("-")[1])
                featmap = np.reshape(featmap, [1, featmap.shape[0]])
                outputs = model(torch.from_numpy(featmap).cuda(device), torch.from_numpy(sent_vec).cuda(device))
                outputs = torch.reshape(outputs, [3])
                sentence_image_mat[k, t] = outputs[0]
                # reg_clip_length = (end - start) * (10 ** outputs[2])
                # reg_mid_point = (start + end) / 2.0 + movie_length * outputs[1]
                reg_end = end + outputs[2]
                reg_start = start + outputs[1]

                sentence_image_reg_mat[k, t, 0] = reg_start
                sentence_image_reg_mat[k, t, 1] = reg_end


        iclips = [b[0].split(".npy")[0].split("_")[1] for b in movie_clip_featmaps]
        sclips = [b[0].split("_")[1] for b in movie_clip_sentences]

        # calculate Recall@m, IoU=n
        for k in range(len(IoU_thresh)):
            IoU = IoU_thresh[k]
            correct_num_10 = compute_IoU_recall_top_n_forreg(10, IoU, sentence_image_mat, sentence_image_reg_mat,
                                                             sclips, iclips)
            correct_num_5 = compute_IoU_recall_top_n_forreg(5, IoU, sentence_image_mat, sentence_image_reg_mat, sclips,
                                                            iclips)
            correct_num_1 = compute_IoU_recall_top_n_forreg(1, IoU, sentence_image_mat, sentence_image_reg_mat, sclips,
                                                            iclips)
            # print(movie_name + " IoU=" + str(IoU) + ", R@10: " + str(correct_num_10 / len(sclips)) + "; IoU=" + str(
            #     IoU) + ", R@5: " + str(correct_num_5 / len(sclips)) + "; IoU=" + str(IoU) + ", R@1: " + str(
            #     correct_num_1 / len(sclips)))

            all_correct_num_10[k] += correct_num_10
            all_correct_num_5[k] += correct_num_5
            all_correct_num_1[k] += correct_num_1
        all_retrievd += len(sclips)

    print('Time:{}'.format(time.time() - start_time))

    for k in range(len(IoU_thresh)):
        # print(" IoU=" + str(IoU_thresh[k]) + ", R@10: " + str(all_correct_num_10[k] / all_retrievd) + "; IoU=" + str(
        #     IoU_thresh[k]) + ", R@5: " + str(all_correct_num_5[k] / all_retrievd) + "; IoU=" + str(
        #     IoU_thresh[k]) + ", R@1: " + str(all_correct_num_1[k] / all_retrievd))
        test_result_output.write("Epoch " + str(iter_step) + ": IoU=" + str(IoU_thresh[k]) + ", R@10: " + str(
            all_correct_num_10[k] / all_retrievd) + "; IoU=" + str(IoU_thresh[k]) + ", R@5: " + str(
            all_correct_num_5[k] / all_retrievd) + "; IoU=" + str(IoU_thresh[k]) + ", R@1: " + str(
            all_correct_num_1[k] / all_retrievd) + "\n")


def run_training():
    initial_steps = 0
    max_steps = 20000
    max_epoch = 400
    batch_size = 448
    vs_lr = 0.005

    train_path = '/data2/wangyan/data/charades_training_samples_pt.npy'
    train_sliding_path = '/data2/wangyan/data/charades_c3dfeature_separate_clip/'
    test_sent_feat_path = '/data2/wangyan/data/charades_testing_samples.npy'
    test_feature_dir = '/data2/wangyan/data/charades_testing_candidate_feat.npy'
    train_set = TrainingDataset(batch_size, train_path, train_sliding_path)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=ctrl_collate_fn, drop_last=True)
    test_set_t = TestingDataset_sent(test_batch_size, test_sent_feat_path)
    test_set_v = TestingDataset_vis(test_batch_size, test_feature_dir)
    test_q_loader = DataLoader(test_set_t, batch_size=test_batch_size, shuffle=True)
    test_p_loader = DataLoader(test_set_v, batch_size=test_batch_size, shuffle=True)
    # test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=True, collate_fn=ctrl_collate_fn_test)

    model = CTRL_Model(batch_size)
    model.cuda()
    vs_optimizer = torch.optim.Adam(model.parameters(), vs_lr)

    test_result_output = open("vmr_proto_result_charades.txt", "a")
    # test_result_path = "vmr_proto_result_charades.txt"

    for epoch in xrange(max_epoch):
        start_time = time.time()

        for step, batch in tqdm(enumerate(train_loader), desc='Start Training: ', total=len(train_loader)):
            batch[0] = batch[0].cuda()
            batch[1] = batch[1].cuda()
            batch[2] = batch[2].cuda()
            batch[3] = batch[3].cuda()
            batch[4] = batch[4].cuda()

            # output = model(batch[0], batch[1])
            v_proto, v_feat = model.forward_vec_v(batch[0])
            t_proto, t_feat = model.forward_vec_t(batch[1])
            _ = model.forward_coarse(v_proto, t_proto)
            output = model.forward_fine(v_feat, t_feat)
            loss_value, offset_pred, loss_reg = model.compute_loss(output, batch[2], batch[3], batch[4])
            loss_value.backward()
            vs_optimizer.step()

            duration = time.time() - start_time

            if step % 700 == 0:
                # Print status to stdout.
                print('Epoch %d: loss = %.3f (%.3f sec)' % (epoch, loss_value, duration))

            if ((epoch + 1) % 100 == 0) and ((step + 1) % 1 == 0):
                print("Start to test:-----------------\n")
                # movie_length_info = pickle.load(open("./video_allframes_info.pkl", 'rb'), encoding='iso-8859-1')
                inference(test_q_loader, test_p_loader, model, epoch + 1, test_result_output)

def main():
    run_training()

if __name__=='__main__':
    main()