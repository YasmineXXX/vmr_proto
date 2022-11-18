import pickle
import os
from torch.utils.data import  Dataset
import torch
import numpy as np

'''
calculate temporal intersection over union
'''
# i0/i1: a tuple of (start, end)
def calculate_IoU(i0, i1):
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
    iou = 1.0*(inter[1]-inter[0])/(union[1]-union[0])
    return iou

'''
calculate the non Intersection part over Length ratia, make sure the input IoU is larger than 0
'''
def calculate_nIoL(base, sliding_clip):
    inter = (max(base[0], sliding_clip[0]), min(base[1], sliding_clip[1]))
    inter_l = inter[1]-inter[0]
    length = sliding_clip[1]-sliding_clip[0]
    nIoL = 1.0*(length-inter_l)/length
    return nIoL

class TrainingDataset(Dataset):
    def __init__(self, batch_size, train_path, sliding_path):
        self.counter = 0
        self.batch_size = batch_size
        self.context_num = 1
        self.context_size = 128
        self.sliding_clip_path = sliding_path
        self.clip_sentence_pairs_iou = np.load(train_path, allow_pickle=True, encoding='latin1')
        self.num_samples_iou = len(self.clip_sentence_pairs_iou)
        # print("{} iou clip-sentence pairs are readed".format(str(len(self.clip_sentence_pairs_iou))))

    def __getitem__(self, item):
        # clip_sentence_pairs_iou: {list: }, each for a tuple
        # clip_name: 's13-d21.avi_252_452'
        # sentence_vec: ndarray: 4800,
        # clip_name: 's22-d46.avi_6326_6454.npy'
        # start_offset:
        featmap = self.clip_sentence_pairs_iou[item]['v_feat']

        # read context features
        left_context_feat, right_context_feat = self.get_context_window(self.clip_sentence_pairs_iou[item]['vid'],
                                                                        self.clip_sentence_pairs_iou[item]['clip_name'],
                                                                        self.context_num)
        v_feat = np.hstack((left_context_feat, featmap, right_context_feat))
        t_feat = np.squeeze(self.clip_sentence_pairs_iou[item]['s_feat'], 0)
        p_offset = self.clip_sentence_pairs_iou[item]['s_off']
        l_offset = self.clip_sentence_pairs_iou[item]['e_off']
        offset = (p_offset, l_offset)
        vp_prob = self.clip_sentence_pairs_iou[item]['vp_prob']
        tp_prob = self.clip_sentence_pairs_iou[item]['tp_prob']

        return v_feat, t_feat, offset, vp_prob, tp_prob

    def __len__(self):
        return  len(self.clip_sentence_pairs_iou)

    '''
    compute left (pre) and right (post) context features
    '''
    def get_context_window(self, movie_name, clip_name, win_length):
        # clip_name: 's22-d46.avi_6326_6454.npy'
        # movie_name: 's22-d46.avi'
        # start: 6326
        # end: 6454
        # win_length: self.context_num = 1
        movie_name = movie_name
        start = int(clip_name.split("-")[0])
        end = int(clip_name.split("-")[1])
        # clip_length:  self.context_size = 128
        clip_length = self.context_size
        left_context_feats = np.zeros([win_length, 4096], dtype=np.float32)
        right_context_feats = np.zeros([win_length, 4096], dtype=np.float32)
        last_left_feat = np.load(self.sliding_clip_path+movie_name+'_'+clip_name+'.npy', allow_pickle=True)
        last_right_feat = np.load(self.sliding_clip_path+movie_name+'_'+clip_name+'.npy', allow_pickle=True)
        for k in range(win_length):
            left_context_start = start-clip_length*(k+1)
            left_context_end = start-clip_length*k
            right_context_start = end+clip_length*k
            right_context_end = end+clip_length*(k+1)
            left_context_name = movie_name+"_"+str(left_context_start)+"_"+str(left_context_end)+".npy"
            right_context_name = movie_name+"_"+str(right_context_start)+"_"+str(right_context_end)+".npy"
            if os.path.exists(self.sliding_clip_path+left_context_name):
                left_context_feat = np.load(self.sliding_clip_path+left_context_name)
                last_left_feat = left_context_feat
            else:
                left_context_feat = last_left_feat
            if os.path.exists(self.sliding_clip_path+right_context_name):
                right_context_feat = np.load(self.sliding_clip_path+right_context_name)
                last_right_feat = right_context_feat
            else:
                right_context_feat = last_right_feat
            left_context_feats[k] = left_context_feat
            right_context_feats[k] = right_context_feat
        return np.mean(left_context_feats, axis=0), np.mean(right_context_feats, axis=0)

class TestingDataset_sent(Dataset):
    def __init__(self, batch_size, sentvec_path):
        self.batch_size = batch_size
        self.semantic_size = 4800
        sent_feat = np.load(sentvec_path, allow_pickle=True, encoding='latin1')
        self.clip_sentence_pairs = []
        for item in sent_feat:
            clip_name = item['clip_name']
            s_feat = np.squeeze(item['s_feat'], 0)
            self.clip_sentence_pairs.append((clip_name, s_feat))
    def __len__(self):
        return len(self.clip_sentence_pairs)
    def __getitem__(self, item):
        gt = self.clip_sentence_pairs[item][0]
        t_feat = self.clip_sentence_pairs[item][1]
        return gt, t_feat

class TestingDataset_vis(Dataset):
    def __init__(self, batch_size, sliding_dir):
        self.batch_size = batch_size
        self.sliding_clip_path = sliding_dir
        vis_feat = np.load(sliding_dir, allow_pickle=True, encoding='latin1')
        self.sliding_clip = []
        for item in vis_feat:
            clip_name = item[0]
            v_feat = np.squeeze(item[1], 0)
            self.sliding_clip.append((clip_name, v_feat))
    def __len__(self):
        return len(self.sliding_clip)
    def __getitem__(self, item):
        clip = self.sliding_clip[item][0]
        v_feat = self.sliding_clip[item][1]
        return clip, v_feat

class TestingDataset(Dataset):
    def __init__(self, batch_size, sentvec_path, sliding_dir):
        # il_path: image_label_file path
        # self.index_in_epoch = 0
        # self.epochs_completed = 0
        self.batch_size = batch_size
        self.image_dir = sliding_dir
        # print('Reading testing data list from {}'.format(sentvec_path))
        self.semantic_size = 4800
        sent_feat = np.load(sentvec_path, allow_pickle=True, encoding='latin1')
        self.clip_sentence_pairs = []
        for item in sent_feat:
            clip_name = item['clip_name']
            s_feat = np.squeeze(item['s_feat'], 0)
            self.clip_sentence_pairs.append((clip_name, s_feat))
        # print("{} pairs are readed".format(str(len(self.clip_sentence_pairs))))
        movie_names_set = set()
        self.movie_clip_names = {}
        for k in range(len(self.clip_sentence_pairs)):
            clip_name = self.clip_sentence_pairs[k][0]
            movie_name = clip_name.split("_")[0]
            if not movie_name in movie_names_set:
                movie_names_set.add(movie_name)
                self.movie_clip_names[movie_name] = []
            self.movie_clip_names[movie_name].append(k)
        # movie_clip_names: {dict: 25}
        # 's27-d50.avi': list of clips(each itom for a )
        self.movie_names = list(movie_names_set)
        #
        # self.clip_num_per_movie_max = 0
        # # movie_name: 's27-d50.avi
        # for movie_name in self.movie_clip_names:
        #     if len(self.movie_clip_names[movie_name]) > self.clip_num_per_movie_max:
        #         self.clip_num_per_movie_max = len(self.movie_clip_names[movie_name])
        # print("Max number of clips in a movie is {}".format(str(self.clip_num_per_movie_max)))

        self.sliding_clip_path = sliding_dir
        sliding_clips_tmp = os.listdir(self.sliding_clip_path)
        self.sliding_clip_names = []
        for clip_name in sliding_clips_tmp:
            if clip_name.split(".")[1] == "npy":
                movie_name = clip_name.split("_")[0]
                if movie_name in self.movie_clip_names:
                    self.sliding_clip_names.append(clip_name)
        # sliding_clip_names: {list: 15881}
        # 's31-d28.avi_3010_3266'
        self.num_samples = len(self.clip_sentence_pairs)
        # print("sliding clips number: {}".format(str(len(self.sliding_clip_names))))
        assert self.batch_size <= self.num_samples

    # def __getitem__(self, item):
    #     feat_path = self.sliding_clip_path + self.sliding_clip_names[item]
    #     featmap = np.load(feat_path)
    #     # read context features
    #     left_context_feat, right_context_feat = self.get_context_window(self.sliding_clip_names[item],
    #                                                                     self.context_num)
    #     v_feat = np.hstack((left_context_feat, featmap, right_context_feat))
    #     t_feat = self.clip_sentence_pairs[item][1]
    #
    #     return v_feat, t_feat

    def __len__(self):
        return self.num_clip_sentence_pairs_iou

    def get_context_window(self, clip_name, win_length):
        movie_name = clip_name.split("_")[0]
        start = int(clip_name.split("_")[1].split('.npy')[0].split('-')[0])
        end = int(clip_name.split("_")[1].split('.npy')[0].split('-')[1])
        clip_length = 128#end-start
        left_context_feats = np.zeros([win_length,4096], dtype=np.float32)
        right_context_feats = np.zeros([win_length,4096], dtype=np.float32)
        last_left_feat = np.load(self.sliding_clip_path+clip_name)
        last_right_feat = np.load(self.sliding_clip_path+clip_name)
        for k in range(win_length):
            left_context_start = start-clip_length*(k+1)
            left_context_end = start-clip_length*k
            right_context_start = end+clip_length*k
            right_context_end = end+clip_length*(k+1)
            left_context_name = movie_name+"_"+str(left_context_start)+"-"+str(left_context_end)+".npy"
            right_context_name = movie_name+"_"+str(right_context_start)+"-"+str(right_context_end)+".npy"
            if os.path.exists(self.sliding_clip_path+left_context_name):
                left_context_feat = np.load(self.sliding_clip_path+left_context_name)
                last_left_feat = left_context_feat
            else:
                left_context_feat = last_left_feat
            if os.path.exists(self.sliding_clip_path+right_context_name):
                right_context_feat = np.load(self.sliding_clip_path+right_context_name)
                last_right_feat = right_context_feat
            else:
                right_context_feat = last_right_feat
            left_context_feats[k] = left_context_feat
            right_context_feats[k] = right_context_feat

        return np.mean(left_context_feats, axis=0), np.mean(right_context_feats, axis=0)

    def load_movie_slidingclip(self, movie_name, sample_num):
        movie_clip_sentences = []
        movie_clip_featmap = []
        clip_set = set()
        for k in range(len(self.clip_sentence_pairs)):
            # clip_sentence_pairs: {list}, each for a tuple
            # clip_name:
            # sent_vec:
            if movie_name in self.clip_sentence_pairs[k][0]:
                # movie_clip_sentences: {list}, each for a tuple
                # clip_name:
                # sent_vec:
                movie_clip_sentences.append((self.clip_sentence_pairs[k][0], self.clip_sentence_pairs[k][1][:self.semantic_size]))
        # sliding_clip_names: {list}
        # 's31-d28.avi_3010_3266'
        for k in range(len(self.sliding_clip_names)):
            if movie_name in self.sliding_clip_names[k]:
                # print str(k)+"/"+str(len(self.movie_clip_names[movie_name]))
                visual_feature_path = self.sliding_clip_path+self.sliding_clip_names[k]
                #context_feat=self.get_context(self.sliding_clip_names[k]+".npy")
                left_context_feat, right_context_feat = self.get_context_window(self.sliding_clip_names[k], 1)
                feature_data = np.load(visual_feature_path)
                #comb_feat=np.hstack((context_feat,feature_data))
                comb_feat = np.hstack((left_context_feat, feature_data, right_context_feat))
                movie_clip_featmap.append((self.sliding_clip_names[k], comb_feat))
        # movie_clip_featmap: {list}, each for a tuple
        # sliding_clip_names[k]: 's31-d28.avi_3010_3266'
        # comb_feat: vector of 4096*3

        # testing_samples_path = '/data2/wangyan/data/charades/testing_samples/'
        # save_path = os.path.join(testing_samples_path, movie_name)
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        #     gt_path = os.path.join(save_path, 'gt.npy')
        #     cand_path = os.path.join(save_path, 'candidate.npy')
        #     np.save(gt_path, movie_clip_sentences)
        #     np.save(cand_path, movie_clip_featmap)

        # testing_samples_path = '/data2/wangyan/data/TACoS/testing_samples/'
        # load_path = os.path.join(testing_samples_path, movie_name)
        # movie_clip_sentences = np.load(os.path.join(load_path, 'gt.npy'), allow_pickle=True)
        # movie_clip_featmap = np.load(os.path.join(load_path, 'candidate.npy'), allow_pickle=True)

        return movie_clip_featmap, movie_clip_sentences