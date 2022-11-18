import random

import torch.nn
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class CTRL_Model(nn.Module):
    def __init__(self, batch_size):
        super(CTRL_Model, self).__init__()
        self.batch_size = batch_size
        self.lambda_regression = 0.01
        self.alpha = 1.0/batch_size
        self.semantic_size = 1024 # the size of visual and semantic comparison size
        self.sentence_embedding_size = 4800
        self.visual_feature_dim = 4096*3

        self.v_fc = torch.nn.Linear(self.visual_feature_dim, self.semantic_size)
        self.s_fc = torch.nn.Linear(self.sentence_embedding_size, self.semantic_size)

        self.vs_conv = torch.nn.Conv2d(self.semantic_size*4, 1000, 1, stride=1)
        self.relu = torch.nn.ReLU()
        self.vs_conv2 = torch.nn.Conv2d(1000, 3, 1, stride=1)

        self.v_protonet = Parameter(torch.randn(100, 1024))
        self.t_protonet = Parameter(torch.randn(100, 1024))

    def forward(self, visual_feature_train, sentence_embed_train):
        batch_size = visual_feature_train.shape[0]
        # print("Building training network...............................\n")
        transformed_clip_train = self.v_fc(visual_feature_train)
        transformed_clip_train_norm = F.normalize(transformed_clip_train, p=2, dim=1)

        transformed_sentence_train = self.s_fc(sentence_embed_train)
        transformed_sentence_train_norm = F.normalize(transformed_sentence_train, p=2, dim=1)

        self.v_proto_pred = F.softmax(F.cosine_similarity(transformed_clip_train_norm.unsqueeze(1).repeat([1, 100, 1]), self.v_protonet.repeat([self.batch_size, 1, 1]), dim=2))
        self.t_proto_pred = F.softmax(F.cosine_similarity(transformed_sentence_train_norm.unsqueeze(1).repeat([1, 100, 1]), self.t_protonet.repeat([self.batch_size, 1, 1]), dim=2))

        vv_feature = torch.reshape(transformed_clip_train_norm.repeat([batch_size, 1]),
            [batch_size, batch_size, self.semantic_size])
        ss_feature = torch.reshape(transformed_sentence_train_norm.repeat([1, batch_size]), [batch_size, batch_size, self.semantic_size])
        concat_feature = torch.reshape(torch.cat([vv_feature, ss_feature], 2),
                                    [batch_size, batch_size, self.semantic_size + self.semantic_size])
        # print(concat_feature.shape)

        mul_feature = torch.mul(vv_feature, ss_feature)
        add_feature = torch.add(vv_feature, ss_feature)

        cross_modal_vec = torch.reshape(torch.cat([mul_feature, add_feature, concat_feature], 2),
                                  [1, batch_size, batch_size, self.semantic_size * 4]).permute(0, 3, 1, 2)
        sim_score_mat = self.vs_conv2(self.relu(self.vs_conv(cross_modal_vec)))

        sim_score_mat = torch.reshape(sim_score_mat, [3, batch_size, batch_size]).permute(1, 2, 0)


        return sim_score_mat

    def forward_vec_v(self, visual_feature_train):
        batch_size = visual_feature_train.shape[0]
        # print("Building training network...............................\n")
        transformed_clip_train = self.v_fc(visual_feature_train)
        transformed_clip_train_norm = F.normalize(transformed_clip_train, p=2, dim=1)

        self.v_proto_pred = F.softmax(F.cosine_similarity(transformed_clip_train_norm.unsqueeze(1).repeat([1, 100, 1]),
                                                     self.v_protonet.repeat([batch_size, 1, 1]), dim=2), dim=-1)
        return self.v_proto_pred, transformed_clip_train_norm

    def forward_vec_t(self, sentence_embed_train):
        batch_size = sentence_embed_train.shape[0]
        # print("Building training network...............................\n")

        transformed_sentence_train = self.s_fc(sentence_embed_train)
        transformed_sentence_train_norm = F.normalize(transformed_sentence_train, p=2, dim=1)

        self.t_proto_pred = F.softmax(F.cosine_similarity(transformed_sentence_train_norm.unsqueeze(1).repeat([1, 100, 1]), self.t_protonet.repeat([batch_size, 1, 1]), dim=2), dim=-1)
        return self.t_proto_pred, transformed_sentence_train_norm

    def forward_coarse(self, v_proto_pred, t_proto_pred):
        coarse_metric = F.cosine_similarity(v_proto_pred, t_proto_pred)
        return coarse_metric

    def forward_fine(self, transformed_clip_train_norm, transformed_sentence_train_norm):
        batch_size = transformed_clip_train_norm.shape[0]
        vv_feature = torch.reshape(transformed_clip_train_norm.repeat([batch_size, 1]),
            [batch_size, batch_size, self.semantic_size])
        ss_feature = torch.reshape(transformed_sentence_train_norm.repeat([1, batch_size]), [batch_size, batch_size, self.semantic_size])
        concat_feature = torch.reshape(torch.cat([vv_feature, ss_feature], 2),
                                    [batch_size, batch_size, self.semantic_size + self.semantic_size])
        # print(concat_feature.shape)

        mul_feature = torch.mul(vv_feature, ss_feature)
        add_feature = torch.add(vv_feature, ss_feature)

        cross_modal_vec = torch.reshape(torch.cat([mul_feature, add_feature, concat_feature], 2),
                                  [1, batch_size, batch_size, self.semantic_size * 4]).permute(0, 3, 1, 2)
        sim_score_mat = self.vs_conv2(self.relu(self.vs_conv(cross_modal_vec)))

        sim_score_mat = torch.reshape(sim_score_mat, [3, batch_size, batch_size]).permute(1, 2, 0)

        return sim_score_mat

    def compute_loss(self, sim_score_mat, offset, v_proto_gt, t_proto_gt):

        self.loss_align_reg, offset_pred, loss_reg = self.compute_loss_reg(sim_score_mat, offset)
        loss_vp = F.kl_div(torch.log(self.v_proto_pred), v_proto_gt)
        loss_tp = F.kl_div(torch.log(self.t_proto_pred), t_proto_gt)
        self.loss_align_reg += loss_tp + loss_vp

        return self.loss_align_reg, offset_pred, loss_reg


    def compute_loss_reg(self, sim_reg_mat, offset_label):
        device = sim_reg_mat.device

        sim_score_mat, p_reg_mat, l_reg_mat = torch.split(sim_reg_mat, 1, dim=2)
        sim_score_mat = torch.reshape(sim_score_mat, [self.batch_size, self.batch_size])
        l_reg_mat = torch.reshape(l_reg_mat, [self.batch_size, self.batch_size])
        p_reg_mat = torch.reshape(p_reg_mat, [self.batch_size, self.batch_size])

        # unit matrix with -2
        I_2 = torch.diag(torch.ones(self.batch_size) * -2.0)
        all1 = torch.ones(self.batch_size, self.batch_size)
        #               | -1  1   1...   |

        #   mask_mat =  | 1  -1   1...   |

        #               | 1   1  -1 ...  |
        mask_mat = torch.add(I_2, all1)
        # loss cls, not considering iou
        I = torch.diag(torch.ones(self.batch_size))
        I_half = torch.diag(torch.ones(self.batch_size) * 0.5)
        # self.alpha = 1/batch_size = 1/N
        batch_para_mat = torch.ones(self.batch_size, self.batch_size) * self.alpha
        para_mat = torch.add(I, batch_para_mat)

        mask_mat = mask_mat.cuda(device)
        all1 = all1.cuda(device)
        I = I.cuda(device)
        para_mat = para_mat.cuda(device)

        # mask_mat, used for distinguishing positive and negative pairs
        loss_mat = torch.log(torch.add(all1, torch.exp(torch.mul(mask_mat, sim_score_mat))))
        loss_mat = torch.mul(loss_mat, para_mat)
        loss_align = torch.mean(loss_mat)
        # regression loss
        l_reg_diag = torch.matmul(torch.mul(l_reg_mat, I), torch.ones(self.batch_size, 1).cuda(device))
        p_reg_diag = torch.matmul(torch.mul(p_reg_mat, I), torch.ones(self.batch_size, 1).cuda(device))
        offset_pred = torch.cat((p_reg_diag, l_reg_diag), dim=1)
        loss_reg = torch.mean(torch.abs(torch.sub(offset_pred, offset_label)))

        loss = torch.add(torch.mul(self.lambda_regression, loss_reg), loss_align)

        return loss, offset_pred, loss_reg