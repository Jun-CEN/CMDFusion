import torch
import torch_scatter
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from network.basic_block import Lovasz_loss
from network.spvcnn_g import get_model as SPVCNN
from network.base_model_distil import LightningBaseModel
from network.basic_block import UpsampleNet_2, ResNetFCN_2
import time
from thop import profile

class CMDFuse(nn.Module):
    def __init__(self,config):
        super(CMDFuse, self).__init__()
        self.hiden_size = config['model_params']['hiden_size']
        self.scale_list = config['model_params']['scale_list']
        self.num_classes = config['model_params']['num_classes']
        self.lambda_xm = config['train_params']['lambda_xm']
        self.lambda_seg2d = config['train_params']['lambda_seg2d']
        self.num_scales = len(self.scale_list)

        self.multihead_3d_classifier = nn.ModuleList()
        for i in range(self.num_scales):
            self.multihead_3d_classifier.append(
                nn.Sequential(
                    nn.Linear(self.hiden_size, 128),
                    nn.ReLU(True),
                    nn.Linear(128, self.num_classes))
            )

        self.multihead_fuse_classifier = nn.ModuleList()
        self.multihead_fuse_classifier_2 = nn.ModuleList()
        for i in range(self.num_scales):
            self.multihead_fuse_classifier.append(
                nn.Sequential(
                    nn.Linear(self.hiden_size, 128),
                    nn.ReLU(True),
                    nn.Linear(128, self.num_classes))
            )
            self.multihead_fuse_classifier_2.append(
                nn.Sequential(
                    nn.Linear(self.hiden_size, 128),
                    nn.ReLU(True),
                    nn.Linear(128, self.num_classes))
            )
        self.leaners = nn.ModuleList()
        self.leaners_2 = nn.ModuleList()
        self.fcs1 = nn.ModuleList()
        self.fcs2 = nn.ModuleList()
        self.fcs1_2 = nn.ModuleList()
        self.fcs2_2 = nn.ModuleList()
        for i in range(self.num_scales):
            self.leaners.append(nn.Sequential(nn.Linear(self.hiden_size, self.hiden_size)))
            self.leaners_2.append(nn.Sequential(nn.Linear(self.hiden_size, self.hiden_size)))
            self.fcs1.append(nn.Sequential(nn.Linear(self.hiden_size * 2, self.hiden_size)))
            self.fcs2.append(nn.Sequential(nn.Linear(self.hiden_size * 2, self.hiden_size)))
            self.fcs1_2.append(nn.Sequential(nn.Linear(self.hiden_size * 2, self.hiden_size)))
            self.fcs2_2.append(nn.Sequential(nn.Linear(self.hiden_size * 2, self.hiden_size)))

        self.classifier = nn.Sequential(
            nn.Linear(self.hiden_size * self.num_scales, 128),
            nn.ReLU(True),
            nn.Linear(128, self.num_classes),
        )

        self.classifier_2 = nn.Sequential(
            nn.Linear(self.hiden_size * self.num_scales, 128),
            nn.ReLU(True),
            nn.Linear(128, self.num_classes),
        )

        if 'seg_labelweights' in config['dataset_params']:
            seg_num_per_class = config['dataset_params']['seg_labelweights']
            seg_labelweights = seg_num_per_class / np.sum(seg_num_per_class)
            seg_labelweights = torch.Tensor(np.power(np.amax(seg_labelweights) / seg_labelweights, 1 / 3.0))
        else:
            seg_labelweights = None

        self.ce_loss = nn.CrossEntropyLoss(weight=seg_labelweights, ignore_index=config['dataset_params']['ignore_label'])
        self.lovasz_loss = Lovasz_loss(ignore=config['dataset_params']['ignore_label'])

    @staticmethod
    def p2img_mapping(pts_fea, p2img_idx, batch_idx):
        img_feat = []
        for b in range(batch_idx.max()+1):
            img_feat.append(pts_fea[batch_idx == b][p2img_idx[b]])
        return torch.cat(img_feat, 0)

    @staticmethod
    def voxelize_labels(labels, full_coors):
        lbxyz = torch.cat([labels.reshape(-1, 1), full_coors], dim=-1)
        unq_lbxyz, count = torch.unique(lbxyz, return_counts=True, dim=0)
        inv_ind = torch.unique(unq_lbxyz[:, 1:], return_inverse=True, dim=0)[1]
        label_ind = torch_scatter.scatter_max(count, inv_ind)[1]
        labels = unq_lbxyz[:, 0][label_ind]
        return labels

    def seg_loss(self, logits, labels):
        ce_loss = self.ce_loss(logits, labels)
        lovasz_loss = self.lovasz_loss(F.softmax(logits, dim=1), labels)
        return ce_loss + lovasz_loss

    def forward(self, data_dict):
        loss = 0
        img_seg_feat = []
        pts_seg_feat = []
        batch_idx = data_dict['batch_idx']
        point2img_index = data_dict['point2img_index']
        img_feat_l = []
        d2_feat_l_p = []
        d2_feat_l = []
        d3_feat_l = []
        fuse_feat_l = []
        fuse_feat_l_i = []
        for idx, scale in enumerate(self.scale_list):
            img_feat = data_dict['img_scale{}'.format(scale)]
            pts_feat = data_dict['layer_{}_L'.format(idx)]['pts_feat']
            pts_feat_f = data_dict['layer_{}_L'.format(idx)]['pts_feat_f']
            coors_inv = data_dict['scale_{}'.format(scale)]['coors_inv']
            g_img_feat = data_dict['layer_{}_C'.format(idx)]['pts_feat_f']
            g_img_feat_p = self.p2img_mapping(g_img_feat, point2img_index, batch_idx)

            # 3D prediction
            pts_pred_full = self.multihead_3d_classifier[idx](pts_feat)

            # correspondence
            pts_label_full = self.voxelize_labels(data_dict['labels'], data_dict['layer_{}_L'.format(idx)]['full_coors'])
            pts_feat = self.p2img_mapping(pts_feat[coors_inv], point2img_index, batch_idx)
            pts_pred = self.p2img_mapping(pts_pred_full[coors_inv], point2img_index, batch_idx)

            # # 3D prediction
            # pts_pred_full = self.multihead_3d_classifier[idx](pts_feat_f)

            # # correspondence
            # pts_label_full = data_dict['labels']
            pts_feat_f_p = self.p2img_mapping(pts_feat_f, point2img_index, batch_idx)
            # pts_pred_f = self.p2img_mapping(pts_pred_full, point2img_index, batch_idx)

            # modality fusion
            feat_learner = self.leaners[idx](pts_feat_f)
            feat_learner = F.relu(feat_learner)
            feat_cat = torch.cat([g_img_feat, feat_learner], 1)
            feat_cat = self.fcs1[idx](feat_cat)

            feat_weight = torch.sigmoid(self.fcs2[idx](torch.cat((feat_cat, feat_cat.mean(0).unsqueeze(0).expand(feat_cat.shape)), -1)))
            feat_cat = F.relu(feat_cat * feat_weight)

            # fusion prediction
            fuse_pred = feat_cat + g_img_feat
            img_seg_feat.append(fuse_pred)
            fuse_pred = self.multihead_fuse_classifier[idx](fuse_pred)

            data_dict['fuse_g_img_scale{}'.format(scale)] = fuse_pred

            # modality fusion 2
            feat_learner_2 = self.leaners_2[idx](g_img_feat)
            feat_learner_2 = F.relu(feat_learner_2)
            feat_cat_2 = torch.cat([pts_feat_f, feat_learner_2], 1)
            feat_cat_2 = self.fcs1_2[idx](feat_cat_2)

            feat_weight_2 = torch.sigmoid(self.fcs2_2[idx](torch.cat((feat_cat_2, feat_cat_2.mean(0).unsqueeze(0).expand(feat_cat.shape)), -1)))
            feat_cat_2 = F.relu(feat_cat_2 * feat_weight_2)

            # fusion prediction 2
            fuse_pred_2 = feat_cat_2 + pts_feat_f
            pts_seg_feat.append(fuse_pred_2)
            fuse_pred_2 = self.multihead_fuse_classifier_2[idx](fuse_pred_2)

            data_dict['fuse_pts_scale{}'.format(scale)] = fuse_pred_2

            # Segmentation Loss
            seg_loss_3d = self.seg_loss(pts_pred_full, pts_label_full)
            # seg_loss_2d = self.seg_loss(fuse_pred, data_dict['img_label'])
            mse_loss = nn.MSELoss()
            g_loss = mse_loss(g_img_feat_p, img_feat)
            data_dict['g_loss'] = g_loss
            # loss += seg_loss_3d + g_loss * self.lambda_seg2d / self.num_scales
            loss += g_loss * self.lambda_seg2d / self.num_scales
            # loss = seg_loss_3d
        
        img_seg_logits = self.classifier(torch.cat(img_seg_feat, 1))
        data_dict['fuse_img_scale_all'] = img_seg_logits
        # loss += self.seg_loss(img_seg_logits, data_dict['labels'])
        pts_seg_logits = self.classifier_2(torch.cat(pts_seg_feat, 1))
        data_dict['fuse_pts_scale_all'] = pts_seg_logits
        loss += self.seg_loss(pts_seg_logits, data_dict['labels'])

        data_dict['loss'] += loss

        return data_dict



class get_model(LightningBaseModel):
    def __init__(self, config):
        super(get_model, self).__init__(config)
        self.save_hyperparameters()
        self.baseline_only = config.baseline_only
        self.num_classes = config.model_params.num_classes
        self.hiden_size = config.model_params.hiden_size
        self.lambda_seg2d = config.train_params.lambda_seg2d
        self.lambda_xm = config.train_params.lambda_xm
        self.scale_list = config.model_params.scale_list
        self.num_scales = len(self.scale_list)
        self.ft = config.train_params.finetune
        self.model_3d = SPVCNN(config)
        if not self.baseline_only:
            self.model_2d = ResNetFCN_2(
                backbone=config.model_params.backbone_2d,
                pretrained=config.model_params.pretrained2d,
                config=config
            )
            model_path = config.clip.path
            state_dict = torch.load(model_path)
            for k, v in list(state_dict['model'].items()):
                if 'backbone' in k:
                    new_key = k.replace('backbone.', '')
                    state_dict['model'][new_key] = state_dict['model'].pop(k)
                print(k, new_key)
            self.model_2d.load_state_dict(state_dict['model'], strict=False)
            self.upsamplenet = UpsampleNet_2(config.dataset_params.pc_dataset_type)
            self.fusion = CMDFuse(config)
        else:
            print('Start vanilla training!')
        

    def forward(self, data_dict):
        # 3D network
        self.model_2d.eval()
        for p in self.model_2d.parameters(): p.detach_()
        if self.ft:
            self.model_3d.spv_enc_C.eval()
            for p in self.model_3d.spv_enc_C.parameters(): p.detach_()
            self.model_3d.voxel_3d_generator.eval()
            for p in self.model_3d.voxel_3d_generator.parameters(): p.detach_()
        # 3D network
        data_dict = self.model_3d(data_dict)
        data_dict['img'] = transforms.Resize([256, 512])(data_dict['img'])
        output_image = self.model_2d(data_dict)
        data_dict = self.upsamplenet(output_image, data_dict)

        process_keys = [k for k in data_dict.keys() if k.find('img_scale') != -1]
        img_indices = data_dict['img_indices']
        temp = {k: [] for k in process_keys}
        for i in range(data_dict['img'].shape[0]):
            for k in process_keys:
                temp[k].append(data_dict[k].permute(0, 2, 3, 1)[i][img_indices[i][:, 0], img_indices[i][:, 1]])
        for k in process_keys:
            data_dict[k] = torch.cat(temp[k], 0)

        data_dict = self.fusion(data_dict)

        return data_dict