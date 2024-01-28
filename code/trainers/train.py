from tqdm import tqdm
import copy
import torch
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist
from torch.cuda.amp import autocast
from utils.ddp_util import all_gather
import utils.metric as metric
from torch_audiomentations import Compose, Gain, AddColoredNoise, PitchShift

class ModelTrainer:
    args = None
    vox1 = None
    vox2 = None
    ffsvc2022 = None
    model = None
    logger = None
    criterion = None
    optimizer = None
    lr_scheduler = None
    scaler = None
    train_set = None
    train_set_sampler = None
    train_loader = None
    enrollment_set = None
    enrollment_loader = None
    enrollment_ffsvc_set = None
    enrollment_ffsvc_loader = None
    spec = None
    epoch_start = 0

    def run(self):
        self.best_eer = 1000
        self.do_test_noise = [0]
        self.apply_augmentation = Compose(
            transforms=[
                Gain(
                    min_gain_in_db=-15.0,
                    max_gain_in_db=5.0,
                    p=0.1,
                ),
                PitchShift(p=0.2, sample_rate=16000),
                AddColoredNoise(p=0.3)
            ]
            )

        for epoch in range(self.epoch_start, self.args['epoch']):
            self.train_set_sampler.set_epoch(epoch)
            self.train(epoch)
            self.test(epoch)
    
            
    def train(self, epoch):
        self.model.train()
        idx_ct_start = len(self.train_loader)*(int(epoch))
        
        _loss = 0.
        _loss_clf = 0.
        if self.args['do_train_feature_enhancement']: _loss_fea_enh = 0.
        if self.args['do_train_code_enhancement']: _loss_code_enh = 0.

        with tqdm(total = len(self.train_loader), ncols = 150) as pbar:
            for idx, (m_batch_1, m_batch_2, m_batch_referance_1, m_batch_referance_2, m_label) in enumerate(self.train_loader):    #torch.Size([(clean,noise)2, bs, (channel)1, (bins)64, (frames) 256]), torch.Size([bs])
                loss = 0
                self.optimizer.zero_grad()
                
                m_label = m_label.tile(2).to(self.args['device'])
                
                m_batch_1 = m_batch_1.to(self.args['device'], non_blocking=True).float()
                m_batch_2 = m_batch_2.to(self.args['device'], non_blocking=True).float()

                m_batch_1 = self.apply_augmentation(m_batch_1.unsqueeze(1), sample_rate=16000).squeeze(1)
                m_batch_2 = self.apply_augmentation(m_batch_2.unsqueeze(1), sample_rate=16000).squeeze(1)

                m_batch_1 = self.spec(m_batch_1)
                m_batch_2 = self.spec(m_batch_2)
                m_batch = torch.cat((m_batch_1, m_batch_2))
                
                m_batch_referance_1 = self.spec(m_batch_referance_1.to(self.args['device'], non_blocking=True))
                m_batch_referance_2 = self.spec(m_batch_referance_2.to(self.args['device'], non_blocking=True))
                m_batch_referance = torch.cat((m_batch_referance_1, m_batch_referance_2)).unsqueeze(dim=1)
                
                with autocast():
                    code, output = self.model(m_batch)
                    description = '%s epoch: %d '%(self.args['name'], epoch)

                # code classification loss
                    loss_clf = self.criterion['classification_loss'](code, m_label)
                    loss += self.args['weight_classification_loss'] * loss_clf
                    _loss_clf += loss_clf.cpu().detach() 
                    description += 'loss_clf:%.3f '%(loss_clf)
                    
                    # feature enhancement loss
                    if self.args['do_train_feature_enhancement']:
                        loss_fea_enh = self.criterion['enhancement_loss'](output, m_batch_referance)
                        loss += self.args['weight_feature_enhancement_loss'] * loss_fea_enh
                        _loss_fea_enh += loss_fea_enh.cpu().detach() 
                        description += 'loss_fea_enh:%.3f '%(loss_fea_enh)

                    # code enhancement loss
                    if self.args['do_train_code_enhancement']:
                        code_1 = code[:len(code)//2]
                        code_2 = code[len(code)//2:]    
                        loss_code_enh = self.criterion['code_enhancement_loss'](code_1, code_2)
                        loss += self.args['weight_code_enhancement_loss'] * loss_code_enh
                        _loss_code_enh += loss_code_enh.cpu().detach() 
                        description += 'loss_code_enh:%.3f '%(loss_code_enh)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                _loss += loss.cpu().detach()
                
                description += 'TOT: %.4f'%(loss)
                pbar.set_description(description)
                pbar.update(1)

                # if the current epoch is match to the logging condition, log
                if idx % self.args['number_iteration_for_log'] == 0:
                    if idx != 0:
                        _loss /= self.args['number_iteration_for_log']
                        _loss_clf /= self.args['number_iteration_for_log']
                        if self.args['do_train_feature_enhancement']: _loss_fea_enh /= self.args['number_iteration_for_log']
                        if self.args['do_train_code_enhancement']: _loss_code_enh /= self.args['number_iteration_for_log']
                    
                        for p_group in self.optimizer.param_groups:
                            lr = p_group['lr']
                            break

                        if self.args['flag_parent']:
                            self.logger.log_metric('loss', _loss, step = idx_ct_start+idx)
                            self.logger.log_metric('loss_clf', _loss_clf, step = idx_ct_start+idx)
                            self.logger.log_metric('lr', lr, step = idx_ct_start+idx)

                            _loss = 0.
                            _loss_clf = 0.
                            if self.args['do_train_feature_enhancement']:
                                self.logger.log_metric('loss_fea_enh', _loss_fea_enh, step = idx_ct_start+idx)
                                _loss_fea_enh = 0.
                            if self.args['do_train_code_enhancement']:
                                self.logger.log_metric('loss_code_enh', _loss_code_enh, step = idx_ct_start+idx)
                                _loss_code_enh = 0.

                if self.args['learning_rate_scheduler'] == 'cosine' or 'warmup': 
                    self.lr_scheduler.step()        
        if self.args['learning_rate_scheduler'] == 'step': 
            self.lr_scheduler.step()        

    def test(self, epoch):
        # clean test data
        self.enrollment_set.Key = 'clean'
        self.enrollment_set.Length = -1
        self.embeddings = self._enrollment(self.enrollment_loader) 
        if self.args['flag_parent']:
            self.cur_eer, min_dcf1, min_dcf10 = self._calculate_eer(-1, self.vox1.test_trials)
            self.logger.log_metric('EER_clean', self.cur_eer, epoch_step=epoch)
            self.logger.log_metric('Min_DCF1_clean', min_dcf1, epoch_step=epoch)
            self.logger.log_metric('Min_DCF10_clean', min_dcf10, epoch_step=epoch)
            
            check_point = {
                'epoch': epoch,
                'model': self.model.state_dict(),
                'classification_loss': self.criterion['classification_loss'].state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                'scaler': self.scaler.state_dict()
            }
            if self.args['do_train_code_enhancement']: check_point['code_enhancement_loss'] = self.criterion['code_enhancement_loss'].state_dict()

            if self.cur_eer < self.best_eer:
                self.best_eer = self.cur_eer
                self.logger.log_metric('BestEER_clean', self.best_eer, epoch_step=epoch)
                self.logger.save_model('Best_checkpoint_{}'.format(epoch), check_point)
                self.do_test_noise = [1]
            self.logger.save_model('Latest_checkpoint', check_point)
            
        self._synchronize()
        self.do_test_noise = all_gather(self.do_test_noise)

        if sum(self.do_test_noise) and epoch >= 100:
            # short test data
            self.embeddings_org = copy.deepcopy(self.embeddings)
            self.test_challenge_set(epoch, 'clean', 1)
            self.test_challenge_set(epoch, 'clean', 2)
            self.test_challenge_set(epoch, 'clean', 5)

            # noise test data
            self.test_challenge_set(epoch, 'noise_0', -1)
            self.test_challenge_set(epoch, 'noise_5', -1)
            self.test_challenge_set(epoch, 'noise_10', -1)
            self.test_challenge_set(epoch, 'noise_15', -1)
            self.test_challenge_set(epoch, 'noise_20', -1)
            
            self.test_challenge_set(epoch, 'speech_0', -1)
            self.test_challenge_set(epoch, 'speech_5', -1)
            self.test_challenge_set(epoch, 'speech_10', -1)
            self.test_challenge_set(epoch, 'speech_15', -1)
            self.test_challenge_set(epoch, 'speech_20', -1)
            
            self.test_challenge_set(epoch, 'music_0', -1)
            self.test_challenge_set(epoch, 'music_5', -1)
            self.test_challenge_set(epoch, 'music_10', -1)
            self.test_challenge_set(epoch, 'music_15', -1)
            self.test_challenge_set(epoch, 'music_20', -1)
        
        self.do_test_noise = [0]

    def test_challenge_set(self, epoch, key, length):
        self.enrollment_set.Key = key
        self.enrollment_set.Length = length
        self.embeddings = self._enrollment(self.enrollment_loader)
        if self.args['flag_parent']:
            eer, min_dcf1, min_dcf10 = self._calculate_eer(length, self.vox1.test_trials)
            if length == -1: length ='full'
            self.logger.log_metric(f'EER_{key}_{length}', eer, epoch_step=epoch)
            self.logger.log_metric(f'Min_DCF1_{key}_{length}', min_dcf1, epoch_step=epoch)
            self.logger.log_metric(f'Min_DCF10_{key}_{length}', min_dcf10, epoch_step=epoch)
        self._synchronize()


    def _enrollment(self, loader):
        """Return embedding dictionary
        (self.enrollment_set is used for processing)
        """
        self.model.eval()

        keys = []
        embeddings = []
        
        with torch.set_grad_enabled(False):
            with self.model.no_sync():
                for utt, key in tqdm(loader, desc='enrollment', ncols=self.args['tqdm_ncols']):
                    utt = utt.to(self.args['device'], non_blocking=True).squeeze(0)
                    bs, seg, length = utt.size(0), utt.size(1), utt.size(2)
                    utt = utt.reshape(-1, length)

                    utt = self.spec(utt)
                    
                    embedding = self.model(utt, only_code = True).to('cpu')

                    embedding = embedding.reshape(bs, seg, -1)

                    keys.extend(key)
                    
                    embeddings.extend(embedding)
        
        self._synchronize()
        
        keys = all_gather(keys)
        embeddings = all_gather(embeddings)
        
        embedding_dict = {}
        for i in range(len(keys)):
            embedding_dict[keys[i]] = embeddings[i]
        
        return embedding_dict
    
    def _calculate_eer(self, length, trial):
        # test
        
        # Use enroll embeddings of full length utterances for short SV scenario
        enroll_embedding = self.embeddings_org if length > 0 else self.embeddings

        labels = []
        cos_sims = []

        for item in tqdm(trial, desc='test', ncols=self.args['tqdm_ncols']):
            cos_sims.append(self._calculate_cosine_similarity(enroll_embedding[item.key1], self.embeddings[item.key2]))
            labels.append(int(item.label))

        eer = metric.calculate_EER(
            scores=cos_sims, labels=labels
        )
        min_dcf1 = metric.calculate_MinDCF(
            scores=cos_sims, labels=labels, p_target=0.01, c_miss=1, c_false_alarm=1
        )
        min_dcf10 = metric.calculate_MinDCF(
            scores=cos_sims, labels=labels, p_target=0.01, c_miss=10, c_false_alarm=1
        )
        return eer, min_dcf1, min_dcf10

    def _synchronize(self):
        torch.cuda.empty_cache()
        dist.barrier()

    def _calculate_cosine_similarity(self, trials, enrollments):
    
        buffer1 = F.normalize(trials, p=2, dim=1)
        buffer2 = F.normalize(enrollments, p=2, dim=1)

        dist = F.pairwise_distance(buffer1.unsqueeze(-1), buffer2.unsqueeze(-1).transpose(0,2)).detach().cpu().numpy()

        score = -1 * np.mean(dist)

        return score