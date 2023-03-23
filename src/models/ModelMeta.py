import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import pearsonr


class Learner(nn.Module):
    """
    This is a learner class, which will accept a specific network module, such as OmniNet that define the network forward
    process. Learner class will create two same network, one as theta network and the other acts as theta_pi network.
    for each episode, the theta_pi network will copy its initial parameters from theta network and update several steps
    by meta-train set and then calculate its loss on meta-test set. All loss on meta-test set will be sumed together and
    then backprop on theta network, which should be done on metalaerner class.
    For learner class, it will be responsible for update for several steps on meta-train set and return with the loss on
    meta-test set.
    """

    def __init__(self, net_cls, args):
        """
        It will receive a class: net_cls and its parameters: args for net_cls.
        :param net_cls: class, not instance
        :param args: the parameters for net_cls
        """
        super(Learner, self).__init__()
        # pls make sure net_cls is a class but NOT an instance of class.
        assert net_cls.__class__ == type

        # we will create two class instance meanwhile and use one as theta network and the other as theta_pi network.
        self.net = net_cls(args)
        # you must call create_pi_net to create pi network additionally
        self.net_pi = net_cls(args)
        # update theta_pi = theta_pi - lr * grad-
        # according to the paper, here we use naive version of SGD to update theta_pi
        # 0.1 here means the learner_lr
        self.optimizer = optim.SGD(self.net_pi.parameters(), 0.001)

    def parameters(self):
        """
        Override this function to return only net parameters for MetaLearner's optimize
        it will ignore theta_pi network parameters.
        :return:
        """
        return self.net.parameters()

    def update_pi(self):
        """sh 
        copy parameters from self.net -> self.net_pi
        :return:
        """
        self.net_pi.load_state_dict(self.net.state_dict())

#         for m_from, m_to in zip(self.net.modules(), self.net_pi.modules()):
#             # print(m_to)
#             # for p in m_to.named_parameters():
#             #     print(p[0])
#             if isinstance(m_to, nn.Linear) or \
#                 isinstance(m_to, nn.BatchNorm1d):
#                 m_to.weight.data = m_from.weight.data.clone()
#                 if m_to.bias is not None:
#                     m_to.bias.data = m_from.bias.data.clone()


    def forward(self, support_x, support_y, query_x, query_y, num_updates, training=True):
        """
        learn on current episode meta-train: support_x & support_y and then calculate loss on meta-test set: query_x&y
        :param support_x: [setsz, c_, h, w]
        :param support_y: [setsz]
        :param query_x:   [querysz, c_, h, w]
        :param query_y:   [querysz]
        :param num_updates: 5
        :return:
        """
        # now try to fine-tune from current $theta$ parameters -> $theta_pi$
        # after num_updates of fine-tune, we will get a good theta_pi parameters so that it will retain satisfying
        # performance on specific task, that's, current episode.
        # firstly, copy theta_pi from theta network
        self.update_pi()
        # losses_q = []
        # with torch.no_grad():
        #     self.net_pi.eval()
        #     loss_q, _ = self.net_pi(query_x, query_y)
        #     losses_q.append(loss_q)
        # # update for several steps
        for i in range(num_updates):
            self.net_pi.train()
            # forward and backward to update net_pi grad.
            loss, pred = self.net_pi(support_x, support_y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        #     with torch.no_grad():
        #         self.net_pi.eval()
        #         loss_q, _ = self.net_pi(query_x, query_y)
        #         losses_q.append(loss_q)
        
        # print("losses_q", query_y['assay_id'][0], torch.Tensor(losses_q))

        if training:
            self.net_pi.train()
            loss, pred = self.net_pi(query_x, query_y)
            grads_pi = torch.autograd.grad(loss, self.net_pi.parameters(), create_graph=True, allow_unused=True)

        else:
            self.net_pi.eval()
            loss, pred = self.net_pi(query_x, query_y)
            grads_pi = None

        # # Compute the meta gradient and return it, the gradient is from one episode
        # # in metalearner, it will merge all loss from different episode and sum over it.
        # loss, pred = self.net_pi(query_x, query_y)
        # # pred: [setsz, n_way], indices: [setsz]
        # # _, indices = torch.max(pred, dim=1)
        # # correct = torch.eq(indices, query_y).sum().data[0]
        # # acc = correct / query_y.size(0)

        # # gradient for validation on theta_pi
        # # after call autorad.grad, you can not call backward again except for setting create_graph = True
        # # as we will use the loss as dummpy loss to conduct a dummy backprop to write our gradients to theta network,
        # # here we set create_graph to true to support second time backward.
        # grads_pi = torch.autograd.grad(loss, self.net_pi.parameters(), create_graph=True)

        # print("loss", loss)

        return loss, grads_pi, pred

    def net_forward(self, support_x, support_y):
        """
        This function is purely for updating net network. In metalearner, we need the get the loss op from net network
        to write our merged gradients into net network, hence will call this function to get a dummy loss op.
        :param support_x: [setsz, c, h, w]
        :param support_y: [sessz, c, h, w]
        :return: dummy loss and dummy pred
        """
        self.net.train()
        loss, pred = self.net(support_x, support_y)
        return loss, pred

    def get_pred_with_batch_size(self, query_x, query_y):
        batch_size = 512
        num_samples = len(query_x[0])
        if num_samples <= batch_size:
            loss, pred = self.net_pi(query_x, query_y)
            for item in pred:
                pred[item] = pred[item].cpu().data
            return loss, pred
        else:
            num_batches = int(np.ceil(num_samples / batch_size))
            all_pred = {}
            for batch in range(num_batches):
                start = batch * batch_size
                stop = int(min((batch + 1) * batch_size, num_samples))
                loss, pred = self.net_pi([item[start:stop] for item in query_x], {item:query_y[item][start:stop] for item in query_y})
                for item in pred:
                    if item not in all_pred:
                        all_pred[item] = []
                    all_pred[item].append(pred[item].cpu().data)
            for item in all_pred:
                all_pred[item] = torch.cat(all_pred[item])
            return None, all_pred
    
    def finetune_step_by_step(self, support_x, support_y, query_x, query_y, num_updates):
        self.update_pi()
        list_pred = []
        list_step = []
        record_step = list(range(10, 100, 10)) + \
            list(range(100, 1000, 100)) + \
            list(range(1000, 10001, 1000))

        with torch.no_grad():
            self.net_pi.eval()
            loss, pred = self.get_pred_with_batch_size(query_x, query_y)
            list_pred.append({'Affinity':pred['Affinity'].numpy()})
            list_step.append(0)
#             pcc = pearsonr(pred['Affinity'].cpu().data.numpy().reshape(-1), query_y['Affinity'].cpu().data.numpy().reshape(-1))[0]
#             print(0, loss.cpu().data.numpy(), pcc)

        for i in range(num_updates):
            self.net_pi.train()
            # forward and backward to update net_pi grad.
            loss, pred = self.net_pi(support_x, support_y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if (i+1) in record_step:
#                 print('step', i+1, '\r', end='')
                with torch.no_grad():
                    self.net_pi.eval()
                    loss, pred = self.get_pred_with_batch_size(query_x, query_y)
                    list_pred.append({'Affinity':pred['Affinity'].numpy()})
                    list_step.append(i+1)
#                     pcc = pearsonr(pred['Affinity'].cpu().data.numpy().reshape(-1), query_y['Affinity'].cpu().data.numpy().reshape(-1))[0]
#                     print(i+1, loss.cpu().data.numpy(), pcc)
        return list_pred, list_step

class Model(nn.Module):
    """
    As we have mentioned in Learner class, the metalearner class will receive a series of loss on different tasks/episodes
    on theta_pi network, and it will merage all loss and then sum over it. The summed loss will be backproped on theta
    network to update theta parameters, which is the initialization point we want to find.
    """

    def __init__(self, train_kwargs, model_kwargs, model):
        """
        :param net_cls: class, not instance. the class of specific Network for learner
        :param net_cls_args: tuple, args for net_cls, like (n_way, imgsz)
        :param n_way:
        :param k_shot:
        :param meta_batchsz: number of tasks/episode
        :param beta: learning rate for meta-learner
        :param num_updates: number of updates for learner
        """
        super(Model, self).__init__()

        self.train_kwargs = train_kwargs
        self.model_kwargs = model_kwargs
        
        self.num_updates = self.train_kwargs['num_updates']
        self.task = "Affinity"
        self.list_task = self.train_kwargs['list_task']
        self.task_eval = self.train_kwargs['task_eval']
        self.use_cuda = self.train_kwargs['use_cuda']

        # it will contains a learner class to learn on episodes and gather the loss together.
        self.learner = Learner(model, self.model_kwargs)
        # the optimizer is to update theta parameters, not theta_pi parameters.
        self.optimizer = optim.Adam(self.learner.parameters(), lr=float(self.train_kwargs['learning_rate']))

    def write_grads(self, dummy_loss, sum_grads_pi):
        """
        write loss into learner.net, gradients come from sum_grads_pi.
        Since the gradients info is not calculated by general backward, we need this function to write the right gradients
        into theta network and update theta parameters as wished.
        :param dummy_loss: dummy loss, nothing but to write our gradients by hook
        :param sum_grads_pi: the summed gradients
        :return:
        """

        # Register a hook on each parameter in the net that replaces the current dummy grad
        # with our grads accumulated across the meta-batch
        hooks = []

        for i, v in enumerate(self.learner.parameters()):
            def closure():
                ii = i
                return lambda grad: sum_grads_pi[ii]

            # if you write: hooks.append( v.register_hook(lambda grad : sum_grads_pi[i]) )
            # it will pop an ERROR, i don't know why?
            hooks.append(v.register_hook(closure()))

        # use our sumed gradients_pi to update the theta/net network,
        # since our optimizer receive the self.net.parameters() only.
        self.optimizer.zero_grad()
        dummy_loss.backward()
        self.optimizer.step()

        # if you do NOT remove the hook, the GPU memory will expode!!!
        for h in hooks:
            h.remove()
        
        return dummy_loss

    def get_data_batch(self, batch_items):
        device = next(self.learner.parameters()).device
        if self.use_cuda: 
            batch_items = [item.to(device) if item is not None and not isinstance(item, list) else \
                [it.to(device) for it in item] if isinstance(item, list) else \
                None for item in batch_items]

        return batch_items  

    def get_label_batch(self, batch_items):
        device = next(self.learner.parameters()).device
        if self.use_cuda: 
            for key in batch_items.keys():
                if key in self.list_task:
                    batch_items[key] = batch_items[key].to(device)

        return batch_items

    def get_results_template(self):
        results = {}
        for task in self.list_task:
            results[task] = {"pred":[], "label":[]}
        return results

    def forward(self, support_x, support_y, query_x, query_y):
        """
        Here we receive a series of episode, each episode will be learned by learner and get a loss on parameters theta.
        we gather the loss and sum all the loss and then update theta network.
        setsz = n_way * k_shotf
        querysz = n_way * k_shot
        :param support_x: [meta_batchsz, setsz, c_, h, w]
        :param support_y: [meta_batchsz, setsz]
        :param query_x:   [meta_batchsz, querysz, c_, h, w]
        :param query_y:   [meta_batchsz, querysz]
        :return:
        """
        sum_grads_pi = None
        dict_collect = self.get_results_template()

        support_x = self.get_data_batch(support_x)
        support_y = self.get_label_batch(support_y)
        query_x = self.get_data_batch(query_x)
        query_y = self.get_label_batch(query_y)
        # meta_batchsz = len(support_x)

        # support_x[i]: [setsz, c_, h, w]
        # we do different learning task sequentially, not parallel.
        # accs = []
        # for each task/episode.
        # for i in range(meta_batchsz):
        _, grad_pi, pred = self.learner(support_x, 
                                        support_y, 
                                        query_x, 
                                        query_y, 
                                        self.num_updates)
        # accs.append(episode_acc)
        if sum_grads_pi is None:
            sum_grads_pi = grad_pi
        else:  # accumulate all gradients from different episode learner
            sum_grads_pi = [torch.add(i, j) for i, j in zip(sum_grads_pi, grad_pi)]

        # As we already have the grads to update
        # We use a dummy forward / backward pass to get the correct grads into self.net
        # the right grads will be updated by hook, ignoring backward.
        # use hook mechnism to write sumed gradient into network.
        # we need to update the theta/net network, we need a op from net network, so we call self.learner.net_forward
        # to get the op from net network, since the loss from self.learner.forward will return loss from net_pi network.
        dummy_loss, _ = self.learner.net_forward(support_x, support_y)
        dummy_loss = self.write_grads(dummy_loss, sum_grads_pi)

        with torch.no_grad():
            dict_collect[self.task]["pred"].extend(pred[self.task].cpu().data.numpy())
            dict_collect[self.task]["label"].extend(query_y[self.task].cpu().data.numpy())

        return dummy_loss, dict_collect

    def pred(self, support_x, support_y, query_x, query_y, num_updates=None):
        """
        predict for query_x
        :param support_x:
        :param support_y:
        :param query_x:
        :param query_y:
        :return:
        """
        if num_updates is None:
            num_updates = self.num_updates
        dict_collect = self.get_results_template()

        support_x = self.get_data_batch(support_x)
        support_y = self.get_label_batch(support_y)
        query_x = self.get_data_batch(query_x)
        query_y = self.get_label_batch(query_y)

        # the learner will copy parameters from current theta network and then fine-tune on support set.
        _, _, pred = self.learner(support_x, 
                                  support_y, 
                                  query_x, 
                                  query_y, 
                                  num_updates, 
                                  training=False)

        with torch.no_grad():
            dict_collect[self.task]["pred"].extend(pred[self.task].cpu().data.numpy())
            dict_collect[self.task]["label"].extend(query_y[self.task].cpu().data.numpy())

        return dict_collect

    def pred_step_by_step(self, support_x, support_y, query_x, query_y, num_updates=None):
        """
        predict for query_x
        :param support_x:
        :param support_y:
        :param query_x:
        :param query_y:
        :return:
        """
        if num_updates is None:
            num_updates = self.num_updates
        dict_collect = self.get_results_template()

        support_x = self.get_data_batch(support_x)
        support_y = self.get_label_batch(support_y)
        query_x = self.get_data_batch(query_x)
        query_y = self.get_label_batch(query_y)

        # the learner will copy parameters from current theta network and then fine-tune on support set.
        list_pred, list_step = self.learner.finetune_step_by_step(support_x, 
                                                        support_y, 
                                                        query_x, 
                                                        query_y, 
                                                        num_updates)
        with torch.no_grad():
            pred = list_pred[-1]
            dict_collect[self.task]["pred"].extend(pred[self.task])
            dict_collect[self.task]["label"].extend(query_y[self.task].cpu().data.numpy())
            list_pred_array = []
            for pred in list_pred:
                list_pred_array.append(pred[self.task])
        return dict_collect, list_pred_array, list_step
