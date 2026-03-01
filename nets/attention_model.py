import torch
from torch import nn
import math
from nets.graph_encoder import GraphAttentionEncoder
from nets.robot_encoder import RobotAttentionEncoder
from nets.task_robot_attention import TaskRobotCrossAttention
from torch.nn import DataParallel
from utils.functions import sample_many
from problems.hrsp.paramet_hrsp import paramet_hrsp
import torch.nn.functional as F
from nets.robot_task_attention import AgentLearnTaskCrossAttention
from torch.utils.checkpoint import checkpoint


def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


class AttentionModel(nn.Module):
    def __init__(
            self,
            embedding_dim,
            hidden_dim,
            problem,
            n_encode_layers=2,
            tanh_clipping=10.,
            mask_inner=True,
            mask_logits=True,
            normalization='batch',
            n_heads=8,
            checkpoint_encoder=False,
            shrink_size=None
    ):
        super(AttentionModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0

        self.tanh_clipping = tanh_clipping
        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.problem = problem
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder  # Gradient checkpointing to save GPU memory
        self.shrink_size = shrink_size

        self.norm_query_task = nn.LayerNorm(embedding_dim)
        self.norm_glimpse_task = nn.LayerNorm(embedding_dim)

        self.norm_query_robot = nn.LayerNorm(embedding_dim)
        self.norm_glimpse_robot = nn.LayerNorm(embedding_dim)

        node_dim = 3 + paramet_hrsp.ROBOT_TYPE_NUM  # x, y, deadline, operation_time
        robot_dim = 3 + paramet_hrsp.ROBOT_TYPE_NUM  # x, y, cur_time, type(one_hot)

        self.robot_init_linear = nn.Linear(robot_dim, embedding_dim, bias=False)
        self.graph_init_linear = nn.Linear(node_dim, embedding_dim, bias=False)

        self.graph_embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization
        )

        self.robot_embedder = RobotAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=1,
            normalization='layer'
        )

        self.task_robot_cross = TaskRobotCrossAttention(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            normalization='layer'
        )

        self.robot_task_cross = AgentLearnTaskCrossAttention(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            normalization='layer'
        )

        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_robot_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_task_query = nn.Linear(2 * embedding_dim, embedding_dim, bias=False)

        assert embedding_dim % n_heads == 0
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_out_robot = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.project_robot_query = nn.Linear(3 * embedding_dim, embedding_dim, bias=False)

        self.empty_robots_embedding = nn.Parameter(torch.empty(embedding_dim))
        nn.init.xavier_uniform_(self.empty_robots_embedding.unsqueeze(0))

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:
            self.temp = temp

    def forward(self, input, return_pi=False):
        if self.checkpoint_encoder and self.training:
            node_embeddings, graph_embedding = checkpoint(self.graph_embedder, self.graph_init_embed(input),
                                                          use_reentrant=False)
        else:
            node_embeddings, graph_embedding = self.graph_embedder(self.graph_init_embed(input))

        _log_p, pi, cost, distance, tardiness = self._inner(input, node_embeddings)

        ll = _log_p.sum(1)
        if return_pi:
            return cost, ll, pi

        return cost, ll, distance, tardiness

    def _calc_log_likelihood(self, _log_p, a):
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)
        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"
        return log_p.sum(1)

    def graph_init_embed(self, input):
        max_deadline = input['deadline'].max()
        paramet_hrsp.time_norm = math.ceil(max_deadline)
        node_info = torch.cat((
            input['source'],
            input['deadline'][:, :, None] / paramet_hrsp.time_norm,
            input['operation_time']
        ), dim=-1)

        return self.graph_init_linear(node_info)

    def robot_init_embed(self, state):
        assert paramet_hrsp.time_norm is not None, 'paramet_hrsp.time_norm is None'
        robot_init_info = torch.cat((
            state.cur_coord,
            state.cur_time.unsqueeze(-1) / paramet_hrsp.time_norm,
            paramet_hrsp.robot_one_hot.unsqueeze(0).expand(state.cur_coord.size(0), -1, -1)
        ), -1)
        return self.robot_init_linear(robot_init_info)

    def get_robot_mask(self, input):
        required_robot = input['required_robot']
        required_status_for_each_robot = required_robot[:, :, paramet_hrsp.robot_type_indices]
        robot_type_mask = (required_status_for_each_robot == 0)
        return robot_type_mask

    def _inner(self, input, node_embeddings):
        outputs = []
        sequences = []
        state = self.problem.make_state(input)
        self.batch_idx = torch.arange(node_embeddings.size(0), device=node_embeddings.device)

        while not state.all_finished():
            if self.checkpoint_encoder and self.training:
                robot_embeddings, _ = checkpoint(self.robot_embedder, self.robot_init_embed(state), use_reentrant=False)
            else:
                robot_embeddings, _ = self.robot_embedder(self.robot_init_embed(state))

            # Task Context via Cross Attention
            if self.checkpoint_encoder and self.training:
                robot_ctx_node_embeddings, robot_ctx_node_mean = checkpoint(
                    self.task_robot_cross, node_embeddings, robot_embeddings, use_reentrant=False)
            else:
                robot_ctx_node_embeddings, robot_ctx_node_mean = self.task_robot_cross(node_embeddings,
                                                                                       robot_embeddings)

            # Robot Context via Cross Attention
            if self.checkpoint_encoder and self.training:
                task_ctx_robot_embeddings, task_ctx_robot_mean = checkpoint(
                    self.robot_task_cross, robot_embeddings, node_embeddings, state.get_mask(), use_reentrant=False)
            else:
                task_ctx_robot_embeddings, task_ctx_robot_mean = self.robot_task_cross(
                    robot_embeddings, node_embeddings, state.get_mask())

            # Task Selection
            task_log_p, mask = self._get_log_p_task(
                robot_ctx_node_embeddings, robot_ctx_node_mean, task_ctx_robot_mean, state)

            selected_task = self._select_node(task_log_p.exp()[:, 0, :], mask[:, 0, :])

            _task_log_p = task_log_p.squeeze(1)
            selected_task_log_p = torch.gather(_task_log_p, dim=1, index=selected_task.unsqueeze(-1))

            selected_task_node_emb = robot_ctx_node_embeddings[self.batch_idx, selected_task]

            # Robot Selection
            selected_robot_one_hot, robot_selected, robot_log_p = self.robot_decision(
                task_ctx_robot_embeddings, selected_task_node_emb)

            _robot_log_p = robot_log_p.squeeze(1)
            selected_robot_log_p = torch.gather(_robot_log_p, dim=1, index=robot_selected)

            sum_selected_robot_log_p = selected_robot_log_p.sum(dim=1).unsqueeze(1)
            log_p = selected_task_log_p + sum_selected_robot_log_p

            state = state.update(selected_task, selected_robot_one_hot)
            action = torch.cat([selected_task.unsqueeze(-1), robot_selected], dim=-1)

            outputs.append(log_p)
            sequences.append(action)

        sum_length = state.length.sum(dim=1)
        cost = sum_length.unsqueeze(-1) * paramet_hrsp.WEIGHT + state.tardiness * (1 - paramet_hrsp.WEIGHT)
        return torch.stack(outputs, 1), torch.stack(sequences, 1), cost, sum_length.unsqueeze(-1), state.tardiness

    def sample_many(self, _input, batch_rep=1, iter_rep=1):
        graph_embeddings = self.graph_embedder(self.graph_init_embed(_input))
        return sample_many(
            lambda x: self._inner(*x),
            (_input, graph_embeddings[0]),
            batch_rep, iter_rep
        )

    def _select_node(self, probs, mask):
        if self.decode_type == "greedy":
            _, selected = probs.max(1)
        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)
        else:
            assert False, "Unknown decode type"
        return selected

    def _select_robot(self, probs):
        assert (probs == probs).all(), "Probs should not contain any nans"
        if self.decode_type == "greedy":
            _, selected = probs.max(1)
        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)
        else:
            assert False, "Unknown decode type"
        return selected

    def _get_log_p_task(self, node_robot_embeddings, node_robot_mean, task_ctx_robot_mean, state, normalize=True):
        query_info = torch.concat([node_robot_mean, task_ctx_robot_mean], dim=-1)
        query = self.project_task_query(query_info)[:, None, :]
        query = self.norm_query_task(query + node_robot_mean.unsqueeze(1))

        glimpse_K, glimpse_V, logit_K = self.project_node_embeddings(node_robot_embeddings[:, None, :, :]).chunk(3,
                                                                                                                 dim=-1)

        glimpse_K = self._make_heads(glimpse_K, 1)
        glimpse_V = self._make_heads(glimpse_V, 1)
        mask = state.get_mask()
        log_p, _ = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)

        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)
        return log_p, mask

    def robot_decision(self, robot_embeddings, selected_task_node_emb):
        selected_robots_embedding_list = []
        selected_robot_one_hot = None
        robot_log_p = None
        robot_selected = None

        for robot_type_index in range(paramet_hrsp.ROBOT_TYPE_NUM):
            robot_type_log_p, robot_type_embeddings = self._get_log_p_robot(
                robot_embeddings, selected_task_node_emb, robot_type_index, selected_robots_embedding_list)

            selected_type_robot = self._select_robot(robot_type_log_p.exp()[:, 0, :])
            selected_type_emb = robot_type_embeddings[self.batch_idx, selected_type_robot]
            selected_robots_embedding_list.append(selected_type_emb[:, None, :])

            selected_type_robot_one_hot = F.one_hot(
                selected_type_robot, num_classes=paramet_hrsp.ROBOT_NUM_LIST[robot_type_index]).float()

            if selected_robot_one_hot is None:
                selected_robot_one_hot = selected_type_robot_one_hot
            else:
                selected_robot_one_hot = torch.cat([selected_robot_one_hot, selected_type_robot_one_hot], -1)

            if robot_log_p is None:
                robot_log_p = robot_type_log_p
            else:
                robot_log_p = torch.cat([robot_log_p, robot_type_log_p], -1)

            if robot_selected is None:
                robot_selected = selected_type_robot.unsqueeze(-1)
            else:
                offset = selected_type_robot + paramet_hrsp.ROBOT_NUM_LIST[:robot_type_index].sum()
                robot_selected = torch.cat([robot_selected, offset.unsqueeze(-1)], -1)

        return selected_robot_one_hot, robot_selected, robot_log_p

    def _get_log_p_robot(self, robot_embeddings, selected_task_emb, robot_type_idx, selected_robot_list,
                         normalize=True):
        start_idx = paramet_hrsp.ROBOT_NUM_LIST[:robot_type_idx].sum()
        end_idx = start_idx + paramet_hrsp.ROBOT_NUM_LIST[robot_type_idx]

        robot_type_embedding = robot_embeddings[:, start_idx: end_idx, :]
        robot_type_mean = robot_type_embedding.mean(1)

        if len(selected_robot_list) == 0:
            selected_robot_mean = self.empty_robots_embedding.expand(robot_type_embedding.size(0), -1)
        else:
            selected_robot_mean = torch.cat(selected_robot_list, dim=1).mean(dim=1)

        context_info = torch.cat((robot_type_mean, selected_robot_mean, selected_task_emb), dim=-1)
        query = self.project_robot_query(context_info).unsqueeze(1)
        query = self.norm_query_robot(query + robot_type_mean.unsqueeze(1))

        glimpse_K_, glimpse_V_, logit_K_ = self.project_robot_embeddings(robot_type_embedding[:, None, :, :]).chunk(3,
                                                                                                                    dim=-1)
        glimpse_K = self._make_heads(glimpse_K_, 1)
        glimpse_V = self._make_heads(glimpse_V_, 1)
        logit_K = logit_K_.contiguous()

        log_p, _ = self._one_to_many_logits_robot(query, glimpse_K, glimpse_V, logit_K)

        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()
        return log_p, robot_type_embedding

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):
        batch_size, num_steps, embed_dim = query.size()
        key_size = embed_dim // self.n_heads

        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))

        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf

        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.embedding_dim))

        glimpse = self.norm_glimpse_task(glimpse + query.unsqueeze(1))
        logits = torch.matmul(glimpse, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(glimpse.size(-1))

        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -math.inf

        return logits, glimpse.squeeze(-2)

    def _one_to_many_logits_robot(self, query, glimpse_K, glimpse_V, logit_K):
        batch_size, num_steps, embed_dim = query.size()
        key_size = embed_dim // self.n_heads

        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)

        glimpse = self.project_out_robot(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.embedding_dim))

        glimpse = self.norm_glimpse_robot(glimpse + query.unsqueeze(1))
        logits = torch.matmul(glimpse, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(glimpse.size(-1))

        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping

        return logits, glimpse.squeeze(-2)

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps
        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)
        )