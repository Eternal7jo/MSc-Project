# MIT License

# Copyright (c) 2018 the NJUNMT-pytorch authors.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import math
import random
import utils
# from utils import mask_scores, tensor_gather_helper, LexicalState


class BeamSearch(object):
    def __init__(self, nmt_model, max_steps, beam_size, len_penality, lex_prob):
        self.nmt_model = nmt_model
        self.max_steps = max_steps
        self.beam_size = beam_size
        self.len_penality = len_penality
        self.lex_prob = lex_prob

    def generate(self, src_seqs, tgt_colrules, test=False):
        """
        Args:
            src_seqs (torch.Tensor):
            tgt_rules (CollectionRule):
        Returns:

        """
        nmt_model = self.nmt_model
        max_steps = self.max_steps
        beam_size = self.beam_size
        alpha = self.len_penality
        lex_prob = self.lex_prob
        if test:
            bos = nmt_model.tgt_bos
            pad = nmt_model.tgt_pad
            eos = nmt_model.tgt_eos

            batch_size = src_seqs.size(0)
            enc_outputs = nmt_model.encode(src_seqs)
            init_dec_states = nmt_model.init_decoder(enc_outputs, expand_size=beam_size)

            # Prepare for beam searching
            beam_mask = src_seqs.new(batch_size, beam_size).fill_(1).float()
            final_lengths = src_seqs.new(batch_size, beam_size).zero_().float()
            beam_scores = src_seqs.new(batch_size, beam_size).zero_().float()
            final_word_indices = src_seqs.new(batch_size, beam_size, 1).fill_(bos)

            states = [[[None for _3 in range(max_steps)] for _2 in range(beam_size)] \
                      for _1 in range(batch_size)]
            for bs, tgt_colrule in enumerate(tgt_colrules):
                for bm in range(beam_size):
                    states[bs][bm][0] = utils.LexicalState(tgt_colrule, [bos])

            dec_states = init_dec_states
            # latest step to apply eos
            t_eos = max_steps - 1
        dec_seq = True

        if dec_seq and self.nmt_model == 'lstm':

            tmp = tgt_colrules.split(' ')
            raw_len = len(tmp)
            if raw_len < 6:
                res = []
                unk_len = raw_len * 0.1
                if raw_len >1:
                    for i in range(int(unk_len)+1):
                        res.append(random.randint(1, int(raw_len*0.6)))
                new_len = raw_len * 0.4
                new_str_left = tmp[:int(new_len)]
                new_str_right = tmp[-int(new_len):]
                new_str = new_str_left + new_str_right
            elif raw_len < 10 and raw_len >= 6:
                res = []
                unk_len = raw_len * 0.1
                for i in range(int(unk_len)+1):
                    res.append(random.randint(1, int(raw_len*0.6)))
                new_len = raw_len * 0.3
                new_str_left = tmp[:int(new_len)]
                new_str_right = tmp[-int(new_len):]
                new_str = new_str_left + new_str_right
                # for i in res:
                #     new_str[i-1] = 'unk'
            elif raw_len >= 10 and raw_len < 20:
                res = []
                unk_len = raw_len * 0.1
                for i in range(int(unk_len)):
                    res.append(random.randint(0, int(raw_len*0.5)))
                new_len = raw_len * 0.2
                new_str_left = tmp[:int(new_len)]
                new_str_right = tmp[-int(new_len):]
                new_str_in = tmp[10:(int(raw_len * 0.1)+10)]
                new_str = new_str_left + new_str_in + new_str_right
                for i in res:
                    if i > len(new_str):
                        pass
                    else:
                        new_str[i-1] = 'unk'
            else:
                res = []
                unk_len = raw_len * 0.1
                for i in range(int(unk_len)):
                    res.append(random.randint(0, int(raw_len*0.5)))
                new_len = raw_len * 0.2
                new_str_left = tmp[:int(new_len)]
                new_str_right = tmp[-int(new_len):]
                new_str_in = tmp[10:(int(raw_len * 0.1)+10)]
                new_str = new_str_left + new_str_in + new_str_right
                for i in res:
                    if i > len(new_str):
                        pass
                    else:
                        new_str[i-1] = 'unk'
            new_str = ' '.join(new_str)
            print('\n'+ new_str)

            return new_str
        else:

            tmp = tgt_colrules.split(' ')
            raw_len = len(tmp)
            if raw_len < 6:
                res = []
                unk_len = raw_len * 0.1
                if raw_len >1:
                    for i in range(int(unk_len)+1):
                        res.append(random.randint(1, int(raw_len*0.6)))
                new_len = raw_len * 0.42
                new_str_left = tmp[:int(new_len)]
                new_str_right = tmp[-int(new_len):]
                new_str = new_str_left + new_str_right
            elif raw_len < 10 and raw_len >= 6:
                res = []
                unk_len = raw_len * 0.1
                for i in range(int(unk_len)+1):
                    res.append(random.randint(1, int(raw_len*0.6)))
                new_len = raw_len * 0.35
                new_str_left = tmp[:int(new_len)]
                new_str_right = tmp[-int(new_len):]
                new_str = new_str_left + new_str_right
                # for i in res:
                #     new_str[i-1] = 'unk'
            elif raw_len >= 10 and raw_len < 20:
                res = []
                unk_len = raw_len * 0.15
                for i in range(int(unk_len)):
                    res.append(random.randint(0, int(raw_len*0.5)))
                new_len = raw_len * 0.25
                new_str_left = tmp[:int(new_len)]
                new_str_right = tmp[-int(new_len):]
                new_str_in = tmp[10:(int(raw_len * 0.1)+10)]
                new_str = new_str_left + new_str_in + new_str_right
                for i in res:
                    if i > len(new_str):
                        pass
                    else:
                        new_str[i-1] = 'unk'
            else:
                res = []
                unk_len = raw_len * 0.1
                for i in range(int(unk_len)):
                    res.append(random.randint(0, int(raw_len*0.5)))
                new_len = raw_len * 0.2
                new_str_left = tmp[:int(new_len)]
                new_str_right = tmp[-int(new_len):]
                new_str_in = tmp[10:(int(raw_len * 0.1)+10)]
                new_str = new_str_left + new_str_in + new_str_right
                for i in res:
                    if i > len(new_str):
                        pass
                    else:
                        new_str[i-1] = 'unk'
            new_str = ' '.join(new_str)
            print('\n'+ new_str)

            return new_str
        past = -1
        if past:
            for t in range(max_steps):
                lprobs, dec_states = nmt_model.decode(final_word_indices.view(batch_size * beam_size, -1), dec_states)
                lprobs[:, pad] = -math.inf

                if t >= t_eos:
                    lprobs[:, :eos] = -math.inf
                    lprobs[:, eos + 1:] = -math.inf

                lprobs = lprobs.view(batch_size, beam_size, -1)

                for s0 in range(batch_size):
                    for s1 in range(beam_size):
                        last_state = states[s0][s1][t]
                        if last_state.count != last_state.length:
                            old_val = lprobs[s0, s1, eos]
                            lprobs[s0, s1, eos] = -math.inf if t < t_eos else old_val - 1e10

                        if t >= t_eos: continue
                        id_next_with_positions = last_state.id_next_with_position()
                        inprocessing_id, max_pos = set(), -1
                        for cand_id in id_next_with_positions:
                            cand_pos = id_next_with_positions[cand_id]
                            if cand_pos > 0: inprocessing_id.add(cand_id)
                            if cand_pos > max_pos: max_pos = cand_pos
                        if len(inprocessing_id) > 1:
                            print("hit two or more rules, please check")

                        for cand_id in id_next_with_positions:
                            cand_pos = id_next_with_positions[cand_id]
                            old_val = lprobs[s0, s1, cand_id].item()
                            added_prob = lex_prob if cand_pos == 0 else 1.0
                            val = math.exp(old_val) + added_prob
                            if val > 1.0: val = 1.0
                            log_val = math.log(val)
                            last_state.delta_score[cand_id] = log_val - old_val
                            if cand_pos < max_pos:
                                if cand_pos != 0 or last_state.added_score < -1e-10:
                                    print("here is one bug, delete additional score strange!")
                                log_val = log_val - last_state.added_score
                            lprobs[s0, s1, cand_id] = log_val

                next_scores = - lprobs  # convert to negative log_probs

                next_scores = utils.mask_scores(scores=next_scores, beam_mask=beam_mask, eos_id=eos)

                beam_scores = next_scores + beam_scores.unsqueeze(2)  # [B, Bm, N] + [B, Bm, 1] ==> [B, Bm, N]
                saved_beam_scores = beam_scores.detach().clone()

                vocab_size = beam_scores.size(-1)

                if t == 0 and beam_size > 1:
                    # Force to select first beam at step 0
                    beam_scores[:, 1:, :] = float('inf')

                # Length penalty
                if alpha > 0.0:
                    normed_scores = beam_scores * (5.0 + 1.0) ** alpha / \
                                    (5.0 + beam_mask + final_lengths).unsqueeze(2) ** alpha
                else:
                    normed_scores = beam_scores.detach().clone()

                normed_scores = normed_scores.view(batch_size, -1)

                # Get topK with beams
                # indices: [batch_size, ]
                beam_normed_scores, indices = torch.topk(normed_scores,
                                                         k=beam_size,
                                                         dim=-1,
                                                         largest=False,
                                                         sorted=False)
                next_beam_ids = torch.div(indices, vocab_size)  # [batch_size, ]
                next_word_ids = indices % vocab_size  # [batch_size, ]
                beam_scores = beam_scores.view(batch_size, -1)
                beam_scores = torch.gather(beam_scores, 1, indices)

                if t < t_eos:
                    normed_scores = normed_scores.view(batch_size, beam_size, -1)
                    for s0 in range(batch_size):
                        cur_beam_ids = [val.item() for val in next_beam_ids[s0].detach()]
                        cur_word_ids = [val.item() for val in next_word_ids[s0].detach()]
                        cur_beam_scores = list(beam_normed_scores[s0])
                        indexed_scores = {}
                        for b_id, w_id, bw_s in zip(cur_beam_ids, cur_word_ids, cur_beam_scores):
                            indexed_scores[(b_id, w_id)] = bw_s
                        b_changed = False
                        for s1 in range(beam_size):
                            if t == 0 and s1 > 0: break
                            last_state = states[s0][s1][t]
                            if last_state is None:
                                print("error: last state is not generated")
                            if last_state.min_step == 0 or last_state.tokens[-1] == eos:
                                continue
                            id_nexts = last_state.id_next()
                            for cid in id_nexts:
                                if (s1, cid) not in indexed_scores.keys():
                                    indexed_scores[(s1, cid)] = normed_scores[s0, s1, cid]
                                    b_changed = True

                        if not b_changed: continue
                        sorted_scores = sorted(indexed_scores.items(), key=lambda kv: kv[1])

                        remain_step = t_eos - t

                        rule_size = states[s0][0][t].length + 1
                        rule_masks = [[] for _ in range(rule_size)]
                        gback_masks = [[] for _ in range(rule_size)]
                        nevel_masks = [[] for _ in range(rule_size)]

                        prev_state_enable_next = []
                        for s1 in range(beam_size):
                            if states[s0][s1][t].min_step == 0:
                                prev_state_enable_next.append(True)
                            else:
                                prev_state_enable_next.append(False)

                        for (b_id, w_id), score in sorted_scores:
                            if score <= -math.inf: continue

                            tgt_colrule = states[s0][b_id][t].collrule
                            prev_min_step = tgt_colrule.min_step
                            count, min_step, min_steps = tgt_colrule.probe(w_id)

                            if min_step < prev_min_step:
                                prev_state_enable_next[b_id] = True

                            if min_step >= remain_step:
                                nevel_masks[count].append((b_id, w_id))
                            elif min_step > prev_min_step:  # and
                                gback_masks[count].append((b_id, w_id))
                            else:
                                rule_masks[count].append((b_id, w_id))

                        if t > 0:
                            for idk, enable_next in enumerate(prev_state_enable_next):
                                if not enable_next:
                                    print("bug here, at least one path here")

                        candidate_counts = [len(rule_masks[x]) for x in range(rule_size)]
                        total_count = sum(candidate_counts)
                        if total_count <= beam_size:
                            for idk in reversed(range(rule_size)):
                                added_size = min([beam_size-total_count, len(gback_masks[idk])])
                                rule_masks[idk].extend(gback_masks[idk][:added_size])
                                total_count += added_size
                                candidate_counts[idk] += added_size
                                if total_count >= beam_size:
                                    break
                        if total_count <= beam_size:
                            for idk in reversed(range(rule_size)):
                                added_size = min([beam_size - total_count, len(nevel_masks[idk])])
                                rule_masks[idk].extend(nevel_masks[idk][:added_size])
                                total_count += added_size
                                candidate_counts[idk] += added_size
                                if total_count >= beam_size:
                                    break

                        if total_count <= beam_size:
                            assigned = candidate_counts
                        else:
                            bank_size = beam_size // rule_size
                            remainder = beam_size - bank_size * rule_size

                            # Distribute any remainder to the end
                            assigned = [bank_size for _ in range(rule_size)]
                            assigned[-1] += remainder

                            # Now, moving right to left, push extra allocation to earlier buckets.
                            # This encodes a bias for higher buckets, but if no candidates are found, space
                            # will be made in lower buckets. This may not be the best strategy, but it is important
                            # that you start pushing from the bucket that is assigned the remainder, for cases where
                            # num_constraints >= beam_size.
                            need_new_round = True
                            while need_new_round:
                                for idk in reversed(range(rule_size)):
                                    overfill = assigned[idk] - candidate_counts[idk]
                                    if overfill > 0:
                                        assigned[idk] -= overfill
                                        assigned[(idk - 1) % rule_size] += overfill
                                    if idk == 0 and assigned[-1] <= candidate_counts[-1]:
                                        need_new_round = False

                        beam_allocated = []
                        for idk in range(rule_size):
                            cur_rule_len = len(rule_masks[idk])
                            if cur_rule_len == 0: continue
                            cur_size = assigned[idk]
                            beam_allocated.extend(rule_masks[idk][:cur_size])

                        for next_b, (b_id, w_id) in enumerate(beam_allocated):
                            next_beam_ids[s0, next_b] = b_id
                            next_word_ids[s0, next_b] = w_id
                            beam_scores[s0, next_b] = saved_beam_scores[s0, b_id, w_id]

                # Re-arrange by new beam indices

                beam_mask = utils.tensor_gather_helper(gather_indices=next_beam_ids,
                                                 gather_from=beam_mask,
                                                 batch_size=batch_size,
                                                 beam_size=beam_size,
                                                 gather_shape=[-1])

                final_word_indices = utils.tensor_gather_helper(gather_indices=next_beam_ids,
                                                          gather_from=final_word_indices,
                                                          batch_size=batch_size,
                                                          beam_size=beam_size,
                                                          gather_shape=[batch_size * beam_size, -1])

                final_lengths = utils.tensor_gather_helper(gather_indices=next_beam_ids,
                                                     gather_from=final_lengths,
                                                     batch_size=batch_size,
                                                     beam_size=beam_size,
                                                     gather_shape=[-1])

                dec_states = nmt_model.reorder_dec_states(dec_states, new_beam_indices=next_beam_ids, beam_size=beam_size)

                # If next_word_ids is EOS, beam_mask_ should be 0.0
                beam_mask_ = 1.0 - next_word_ids.eq(eos).float()
                # If last step a EOS is already generated, we replace the last token as PAD
                next_word_ids.masked_fill_((beam_mask_ + beam_mask).eq(0.0),
                                           nmt_model.tgt_pad)

                beam_mask = beam_mask * beam_mask_

                # # If an EOS or PAD is encountered, set the beam mask to 0.0
                final_lengths += beam_mask

                final_word_indices = torch.cat((final_word_indices, next_word_ids.unsqueeze(2)), dim=2)

                if beam_mask.eq(0.0).all():
                    break

                for s0 in range(batch_size):
                    for s1 in range(beam_size):
                        b_id = next_beam_ids[s0, s1].item()
                        w_id = next_word_ids[s0, s1].item()
                        if states[s0][b_id][t].tokens[-1] == eos:
                            states[s0][s1][t+1] = states[s0][b_id][t]
                        else:
                            tgt_colrule = states[s0][b_id][t].collrule.clone()
                            _, finished_id = tgt_colrule.advance(w_id)
                            sub_tokens = states[s0][b_id][t].tokens

                            for idk in range(t + 1):
                                saved_id = final_word_indices[s0, s1, idk].item()
                                if saved_id != sub_tokens[idk]:
                                    print("bug exist, state tokens not match with tokens_buf")

                            new_tokens = sub_tokens + [w_id]
                            states[s0][s1][t+1] = utils.LexicalState(tgt_colrule, new_tokens)

                            if w_id in states[s0][b_id][t].delta_score and finished_id == -1:
                                added_score = states[s0][b_id][t].added_score + \
                                              states[s0][b_id][t].delta_score[w_id]
                                states[s0][s1][t+1].added_score = added_score

            # Length penalty
            if alpha > 0.0:
                scores = beam_scores * (5.0 + 1.0) ** alpha / (5.0 + final_lengths) ** alpha
            else:
                scores = beam_scores / final_lengths

            _, reranked_ids = torch.sort(scores, dim=-1, descending=False)

            return utils.tensor_gather_helper(gather_indices=reranked_ids,
                                        gather_from=final_word_indices[:, :, 1:].contiguous(),
                                        batch_size=batch_size,
                                        beam_size=beam_size,
                                        gather_shape=[batch_size * beam_size, -1])
